import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
import collections


"""tensorboard --logdir logs"""

class A2CTrainer(torch.nn.Module):

    def __init__(self, model: nn.Module, vec_env: gym.vector.AsyncVectorEnv,
                 device: torch.device, num_envs: int,
                 lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 ppo_epochs: int = 10, num_steps: int = 256,
                 batch_size: int = 64, entropy_coef: float = 0.01):

        super().__init__()

        self.model = model
        self.vec_env = vec_env
        self.device = device
        self.num_envs = num_envs

        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef

        self.global_step = 0

        self.episode_score_deque = collections.deque(maxlen=100)
        self.episode_week_deque = collections.deque(maxlen=100)

        self.episode_truncated_deque = collections.deque(maxlen=100)
        self.episode_terminated_deque = collections.deque(maxlen=100)

        self.episode_score_acc = np.zeros(num_envs, dtype=np.float32)
        self.episode_week_acc = np.zeros(num_envs, dtype=np.float32)

        init_obs_cpu, init_info_cpu = self.vec_env.reset(seed=42)
        # init_obs_cpu, init_info_cpu = self.vec_env.reset(seed=np.random.randint(0, 100000))
        self.current_obs_gpu = self._obs_to_gpu(init_obs_cpu)

        self.current_masks_cpu = init_obs_cpu["action_masks"]

        self.current_dones = torch.zeros(self.num_envs, device=self.device)

    def _obs_to_gpu(self, obs_cpu: dict) -> dict:
        obs_gpu = {}
        for k, v in obs_cpu.items():
            if k == "action_masks": continue
            obs_gpu[k] = torch.as_tensor(v, device=self.device)
        return obs_gpu

    def forward(self, obs_gpu: dict) -> tuple[torch.Tensor, dict]:
        return self.model(obs_gpu, self.device)

    @torch.no_grad()
    def get_actions_and_values(self, obs_gpu: dict, masks_cpu: dict) -> tuple:
        value_batch, logits_batch = self(obs_gpu)

        masks_gpu = {
            k: torch.as_tensor(v, dtype=torch.bool, device=self.device)
            for k, v in masks_cpu.items()
        }

        all_hl_actions = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        all_ll_params = torch.zeros((self.num_envs, 3), dtype=torch.long, device=self.device)
        all_log_probs = torch.zeros(self.num_envs, device=self.device)
        all_entropies = torch.zeros(self.num_envs, device=self.device)

        for i in range(self.num_envs):
            hl_mask = masks_gpu["high_level"][i]
            hl_logits_masked = logits_batch["high_level"][i].masked_fill(~hl_mask, -float('inf'))
            hl_dist = Categorical(logits=hl_logits_masked)
            hl_action = hl_dist.sample()

            all_hl_actions[i] = hl_action
            log_prob = hl_dist.log_prob(hl_action)
            entropy = hl_dist.entropy()

            ll_params = torch.zeros(3, dtype=torch.long, device=self.device)
            action_item = hl_action.item()

            if action_item == 1:
                type_mask = masks_gpu["manage_line_type"][i]
                type_logits = logits_batch["manage_line_type"][i].masked_fill(~type_mask, -float('inf'))
                type_dist = Categorical(logits=type_logits)
                type_action = type_dist.sample()

                log_prob += type_dist.log_prob(type_action)
                entropy += type_dist.entropy()
                ll_params[0] = type_action

                if type_action.item() == 0:
                    p1_mask = masks_gpu["manage_line"][i, 0].any(dim=1)
                    p1_logits = logits_batch["manage_line_p1"][i].masked_fill(~p1_mask, -float('inf'))
                    p1_dist = Categorical(logits=p1_logits)
                    p1_action = p1_dist.sample()

                    log_prob += p1_dist.log_prob(p1_action)
                    entropy += p1_dist.entropy()
                    ll_params[1] = p1_action

                    p2_mask = masks_gpu["manage_line"][i, 0, p1_action.item()]
                    p2_logits = logits_batch["manage_line_p2"][i].masked_fill(~p2_mask, -float('inf'))
                    p2_dist = Categorical(logits=p2_logits)
                    p2_action = p2_dist.sample()

                    log_prob += p2_dist.log_prob(p2_action)
                    entropy += p2_dist.entropy()
                    ll_params[2] = p2_action

            elif action_item == 2:
                mask = masks_gpu["deploy_train"][i]
                logits = logits_batch["deploy_train"][i].masked_fill(~mask, -float('inf'))
                dist = Categorical(logits=logits)
                action = dist.sample()

                log_prob += dist.log_prob(action)
                entropy += dist.entropy()
                ll_params[1] = action

            elif action_item == 3:
                mask = masks_gpu["select_line"][i]
                logits = logits_batch["select_line"][i].masked_fill(~mask, -float('inf'))
                dist = Categorical(logits=logits)
                action = dist.sample()

                log_prob += dist.log_prob(action)
                entropy += dist.entropy()
                ll_params[1] = action

            all_ll_params[i] = ll_params
            all_log_probs[i] = log_prob
            all_entropies[i] = entropy

        actions_for_env = {
            "high_level_action": all_hl_actions.cpu().numpy(),
            "low_level_params": all_ll_params.cpu().numpy()
        }

        return actions_for_env, all_log_probs, value_batch.flatten(), all_entropies

    @torch.no_grad()
    def _collect_rollouts(self) -> dict:
        buf_log_probs = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        buf_rewards = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        buf_dones = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        buf_values = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        buf_entropies = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        buf_obs = {}
        for k, v in self.current_obs_gpu.items():
            buf_obs[k] = torch.zeros((self.num_steps,) + v.shape, device=self.device)
        buf_masks = {}
        for k, v in self.current_masks_cpu.items():
            buf_masks[k] = np.zeros((self.num_steps, self.num_envs) + v.shape[1:], dtype=v.dtype)

        for step in range(self.num_steps):
            for k, v in self.current_obs_gpu.items():
                buf_obs[k][step] = v
            buf_dones[step] = self.current_dones
            for k in buf_masks:
                buf_masks[k][step] = self.current_masks_cpu[k]

            actions_list_cpu, log_probs, values, entropies = self.get_actions_and_values(
                self.current_obs_gpu,
                {k: v[step] for k, v in buf_masks.items()}
            )

            buf_log_probs[step] = log_probs
            buf_values[step] = values
            buf_entropies[step] = entropies

            next_obs_cpu, rewards_cpu, terminated_cpu, truncated_cpu, info = self.vec_env.step(actions_list_cpu)

            self.current_obs_gpu = self._obs_to_gpu(next_obs_cpu)
            buf_rewards[step] = torch.as_tensor(rewards_cpu, dtype=torch.float32, device=self.device)
            self.current_dones = torch.as_tensor(terminated_cpu, dtype=torch.float32, device=self.device)
            self.current_masks_cpu = next_obs_cpu["action_masks"]

            self.episode_score_acc += rewards_cpu
            self.episode_week_acc += (terminated_cpu | truncated_cpu)

            finished_cpu = np.logical_or(terminated_cpu, truncated_cpu)

            if np.any(finished_cpu):
                for i, finished in enumerate(finished_cpu):
                    if finished:
                        self.episode_score_deque.append(self.episode_score_acc[i])

                        self.episode_truncated_deque.append(1.0 if truncated_cpu[i] else 0.0)
                        self.episode_terminated_deque.append(1.0 if terminated_cpu[i] else 0.0)

                        self.episode_week_deque.append(1.0)

                        self.episode_score_acc[i] = 0.0
                        self.episode_week_acc[i] = 0.0

        with torch.no_grad():
            next_value, _ = self(self.current_obs_gpu)
            next_value = next_value.flatten()
            advantages = torch.zeros_like(buf_rewards)
            last_gae_lam = 0
            for t in reversed(range(self.num_steps)):
                next_non_terminal = 1.0 - buf_dones[t + 1] if t < self.num_steps - 1 else 1.0 - self.current_dones
                next_values_step = buf_values[t + 1] if t < self.num_steps - 1 else next_value
                delta = buf_rewards[t] + self.gamma * next_values_step * next_non_terminal - buf_values[t]
                advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            returns = advantages + buf_values

        flat_obs = {}
        for k, v in buf_obs.items():
            flat_obs[k] = v.reshape((-1,) + v.shape[2:])
        flat_masks = {}
        for k, v in buf_masks.items():
            flat_masks[k] = v.reshape((-1,) + v.shape[2:])
        flat_log_probs = buf_log_probs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_entropies = buf_entropies.reshape(-1)
        self.last_avg_reward = buf_rewards.mean().item()
        self.last_total_reward = buf_rewards.sum().item()

        return flat_obs, flat_masks, flat_log_probs, flat_advantages, flat_returns, flat_entropies

    def training_step(self, optimizer: torch.optim.Optimizer):

        obs, masks, old_log_probs, advantages, returns, entropies = self._collect_rollouts()

        num_samples = self.num_steps * self.num_envs
        indices = np.arange(num_samples)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        last_loss = 0.0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                minibatch_indices = indices[start:end]

                mb_obs = {k: v[minibatch_indices] for k, v in obs.items()}
                mb_advantages = advantages[minibatch_indices]
                mb_returns = returns[minibatch_indices]
                mb_old_log_probs = old_log_probs[minibatch_indices]
                mb_old_entropies = entropies[minibatch_indices]

                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                new_values, _ = self(mb_obs)
                new_values = new_values.flatten()

                value_loss = F.mse_loss(new_values, mb_returns)
                policy_loss = -(mb_old_log_probs * mb_advantages).mean()
                entropy_loss = -mb_old_entropies.mean()

                loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                last_loss = loss.item()

        self.global_step += num_samples

        num_updates = self.ppo_epochs * (num_samples // self.batch_size)
        if num_updates == 0: num_updates = 1

        avg_score = np.mean(self.episode_score_deque) if self.episode_score_deque else 0.0
        avg_week = np.mean(self.episode_week_deque) if self.episode_week_deque else 0.0

        avg_truncated = np.mean(self.episode_truncated_deque) if self.episode_truncated_deque else 0.0
        avg_terminated = np.mean(self.episode_terminated_deque) if self.episode_terminated_deque else 0.0

        return {
            "avg_reward_per_step": self.last_avg_reward,
            "loss": last_loss,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy_loss": total_entropy_loss / num_updates,
            "avg_episode_score": avg_score,
            "avg_episode_week": avg_week,
            "episodes_in_window": len(self.episode_score_deque),
            "avg_ep_truncated": avg_truncated,
            "avg_ep_terminated": avg_terminated,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)