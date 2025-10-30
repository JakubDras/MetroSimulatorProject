import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

from gymnasium_env_metro.environment import MiniMetroEnv


class PPOTrainer(pl.LightningModule):
    def __init__(self, model: nn.Module, env: MiniMetroEnv, lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_epsilon: float = 0.2, ppo_epochs: int = 10, rollout_len: int = 2048,
                 batch_size: int = 64, entropy_coef: float = 0.01):
        super().__init__()
        self.save_hyperparameters(ignore=['env', 'model'])
        self.model = model
        self.env = env
        self.automatic_optimization = False

    def forward(self, obs: dict) -> tuple[torch.Tensor, dict]:
        # Przekazuje wywołanie do właściwego modelu
        return self.model(obs, self.device)

    def get_action_and_value(self, obs: dict, masks: dict, action_dict: dict = None) -> tuple:
        """Uniwersalna metoda do próbkowania nowej akcji lub oceny istniejącej."""
        value, logits = self(obs)

        total_log_prob = 0
        total_entropy = 0

        # --- Akcja Wysokiego Poziomu ---
        hl_mask = torch.as_tensor(masks["high_level"], dtype=torch.bool, device=self.device)
        hl_logits_masked = logits["high_level"].masked_fill(~hl_mask, -float('inf'))
        hl_dist = Categorical(logits=hl_logits_masked)
        hl_action = torch.as_tensor([action_dict["high_level_action"]],
                                    device=self.device) if action_dict else hl_dist.sample()
        total_log_prob += hl_dist.log_prob(hl_action)
        total_entropy += hl_dist.entropy()

        # --- Akcje Niskiego Poziomu ---
        ll_params = np.zeros(3, dtype=int)
        ll_actions_tensor = torch.as_tensor(action_dict["low_level_params"],
                                            device=self.device) if action_dict else None
        action_item = hl_action.item()

        if action_item == 1:  # manage_line
            # Typ
            type_mask = torch.as_tensor(masks["manage_line_type"], dtype=torch.bool, device=self.device)
            type_logits = logits["manage_line_type"].masked_fill(~type_mask, -float('inf'))
            type_dist = Categorical(logits=type_logits)
            type_action = ll_actions_tensor[0] if action_dict else type_dist.sample()
            total_log_prob += type_dist.log_prob(type_action)
            total_entropy += type_dist.entropy()
            ll_params[0] = type_action.item()

            # Wybieramy stacje p1 i p2 tylko jeśli tworzymy/rozszerzamy linię (typ 0)

            """TODO: zrob to stopa do debugowania (red dote) i zobacz czemu ci wypierdala projekt"""
            if type_action.item() == 0:
                # P1
                p1_mask = torch.as_tensor(masks["manage_line"][0].any(axis=1), dtype=torch.bool,
                                          device=self.device)
                p1_logits_masked = logits["manage_line_p1"].masked_fill(~p1_mask, -float('inf'))
                p1_dist = Categorical(logits=p1_logits_masked)
                p1_action = ll_actions_tensor[1] if action_dict else p1_dist.sample()
                total_log_prob += p1_dist.log_prob(p1_action)
                total_entropy += p1_dist.entropy()
                ll_params[1] = p1_action.item()

                # P2
                p2_mask = torch.as_tensor(masks["manage_line"][0, p1_action.item()], dtype=torch.bool,
                                          device=self.device)
                p2_logits_masked = logits["manage_line_p2"].masked_fill(~p2_mask, -float('inf'))
                p2_dist = Categorical(logits=p2_logits_masked)
                p2_action = ll_actions_tensor[2] if action_dict else p2_dist.sample()
                total_log_prob += p2_dist.log_prob(p2_action)
                total_entropy += p2_dist.entropy()
                ll_params[2] = p2_action.item()

        elif action_item == 2:  # deploy_train
            station_action = ll_actions_tensor[1] if action_dict else None
            station_mask = torch.as_tensor(masks["deploy_train"], dtype=torch.bool, device=self.device)
            station_logits_masked = logits["deploy_train"].masked_fill(~station_mask, -float('inf'))
            station_dist = Categorical(logits=station_logits_masked)
            if station_action is None: station_action = station_dist.sample()
            total_log_prob += station_dist.log_prob(station_action)
            total_entropy += station_dist.entropy()
            ll_params[1] = station_action.item()

        elif action_item == 3:  # select_line
            color_action = ll_actions_tensor[1] if action_dict else None
            color_mask = torch.as_tensor(masks["select_line"], dtype=torch.bool, device=self.device)
            color_logits_masked = logits["select_line"].masked_fill(~color_mask, -float('inf'))
            color_dist = Categorical(logits=color_logits_masked)
            if color_action is None: color_action = color_dist.sample()
            total_log_prob += color_dist.log_prob(color_action)
            total_entropy += color_dist.entropy()
            ll_params[1] = color_action.item()

        if action_dict is None:
            action_dict = {"high_level_action": hl_action.item(), "low_level_params": ll_params}

        return action_dict, total_log_prob, value, total_entropy

    @torch.no_grad()
    def _collect_rollouts(self) -> dict:
        observations, actions, log_probs, rewards, dones, values, masks = [], [], [], [], [], [], []
        if not hasattr(self, 'current_obs'):
            self.current_obs, _ = self.env.reset()

        for _ in range(self.hparams.rollout_len):
            current_masks = self.env._get_action_masks()
            action_dict, log_prob, value, _ = self.get_action_and_value(self.current_obs, current_masks)

            next_obs, reward, terminated, truncated, _ = self.env.step(action_dict)
            done = terminated or truncated

            observations.append(self.current_obs)
            actions.append(action_dict)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value.squeeze())
            masks.append(current_masks)  # ZAPISUJEMY MASKI

            self.current_obs = next_obs
            if done: self.current_obs, _ = self.env.reset()

        return {"observations": observations, "actions": actions, "log_probs": torch.stack(log_probs),
                "rewards": torch.tensor(rewards, dtype=torch.float32, device=self.device),
                "dones": torch.tensor(dones, dtype=torch.float32, device=self.device),
                "values": torch.stack(values),
                "masks": masks}  # ZWRACAMY ZAPISANE MASKI

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        rollout_data = self._collect_rollouts()

        advantages = torch.zeros(self.hparams.rollout_len, device=self.device)
        last_advantage = 0
        with torch.no_grad():
            last_value, _ = self(self.current_obs)
            last_value = last_value.squeeze()

        for t in reversed(range(self.hparams.rollout_len)):
            next_non_terminal = 1.0 - rollout_data["dones"][t + 1] if t < self.hparams.rollout_len - 1 else 1.0
            next_value = rollout_data["values"][t + 1] if t < self.hparams.rollout_len - 1 else last_value
            delta = rollout_data["rewards"][t] + self.hparams.gamma * next_value * next_non_terminal - \
                    rollout_data["values"][t]
            advantages[
                t] = last_advantage = delta + self.hparams.gamma * self.hparams.gae_lambda * next_non_terminal * last_advantage

        returns = advantages + rollout_data["values"]

        indices = np.arange(self.hparams.rollout_len)

        for _ in range(self.hparams.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, self.hparams.rollout_len, self.hparams.batch_size):
                end = start + self.hparams.batch_size
                minibatch_indices = indices[start:end]

                # --- NOWA, CZYSTA I POPRAWNA LOGIKA MINIBATCHA ---
                new_log_probs_list, new_values_list, entropy_list = [], [], []
                for idx in minibatch_indices:
                    obs = rollout_data["observations"][idx]
                    action = rollout_data["actions"][idx]
                    masks = rollout_data["masks"][idx]  # Używamy zapisanej, historycznej maski

                    _, log_prob, value, entropy = self.get_action_and_value(obs, masks, action_dict=action)

                    new_log_probs_list.append(log_prob)
                    new_values_list.append(value)
                    entropy_list.append(entropy)

                new_log_probs = torch.stack(new_log_probs_list).squeeze()
                new_values = torch.stack(new_values_list).squeeze()
                entropy = torch.stack(entropy_list).mean()

                old_log_probs_batch = rollout_data["log_probs"][minibatch_indices]
                advantages_batch = advantages[minibatch_indices]
                returns_batch = returns[minibatch_indices]

                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.hparams.clip_epsilon,
                                    1 + self.hparams.clip_epsilon) * advantages_batch

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, returns_batch)

                loss = policy_loss + 0.5 * value_loss - self.hparams.entropy_coef * entropy

                optimizer.zero_grad()
                self.manual_backward(loss)
                optimizer.step()

        self.log_dict({"train_loss": loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item()})

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return DataLoader(range(1))