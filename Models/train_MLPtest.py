import torch
import gymnasium as gym
from gymnasium_env_metro.environment import MiniMetroEnv
from model_trainer import A2CTrainer
from model.MLPtest import MLPModel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import time

# --- Funkcja pomocnicza do tworzenia środowisk ---
def make_env():
    def _init():
        env = MiniMetroEnv()
        return env

    return _init
# ------------------------------------------------

if __name__ == "__main__":

    print("--- URUCHAMIANIE TESTU Z RÓWNOLEGŁYMI ŚRODOWISKAMI ---")

    EXPERIMENT_NAME = "A2C_MLP_test_final"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Używane urządzenie: {device} ---")

    NUM_ENVS = 8
    print(f"--- Uruchamianie {NUM_ENVS} równoległych środowisk ---")

    temp_env = MiniMetroEnv()
    num_node_features = temp_env.observation_space["node_features"].shape[1]
    num_stations = temp_env.observation_space["node_features"].shape[0]
    temp_env.close()

    vec_env = gym.vector.AsyncVectorEnv(
        [make_env() for _ in range(NUM_ENVS)]
    )

    model_to_train = MLPModel(
        num_node_features=num_node_features,
        hidden_dim=128,
        num_stations=num_stations
    )

    a2c_system = A2CTrainer(
        model=model_to_train,
        vec_env=vec_env,
        device=device,
        num_envs=NUM_ENVS,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=20,
        num_steps=128,
        batch_size=128,
        entropy_coef=0.01
    )

    a2c_system.to(device)
    optimizer = a2c_system.configure_optimizers()
    a2c_system.model.train()

    MAX_EPOCHS = 10000

    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    RUN_NAME = f"{EXPERIMENT_NAME}_{current_time}"

    LOG_PATH = PROJECT_ROOT / "logs" / RUN_NAME

    writer = SummaryWriter(LOG_PATH)
    print(f"Uruchomiono logowanie TensorBoard. Logi w: {LOG_PATH}")
    print(f"Użyj: tensorboard --logdir logs")

    print("Rozpoczynanie treningu z ręczną pętlą (wektoryzacja)...")

    for epoch in (pbar := tqdm(range(MAX_EPOCHS))):

        metrics = a2c_system.training_step(optimizer)

        for key, value in metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)

        pbar.set_description(
            f"Epoch {epoch} | Avg Score: {metrics['avg_episode_score']:.2f} | "
            f"Avg Week: {metrics['avg_episode_week']:.2f} | "
            f"Loss: {metrics['loss']:.4f}"
        )

    print("Trening zakończony.")
    writer.close()
    vec_env.close()