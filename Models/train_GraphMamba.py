import torch
import gymnasium as gym
from gymnasium_env_metro.environment import MetroSimulatorEnv
from model_trainer import A2CTrainer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time

from model.GraphMamba import GraphMambaModel


# --- Funkcja pomocnicza do tworzenia środowisk ---
def make_env():
    def _init():
        env = MetroSimulatorEnv()
        return env

    return _init
# ------------------------------------------------

if __name__ == "__main__":

    print("--- URUCHAMIANIE TESTU Z GRAPH MAMBA I RÓWNOLEGŁYMI ŚRODOWISKAMI ---")

    EXPERIMENT_NAME = "A2C_GraphMamba"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Używane urządzenie: {device} ---")

    NUM_ENVS = 7
    print(f"--- Uruchamianie {NUM_ENVS} równoległych środowisk ---")

    temp_env = MetroSimulatorEnv()
    num_node_features = temp_env.observation_space["node_features"].shape[1]
    num_stations = temp_env.observation_space["node_features"].shape[0]
    temp_env.close()

    vec_env = gym.vector.AsyncVectorEnv(
        [make_env() for _ in range(NUM_ENVS)]
    )

    model_to_train = GraphMambaModel(
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
        ppo_epochs=8,
        num_steps=512,
        batch_size=256,
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
    LOG_PATH.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(LOG_PATH)
    print(f"Uruchomiono logowanie TensorBoard. Logi w: {LOG_PATH}")
    print(f"Użyj: tensorboard --logdir logs")

    print("Rozpoczynanie treningu z ręczną pętlą (wektoryzacja)...")

    # --- [LOGIKA MROŻENIA] ---
    FREEZE_EPOCH = 1000  # Epoka, w której mrozimy enkodera
    is_frozen = False
    # ---

    # --- [EARLY STOPPING] ---
    best_avg_score = -float('inf')
    patience_counter = 0
    PATIENCE_EPOCHS = 500
    EPISODES_FOR_AVG = 100
    # ---

    for epoch in (pbar := tqdm(range(MAX_EPOCHS))):

        if not is_frozen and epoch >= FREEZE_EPOCH:
            a2c_system.model.freeze_encoder_layers()

            print("\n---  Mrożenie warstw enkodera. Tworzenie nowego optymalizatora... ---")
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, a2c_system.model.parameters()),
                lr=a2c_system.lr
            )
            is_frozen = True

        metrics = a2c_system.training_step(optimizer)

        for key, value in metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)

        pbar.set_description(
            f"Epoch {epoch} | Avg Score: {metrics['avg_episode_score']:.2f} | "
            f"Avg Week: {metrics['avg_episode_week']:.2f} | "
            f"Loss: {metrics['loss']:.4f}"
        )

        is_ready_to_check = metrics.get("episodes_in_window", 0) >= EPISODES_FOR_AVG

        if is_ready_to_check:
            current_score = metrics['avg_episode_score']

            if current_score > best_avg_score:
                best_avg_score = current_score
                patience_counter = 0
                print(f"\n✨ Nowy najlepszy wynik: {best_avg_score:.2f} w epoce {epoch}. Zapisywanie modelu...")
                torch.save(a2c_system.model.state_dict(), LOG_PATH / f"best_model.pth")

            else:
                patience_counter += 1

            if patience_counter >= PATIENCE_EPOCHS:
                print(f"\n--- EARLY STOPPING ---")
                print(f"Model nie poprawił wyniku {best_avg_score:.2f} przez {PATIENCE_EPOCHS} epok.")
                print(f"Zatrzymywanie treningu w epoce {epoch}.")
                break

    print("Trening zakończony.")
    writer.close()
    vec_env.close()