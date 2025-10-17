import os
import pytorch_lightning as pl
from gymnasium_env_metro.environment import MiniMetroEnv
from model_trainer import PPOTrainer

from model.GraphMamba import GraphMambaHFModel

if __name__ == "__main__":

    env = MiniMetroEnv()
    num_node_features = env.observation_space["node_features"].shape[1]
    num_stations = env.observation_space["node_features"].shape[0]

    MAMBA_MODEL_NAME = "state-spaces/mamba-130m-slimpj"

    #Fixed number don't touch
    HIDDEN_DIM = 768

    print(f"Tworzenie modelu GraphMambaHF z użyciem pre-trenowanego '{MAMBA_MODEL_NAME}'...")
    print("UWAGA: Przy pierwszym uruchomieniu nastąpi pobieranie modelu, co może potrwać kilka minut.")

    model_to_train = GraphMambaHFModel(
        num_node_features=num_node_features,
        hidden_dim=HIDDEN_DIM,
        num_stations=num_stations,
        mamba_model_name=MAMBA_MODEL_NAME,
        freeze_mamba=True
    )

    ppo_system = PPOTrainer(model=model_to_train, env=env, lr=3e-4)

    trainer = pl.Trainer(
        max_epochs=10000,
        accelerator="auto",
        logger=pl.loggers.TensorBoardLogger("logs/", name="GraphMambaHF")
    )

    print("Rozpoczynanie treningu z modelem GraphMambaHF...")
    trainer.fit(ppo_system)

    print("Trening zakończony.")
    env.close()