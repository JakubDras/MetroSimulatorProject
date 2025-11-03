import os
import pytorch_lightning as pl
from gymnasium_env_metro.environment import MiniMetroEnv
import gymnasium_env_metro.config as config
from model_trainer import A2CTrainer

from model.GraphMamba import GraphMamba

if __name__ == "__main__":

    env = MiniMetroEnv()

    num_node_features = env.observation_space["node_features"].shape[1]
    num_stations = env.observation_space["node_features"].shape[0]
    num_line_colors = len(config.LINE_COLORS)

    model_to_train = GraphMamba(
        num_node_features=num_node_features,
        hidden_dim=32,
        num_stations=num_stations,
        num_line_colors=num_line_colors
    )

    ppo_system = A2CTrainer(model=model_to_train, env=env, lr=3e-4)

    trainer = pl.Trainer(
        max_epochs=10000,
        accelerator="auto",
        logger=pl.loggers.TensorBoardLogger("logs/", name="GraphMamba")
    )

    print("Rozpoczynanie treningu z modelem Graph Mamba...")
    trainer.fit(ppo_system)

    print("Trening zakończony.")
    env.close()