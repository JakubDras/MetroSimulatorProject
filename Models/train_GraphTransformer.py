import os
import pytorch_lightning as pl
from gymnasium_env_metro.environment import MiniMetroEnv
from model_trainer import PPOTrainer

from model.GraphTransformer import GraphTransformerModel

if __name__ == "__main__":

    env = MiniMetroEnv()

    num_node_features = env.observation_space["node_features"].shape[1]
    num_stations = env.observation_space["node_features"].shape[0]

    model_to_train = GraphTransformerModel(
        num_node_features=num_node_features,
        hidden_dim=32,
        num_stations=num_stations,
        heads=4
    )

    ppo_system = PPOTrainer(model=model_to_train, env=env, lr=3e-4)

    trainer = pl.Trainer(
        max_epochs=10000,
        accelerator="auto",
        logger=pl.loggers.TensorBoardLogger("logs/")
    )

    print("Rozpoczynanie treningu z modelem Graph Transformer...")
    trainer.fit(ppo_system)

    print("Trening zakończony.")
    env.close()