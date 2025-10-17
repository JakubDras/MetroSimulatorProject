import pytorch_lightning as pl
from gymnasium_env_metro.environment import MiniMetroEnv
from model_trainer import PPOTrainer

from model.GNNencoder import GNNModel

if __name__ == "__main__":

    env = MiniMetroEnv()

    num_node_features = env.observation_space["node_features"].shape[1]
    num_stations = env.observation_space["node_features"].shape[0]

    model_to_train = GNNModel(
        num_node_features=num_node_features,
        hidden_dim=128,
        num_stations=num_stations
    )

    ppo_system = PPOTrainer(model=model_to_train, env=env)

    trainer = pl.Trainer(
        max_epochs=10000,
        accelerator="auto",
        logger=pl.loggers.TensorBoardLogger("logs/"),
        # fast_dev_run=True
    )

    print("Rozpoczynanie treningu...")
    trainer.fit(ppo_system)

    print("Trening zakończony.")
    env.close()