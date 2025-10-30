import pytorch_lightning as pl
from gymnasium_env_metro.environment import MiniMetroEnv
from model_trainer import PPOTrainer

# Krok 1: Importujemy nasz nowy, prosty model MLPModel
from model.MLPtest import MLPModel

if __name__ == "__main__":

    print("--- URUCHAMIANIE TESTU Z PROSTYM MODELEM MLP ---")

    env = MiniMetroEnv()

    # Pobieramy wymiary ze środowiska (tak jak wcześniej)
    num_node_features = env.observation_space["node_features"].shape[1]
    num_stations = env.observation_space["node_features"].shape[0]

    # Krok 2: Tworzymy instancję MLPModel (zamiast GNNModel)
    # Zauważ, że przekazujemy te same parametry - model jest kompatybilny
    model_to_train = MLPModel(
        num_node_features=num_node_features,
        hidden_dim=128,
        num_stations=num_stations
    )

    # Krok 3: Przekazujemy model testowy do tego samego, starego PPOTrainer
    # PPOTrainer jest uniwersalny i nie wie, czy w środku jest GNN czy MLP
    ppo_system = PPOTrainer(model=model_to_train, env=env)

    trainer = pl.Trainer(
        max_epochs=10000,
        accelerator="auto",
        logger=pl.loggers.TensorBoardLogger("logs/"),
        # Użyj fast_dev_run=True, aby szybko sprawdzić, czy działa 1 cykl
        # fast_dev_run=True
    )

    print("Rozpoczynanie treningu z modelem MLP...")
    trainer.fit(ppo_system)

    print("Trening zakończony.")
    env.close()