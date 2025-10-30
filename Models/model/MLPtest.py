# plik: model/MLPtest.py
# Prosty model testowy (MLP), który ignoruje strukturę grafu.

import torch
import torch.nn as nn
import gymnasium_env_metro.config as config


class MLPModel(nn.Module):
    """
    Prosty model testowy (MLP), który spłaszcza obserwacje.
    Ma taki sam "interfejs" (te same głowice wyjściowe) co GNNModel,
    więc można go podłączyć do tego samego trenera.
    """

    def __init__(self, num_node_features: int, hidden_dim: int, num_stations: int):
        super().__init__()
        num_line_colors = len(config.LINE_COLORS)

        # Obliczamy rozmiar "spłaszczonego" wektora wejściowego:
        # 4 cechy globalne + (Liczba stacji * Liczba cech na stację)
        # UWAGA: num_stations to MAX_STATIONS z configu
        self.input_dim = 4 + (num_stations * num_node_features)

        print(f"--- INFO: Uruchamiam model testowy MLP z wejściem o rozmiarze {self.input_dim} ---")

        # Wspólna sieć (zamiast GNN)
        self.shared_network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Głowice Aktora i Krytyka (DOKŁADNIE takie same jak w GNNModel)
        self.critic_head = nn.Linear(hidden_dim, 1)
        self.high_level_head = nn.Linear(hidden_dim, 4)
        self.manage_line_type_head = nn.Linear(hidden_dim, 3)
        self.manage_line_p1_head = nn.Linear(hidden_dim, num_stations)
        self.manage_line_p2_head = nn.Linear(hidden_dim, num_stations)
        self.deploy_train_head = nn.Linear(hidden_dim, num_stations)
        self.select_line_head = nn.Linear(hidden_dim, num_line_colors)

    def forward(self, obs: dict, device: str) -> tuple[torch.Tensor, dict]:

        # 1. Pobieramy tensory z obserwacji
        global_features = torch.as_tensor(obs["global_features"], dtype=torch.float32, device=device)
        node_features = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)

        # 2. Spłaszczamy tensory do jednego, długiego wektora
        node_features_flat = node_features.flatten()

        # === POCZĄTEK ZMIANY: Poprawione debugowanie ===
        if global_features.dim() == 0:
            global_features = global_features.unsqueeze(0)
        if node_features_flat.dim() == 0:
            node_features_flat = node_features_flat.unsqueeze(0)

        try:
            flat_obs = torch.cat([global_features, node_features_flat])
        except RuntimeError as e:
            print(f"BŁĄD CAT w MLPModel: global={global_features.shape}, node_flat={node_features_flat.shape}",
                  flush=True)
            raise e

        flat_obs = flat_obs.unsqueeze(
            0)  # unsqueeze(0) dodaje wymiar batcha [1, input_dim]
        # === KONIEC ZMIANY ===

        # 3. Przepuszczamy przez wspólną sieć
        if torch.isnan(flat_obs).any() or torch.isinf(flat_obs).any():
            print("!!! KRYTYCZNY BŁĄD: NaN lub Inf na WEJŚCIU do modelu MLP (w obs) !!!", flush=True)
            print(f"flat_obs: {flat_obs}", flush=True)

        shared_embedding = self.shared_network(flat_obs)

        if torch.isnan(shared_embedding).any() or torch.isinf(shared_embedding).any():
            print("!!! KRYTYCZNY BŁĄD: NaN lub Inf na WYJŚCIU z modelu MLP (po shared_network) !!!", flush=True)
            print(f"shared_embedding: {shared_embedding}", flush=True)

        # 4. Obliczamy wyjścia (tak samo jak w GNN)
        value = self.critic_head(shared_embedding)
        logits = {
            "high_level": self.high_level_head(shared_embedding),
            "manage_line_type": self.manage_line_type_head(shared_embedding),
            "manage_line_p1": self.manage_line_p1_head(shared_embedding),
            "manage_line_p2": self.manage_line_p2_head(shared_embedding),
            "deploy_train": self.deploy_train_head(shared_embedding),
            "select_line": self.select_line_head(shared_embedding)
        }
        return value, logits