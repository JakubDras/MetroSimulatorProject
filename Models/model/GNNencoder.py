# plik: model/GNNencoder.py
# WERSJA Z EFEKTYWNYM I ELEGANCKIM ENKODOWANIEM

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

import gymnasium_env_metro.config as config


class GNNModel(nn.Module):
    """
    Definicja architektury sieci opartej na GNN.
    Używa projekcji cech węzłów, a następnie wzbogacenia grafowego (GNN).
    """

    def __init__(self, num_node_features: int, hidden_dim: int, num_stations: int):
        super().__init__()
        num_line_colors = len(config.LINE_COLORS)

        # Krok 1: Warstwa projekcji cech węzłów (zawsze aktywna)
        # Przekształca surowe cechy [num_node_features] -> [hidden_dim]
        self.initial_projection = nn.Linear(num_node_features, hidden_dim)

        # Krok 2: Warstwy GNN (propagacja informacji)
        # Teraz działają na już przekształconych cechach [hidden_dim] -> [hidden_dim]
        self.encoder_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.encoder_conv2 = GCNConv(hidden_dim, hidden_dim)

        # Głowice Aktora i Krytyka pozostają bez zmian
        self.critic_head = nn.Linear(hidden_dim, 1)
        self.high_level_head = nn.Linear(hidden_dim, 4)
        self.manage_line_type_head = nn.Linear(hidden_dim, 3)
        self.manage_line_p1_head = nn.Linear(hidden_dim, num_stations)
        self.manage_line_p2_head = nn.Linear(hidden_dim, num_stations)
        self.deploy_train_head = nn.Linear(hidden_dim, num_stations)
        self.select_line_head = nn.Linear(hidden_dim, num_line_colors)

    def encode(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Przetwarza cechy węzłów, a następnie opcjonalnie wzbogaca je o informacje z grafu.
        """
        # Krok 1: ZAWSZE wykonaj projekcję cech dla każdego węzła niezależnie.
        # To jest nasza bazowa reprezentacja węzłów.
        h = self.initial_projection(node_features).relu()

        # Krok 2: Jeśli graf ma krawędzie, użyj GNN, aby "wymieszać" informacje
        # między sąsiednimi węzłami i wzbogacić ich reprezentacje.
        if edge_index.shape[1] > 0:
            h = self.encoder_conv1(h, edge_index).relu()
            h = self.encoder_conv2(h, edge_index).relu()

        # Jeśli nie ma krawędzi, po prostu zwracamy reprezentację z Kroku 1.
        # To naturalnie obsługuje przypadek startowy bez generowania NaN.
        return h

    def forward(self, obs: dict, device: str) -> tuple[torch.Tensor, dict]:
        """Metoda forward odczytuje dane, tnie je do rzeczywistego rozmiaru i wywołuje enkoder."""

        # Pobieramy pełne, dopełnione zera tensory
        node_features_padded = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
        edge_index_padded = torch.as_tensor(obs["edge_index"], dtype=torch.long, device=device)

        # ⬇️ ⬇️ ⬇️ NOWA, KLUCZOWA POPRAWKA ⬇️ ⬇️ ⬇️

        # Pobieramy prawdziwą liczbę węzłów i krawędzi
        num_nodes = obs["num_nodes"][0]
        num_edges = obs["num_edges"][0]

        # Tniemy tensory do ich RZECZYWISTYCH rozmiarów
        # Bierzemy tylko 'num_nodes' pierwszych węzłów
        valid_node_features = node_features_padded[:num_nodes]
        # Bierzemy tylko 'num_edges' pierwszych krawędzi
        valid_edge_index = edge_index_padded[:, :num_edges]

        # Przekazujemy "oczyszczone" tensory do enkodera
        # Teraz GNN będzie działać tylko na 3 węzłach, a nie 40
        node_embeddings = self.encode(valid_node_features, valid_edge_index)

        # Tworzymy wektor batcha o RZECZYWISTYM rozmiarze
        # (np. [3] dla 3 stacji, a nie [40])
        batch_vector = torch.zeros(num_nodes, dtype=torch.long, device=device)

        # ⬆️ ⬆️ ⬆️ KONIEC POPRAWKI ⬆️ ⬆️ ⬆️

        # global_mean_pool teraz poprawnie uśredni tylko 3 prawdziwe węzły
        graph_embedding = global_mean_pool(node_embeddings, batch_vector)

        value = self.critic_head(graph_embedding)
        logits = {
            "high_level": self.high_level_head(graph_embedding),
            "manage_line_type": self.manage_line_type_head(graph_embedding),
            "manage_line_p1": self.manage_line_p1_head(graph_embedding),
            "manage_line_p2": self.manage_line_p2_head(graph_embedding),
            "deploy_train": self.deploy_train_head(graph_embedding),
            "select_line": self.select_line_head(graph_embedding)
        }
        return value, logits