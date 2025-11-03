# plik: model/GNNencoder.py
# WERSJA POPRAWIONA DLA ZRÓWNOLEGLONYCH ŚRODOWISK (BATCHING)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

import gymnasium_env_metro.config as config


class GNNModel(nn.Module):
    """
    Definicja architektury sieci opartej na GNN.
    Obsługuje teraz paczki (batche) grafów ze zrównoleglonych środowisk.
    """

    def __init__(self, num_node_features: int, hidden_dim: int, num_stations: int):
        super().__init__()
        num_line_colors = len(config.LINE_COLORS)

        self.initial_projection = nn.Linear(num_node_features, hidden_dim)
        self.encoder_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.encoder_conv2 = GCNConv(hidden_dim, hidden_dim)

        # Głowice Aktora i Krytyka
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
        Ta metoda jest już poprawna i nie wymaga zmian.
        """
        h = self.initial_projection(node_features).relu()
        if edge_index.shape[1] > 0:
            h = self.encoder_conv1(h, edge_index).relu()
            h = self.encoder_conv2(h, edge_index).relu()
        return h

    def forward(self, obs: dict, device: str) -> tuple[torch.Tensor, dict]:
        """
        NOWA METODA FORWARD (bardziej odporna na błędy):
        Poprawnie obsługuje paczkę (batch) obserwacji.
        Tworzy jeden wielki "super-graf" (sparse batch) do przetworzenia przez GNN.
        """

        # 1. Pobieramy tensory z obserwacji.
        # Niezależnie od tego, czy pochodzą z env.step() czy z bufora, konwertujemy je na tensory na odpowiednim urządzeniu.
        node_features_batch = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
        edge_index_batch = torch.as_tensor(obs["edge_index"], dtype=torch.long, device=device)

        # num_nodes i num_edges mogą mieć kształt [batch_size] lub [batch_size, 1]. Ujednolicamy to.
        num_nodes_batch = torch.as_tensor(obs["num_nodes"], dtype=torch.long, device=device).flatten()
        num_edges_batch = torch.as_tensor(obs["num_edges"], dtype=torch.long, device=device).flatten()

        # Pobieramy cechy globalne, upewniając się, że są 2D [batch_size, num_features]
        global_features_batch = torch.as_tensor(obs["global_features"], dtype=torch.float32, device=device)
        if global_features_batch.dim() == 1:
            global_features_batch = global_features_batch.unsqueeze(
                0)  # Poprawka dla pojedynczej obserwacji (np. przy 'evaluate')

        batch_size = node_features_batch.shape[0]

        all_valid_nodes = []
        all_valid_edges = []
        batch_vector = []
        current_node_offset = 0

        # 2. Tworzymy "super-graf" (batching grafów)
        for i in range(batch_size):
            # Używamy .item(), aby dostać pythonowe liczby całkowite do indeksowania
            num_nodes = num_nodes_batch[i].item()
            num_edges = num_edges_batch[i].item()

            # Jeśli nie ma węzłów, pomijamy (to ważne!)
            if num_nodes == 0:
                continue

            # Tniemy, aby uzyskać tylko aktywne węzły
            valid_nodes = node_features_batch[i, :num_nodes]
            all_valid_nodes.append(valid_nodes)

            # Tworzymy wektor batcha dla global_mean_pool
            batch_vector.append(torch.full((num_nodes,), fill_value=i, device=device, dtype=torch.long))

            if num_edges > 0:
                # Tniemy, aby uzyskać tylko aktywne krawędzie
                valid_edges = edge_index_batch[i, :, :num_edges]
                # Przesuwamy indeksy krawędzi o offset
                all_valid_edges.append(valid_edges + current_node_offset)

            current_node_offset += num_nodes

        # 3. Sprawdzamy, czy w ogóle mamy jakieś węzły (możliwy pusty batch)
        if current_node_offset == 0:
            # Nie ma żadnych węzłów w całej paczce, zwracamy zerowe tensory
            # To jest przypadek brzegowy, ale ważny do obsłużenia
            print("Ostrzeżenie: Pusty batch w GNNModel.forward")
            graph_embedding = torch.zeros(batch_size, self.encoder_conv2.out_channels, device=device)

        else:
            # 4. Łączymy w jeden duży graf
            h_nodes = torch.cat(all_valid_nodes, dim=0)  # Kształt [N_total, F]
            h_batch = torch.cat(batch_vector, dim=0)  # Kształt [N_total]

            if all_valid_edges:  # Sprawdzamy, czy lista nie jest pusta
                h_edges = torch.cat(all_valid_edges, dim=1)  # Kształt [2, E_total]
            else:
                # Nie ma żadnych krawędzi w całej paczce
                h_edges = torch.empty((2, 0), dtype=torch.long, device=device)

            # 5. Uruchamiamy GNN
            node_embeddings = self.encode(h_nodes, h_edges)

            # 6. Agregujemy (pool) do poziomu grafu
            graph_embedding = global_mean_pool(node_embeddings, h_batch)

            # Upewniamy się, że mamy wynik dla każdego elementu w batchu
            # (jeśli jakiś graf miał 0 węzłów, global_mean_pool da 0, ale może zmienić rozmiar)
            if graph_embedding.shape[0] < batch_size:
                full_graph_embedding = torch.zeros(batch_size, self.encoder_conv2.out_channels, device=device)
                full_graph_embedding[torch.unique(h_batch)] = graph_embedding
                graph_embedding = full_graph_embedding

        # 7. Głowice decyzyjne
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