# plik: model/GraphMamba.py
# WERSJA POPRAWIONA DLA ZRÓWNOLEGLONYCH ŚRODOWISK (BATCHING)

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from mamba_ssm import Mamba

import gymnasium_env_metro.config as config


class GraphMambaModel(nn.Module):
    """
    Definicja architektury sieci opartej na GCN + Mamba.
    Obsługuje teraz paczki (batche) grafów ze zrównoleglonych środowisk.
    """

    def __init__(self, num_node_features: int, hidden_dim: int, num_stations: int):
        super().__init__()
        # Pobieramy num_line_colors z configu, tak jak w innych modelach
        num_line_colors = len(config.LINE_COLORS)

        # Warstwy GCN
        self.gnn_conv1 = GCNConv(num_node_features, hidden_dim)
        self.gnn_conv2 = GCNConv(hidden_dim, hidden_dim)

        # Mamba
        self.mamba = Mamba(
            d_model=hidden_dim,
            d_state=16,  # Możesz chcieć to dostroić
            d_conv=4,
            expand=2,
        )

        # Warstwa normalizująca po połączeniu rezydualnym
        self.norm = nn.LayerNorm(hidden_dim)

        # Głowice Aktora i Krytyka (bez zmian)
        self.critic_head = nn.Linear(hidden_dim, 1)
        self.high_level_head = nn.Linear(hidden_dim, 4)
        self.manage_line_type_head = nn.Linear(hidden_dim, 3)
        self.manage_line_p1_head = nn.Linear(hidden_dim, num_stations)
        self.manage_line_p2_head = nn.Linear(hidden_dim, num_stations)
        self.deploy_train_head = nn.Linear(hidden_dim, num_stations)
        self.select_line_head = nn.Linear(hidden_dim, num_line_colors)

    def encode(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Twoja autorska funkcja enkodera (GCN -> Mamba -> Residual)
        Ta funkcja jest teraz poprawnie wywoływana przez batched 'forward'.
        'node_features' będzie miało kształt [N_total, F]
        """
        h_gnn = self.gnn_conv1(node_features, edge_index).relu()
        h_gnn = self.gnn_conv2(h_gnn, edge_index).relu()

        # Mamba oczekuje (batch, seq_len, d_model)
        # Traktujemy wszystkie węzły w paczce jako jedną, długą sekwencję
        seq_in = h_gnn.unsqueeze(0)  # Kształt: [1, N_total, hidden_dim]
        seq_out = self.mamba(seq_in)
        h_mamba = seq_out.squeeze(0)  # Kształt: [N_total, hidden_dim]

        # Połączenie rezydualne i normalizacja
        h_final = self.norm(h_gnn + h_mamba)
        return h_final

    def forward(self, obs: dict, device: str) -> tuple[torch.Tensor, dict]:
        """
        NOWA METODA FORWARD (skopiowana z GNNModel/GraphTransformer)
        Poprawnie obsługuje paczkę (batch) obserwacji.
        """

        # 1. Pobieramy tensory z obserwacji.
        node_features_batch = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
        edge_index_batch = torch.as_tensor(obs["edge_index"], dtype=torch.long, device=device)
        num_nodes_batch = torch.as_tensor(obs["num_nodes"], dtype=torch.long, device=device).flatten()
        num_edges_batch = torch.as_tensor(obs["num_edges"], dtype=torch.long, device=device).flatten()
        global_features_batch = torch.as_tensor(obs["global_features"], dtype=torch.float32, device=device)

        if global_features_batch.dim() == 1:
            global_features_batch = global_features_batch.unsqueeze(0)

        batch_size = node_features_batch.shape[0]

        all_valid_nodes = []
        all_valid_edges = []
        batch_vector = []
        current_node_offset = 0

        # 2. Tworzymy "super-graf" (batching grafów)
        for i in range(batch_size):
            num_nodes = num_nodes_batch[i].item()
            num_edges = num_edges_batch[i].item()

            if num_nodes == 0:
                continue

            valid_nodes = node_features_batch[i, :num_nodes]
            all_valid_nodes.append(valid_nodes)
            batch_vector.append(torch.full((num_nodes,), fill_value=i, device=device, dtype=torch.long))

            if num_edges > 0:
                valid_edges = edge_index_batch[i, :, :num_edges]
                all_valid_edges.append(valid_edges + current_node_offset)

            current_node_offset += num_nodes

        # 3. Sprawdzamy, czy w ogóle mamy jakieś węzły (możliwy pusty batch)
        if current_node_offset == 0:
            print("Ostrzeżenie: Pusty batch w GraphMambaModel.forward")
            output_dim = self.gnn_conv2.out_channels
            graph_embedding = torch.zeros(batch_size, output_dim, device=device)

        else:
            # 4. Łączymy w jeden duży graf
            h_nodes = torch.cat(all_valid_nodes, dim=0)  # Kształt [N_total, F]
            h_batch = torch.cat(batch_vector, dim=0)  # Kształt [N_total]

            if all_valid_edges:
                h_edges = torch.cat(all_valid_edges, dim=1)  # Kształt [2, E_total]
            else:
                h_edges = torch.empty((2, 0), dtype=torch.long, device=device)

            # 5. Uruchamiamy GNN (tutaj wywoła się Twój nowy `encode`)
            node_embeddings = self.encode(h_nodes, h_edges)

            # 6. Agregujemy (pool) do poziomu grafu
            graph_embedding = global_mean_pool(node_embeddings, h_batch)

            # 7. Upewniamy się, że mamy wynik dla każdego elementu w batchu
            if graph_embedding.shape[0] < batch_size:
                output_dim = self.gnn_conv2.out_channels
                full_graph_embedding = torch.zeros(batch_size, output_dim, device=device)
                full_graph_embedding[torch.unique(h_batch)] = graph_embedding
                graph_embedding = full_graph_embedding

        # 8. Głowice decyzyjne (bez zmian)
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