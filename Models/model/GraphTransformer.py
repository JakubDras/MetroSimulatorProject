import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool

import gymnasium_env_metro.config as config


class GraphTransformerModel(nn.Module):
    """
    Definicja architektury sieci opartej na Graph Transformer.
    Obsługuje teraz paczki (batche) grafów ze zrównoleglonych środowisk.
    """

    def __init__(self, num_node_features: int, hidden_dim: int, num_stations: int, heads: int = 4):
        super().__init__()
        num_line_colors = len(config.LINE_COLORS)

        self.initial_projection = nn.Linear(num_node_features, hidden_dim)
        self.encoder_conv1 = TransformerConv(hidden_dim, hidden_dim, heads=heads)
        self.encoder_conv2 = TransformerConv(hidden_dim * heads, hidden_dim, heads=1)

        self.critic_head = nn.Linear(hidden_dim, 1)
        self.high_level_head = nn.Linear(hidden_dim, 4)
        self.manage_line_type_head = nn.Linear(hidden_dim, 3)
        self.manage_line_p1_head = nn.Linear(hidden_dim, num_stations)
        self.manage_line_p2_head = nn.Linear(hidden_dim, num_stations)
        self.deploy_train_head = nn.Linear(hidden_dim, num_stations)
        self.select_line_head = nn.Linear(hidden_dim, num_line_colors)

    def encode(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Przetwarza cechy węzłów, a następnie wzbogaca je o informacje z grafu.
        """
        h = self.initial_projection(node_features).relu()

        if edge_index.shape[1] > 0:
            h = self.encoder_conv1(h, edge_index).relu()
            h = self.encoder_conv2(h, edge_index).relu()
        return h

    def freeze_encoder_layers(self):
        """
        Wyłącza obliczanie gradientów dla warstw enkodera Graph Transformer.
        """

        for param in self.initial_projection.parameters():
            param.requires_grad = False
        for param in self.encoder_conv1.parameters():
            param.requires_grad = False
        for param in self.encoder_conv2.parameters():
            param.requires_grad = False

    def forward(self, obs: dict, device: str) -> tuple[torch.Tensor, dict]:
        """
        NOWA METODA FORWARD (bez zmian)
        """
        node_features_batch = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
        edge_index_batch = torch.as_tensor(obs["edge_index"], dtype=torch.long, device=device)
        num_nodes_batch = torch.as_tensor(obs["num_nodes"], dtype=torch.long, device=device).flatten()
        num_edges_batch = torch.as_tensor(obs["num_edges"], dtype=torch.long, device=device).flatten()

        batch_size = node_features_batch.shape[0]

        all_valid_nodes = []
        all_valid_edges = []
        batch_vector = []
        current_node_offset = 0

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

        if current_node_offset == 0:
            print("Ostrzeżenie: Pusty batch w GraphTransformerModel.forward")
            output_dim = self.encoder_conv2.out_channels
            graph_embedding = torch.zeros(batch_size, output_dim, device=device)

        else:
            h_nodes = torch.cat(all_valid_nodes, dim=0)
            h_batch = torch.cat(batch_vector, dim=0)

            if all_valid_edges:
                h_edges = torch.cat(all_valid_edges, dim=1)
            else:
                h_edges = torch.empty((2, 0), dtype=torch.long, device=device)

            node_embeddings = self.encode(h_nodes, h_edges)

            graph_embedding = global_mean_pool(node_embeddings, h_batch)

            if graph_embedding.shape[0] < batch_size:
                output_dim = self.encoder_conv2.out_channels
                full_graph_embedding = torch.zeros(batch_size, output_dim, device=device)
                full_graph_embedding[torch.unique(h_batch)] = graph_embedding
                graph_embedding = full_graph_embedding

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