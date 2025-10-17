import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

import gymnasium_env_metro.config as config


class GNNModel(nn.Module):
    def __init__(self, num_node_features: int, hidden_dim: int, num_stations: int):
        super().__init__()
        num_line_colors = len(config.LINE_COLORS)

        #GNN
        self.encoder_conv1 = GCNConv(num_node_features, hidden_dim)
        self.encoder_conv2 = GCNConv(hidden_dim, hidden_dim)

        #Critic
        self.critic_head = nn.Linear(hidden_dim, 1)

        #Actor
        self.high_level_head = nn.Linear(hidden_dim, 4)
        self.manage_line_type_head = nn.Linear(hidden_dim, 3)
        self.manage_line_p1_head = nn.Linear(hidden_dim, num_stations)
        self.manage_line_p2_head = nn.Linear(hidden_dim, num_stations)
        self.deploy_train_head = nn.Linear(hidden_dim, num_stations)
        self.select_line_head = nn.Linear(hidden_dim, num_line_colors)

    def encode(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv1(node_features, edge_index).relu()
        h = self.encoder_conv2(h, edge_index).relu()
        return h

    def forward(self, obs: dict, device: str) -> tuple[torch.Tensor, dict]:
        node_features = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
        edge_index = torch.as_tensor(obs["edge_index"], dtype=torch.long, device=device)

        node_embeddings = self.encode(node_features, edge_index)
        batch_vector = torch.zeros(node_embeddings.shape[0], dtype=torch.long, device=device)
        graph_embedding = global_mean_pool(node_embeddings, batch_vector)

        value = self.critic_head(graph_embedding)
        logits = {
            "high_level": self.high_level_head(graph_embedding),
            "manage_line_type": self.manage_line_type_head(graph_embedding),
            "manage_line_p1": self.manage_line_p1_head(node_embeddings),
            "manage_line_p2": self.manage_line_p2_head(node_embeddings),
            "deploy_train": self.deploy_train_head(node_embeddings),
            "select_line": self.select_line_head(graph_embedding)
        }
        return value, logits