import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from mamba_ssm import Mamba

class GraphMamba(nn.Module):
    def __init__(self, num_node_features: int, hidden_dim: int, num_stations: int, num_line_colors: int):
        super().__init__()

        self.gnn_conv1 = GCNConv(num_node_features, hidden_dim)
        self.gnn_conv2 = GCNConv(hidden_dim, hidden_dim)

        self.mamba = Mamba(
            d_model=hidden_dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.critic_head = nn.Linear(hidden_dim, 1)

        self.high_level_head = nn.Linear(hidden_dim, 4)
        self.manage_line_type_head = nn.Linear(hidden_dim, 3)
        self.manage_line_p1_head = nn.Linear(hidden_dim, num_stations)
        self.manage_line_p2_head = nn.Linear(hidden_dim, num_stations)
        self.deploy_train_head = nn.Linear(hidden_dim, num_stations)
        self.select_line_head = nn.Linear(hidden_dim, num_line_colors)

    def encode(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_gnn = self.gnn_conv1(node_features, edge_index).relu()
        h_gnn = self.gnn_conv2(h_gnn, edge_index).relu()

        seq_in = h_gnn.unsqueeze(0)
        seq_out = self.mamba(seq_in)
        h_mamba = seq_out.squeeze(0)

        h_final = self.norm(h_gnn + h_mamba)
        return h_final

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
            "manage_line_p1": self.manage_line_p1_head(graph_embedding),  # POPRAWKA
            "manage_line_p2": self.manage_line_p2_head(graph_embedding),  # POPRAWKA
            "deploy_train": self.deploy_train_head(graph_embedding),  # POPRAWKA
            "select_line": self.select_line_head(graph_embedding)
        }
        return value, logits