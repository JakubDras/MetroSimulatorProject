import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import MambaForCausalLM, MambaConfig

import gymnasium_env_metro.config as config


class GraphMambaHFModel(nn.Module):
    """
    GNN + Mamba.
    Pretrained Mamba model from HuggingFace`.
    """
    def __init__(self, num_node_features: int, hidden_dim: int, num_stations: int,
                 mamba_model_name: str = "state-spaces/mamba-130m-slimpj", freeze_mamba: bool = True):
        super().__init__()
        num_line_colors = len(config.LINE_COLORS)

        mamba_config = MambaConfig.from_pretrained(mamba_model_name)
        mamba_full_model = MambaForCausalLM.from_pretrained(mamba_model_name)

        self.mamba_backbone = mamba_full_model.backbone

        mamba_hidden_size = mamba_config.hidden_size
        if hidden_dim != mamba_hidden_size:
            raise ValueError(
                f"Parametr 'hidden_dim' ({hidden_dim}) musi pasować do 'hidden_size' modelu Mamba ({mamba_hidden_size})!")

        if freeze_mamba:
            for param in self.mamba_backbone.parameters():
                param.requires_grad = False

        #GNN
        self.encoder_conv1 = GCNConv(num_node_features, hidden_dim)
        self.encoder_conv2 = GCNConv(hidden_dim, hidden_dim)


        self.norm = nn.LayerNorm(hidden_dim)

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
        h_gnn = self.encoder_conv1(node_features, edge_index).relu()
        h_gnn = self.encoder_conv2(h_gnn, edge_index).relu()

        seq_in = h_gnn.unsqueeze(0)
        mamba_outputs = self.mamba_backbone(inputs_embeds=seq_in)
        h_mamba = mamba_outputs.last_hidden_state.squeeze(0)

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
            "manage_line_p1": self.manage_line_p1_head(node_embeddings),
            "manage_line_p2": self.manage_line_p2_head(node_embeddings),
            "deploy_train": self.deploy_train_head(node_embeddings),
            "select_line": self.select_line_head(graph_embedding)
        }
        return value, logits