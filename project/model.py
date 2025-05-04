import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import DataLoader, Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import subgraph, to_dense_adj, to_dense_batch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, global_mean_pool, global_add_pool, global_max_pool,
    TopKPooling, SAGPooling, GlobalAttention, Set2Set, EdgePooling, max_pool
)
from utils import negative_edges, batched_negative_edges
from torch_geometric.nn.dense import DMoNPooling, dense_mincut_pool
# -----------------------------------------------------------------------------
# 1. Model Definitions
# -----------------------------------------------------------------------------

"""
    Simple 2-layer GCN + global pooling for graph classification.
    Supports mean, sum, or max pooling.
"""
class GNNGlobal(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, pool='mean'):
        super().__init__()
        # Two GCN layers
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # Pooling strategy: 'mean', 'sum', or 'max'
        self.pool  = pool
        # Final linear classifier
        self.lin   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        # Message passing + ReLU activations
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Global pooling to get graph-level embedding
        if self.pool == 'mean':
            out = global_mean_pool(x, batch)
        elif self.pool == 'sum':
            out = global_add_pool(x, batch)
        elif self.pool == 'max':
            out  = global_max_pool(x, batch)
        elif self.pool == 'none':
            out  = x
        # Classification output
        return self.lin(out)


"""
    GCN + Top-K pooling + another GCN + global mean pooling.
"""
class GNNTopK(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, ratio=0.5):
        super().__init__()
        # First GCN layer
        self.conv1 = GCNConv(in_feats, hidden_dim)
        # TopK pooling layer retains top `ratio` fraction of nodes
        self.pool1 = TopKPooling(hidden_dim, ratio)
        # Second GCN layer
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # Final classifier
        self.lin   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        # 1) Initial convolution + activation
        x = F.relu(self.conv1(x, edge_index))
        # 2) Top-K pooling: returns pruned x, edges, batch, etc.
        x, edge_index, edge_attr, batch, perm, score = self.pool1(x, edge_index, None, batch)
        # 3) Second convolution + activation
        x = F.relu(self.conv2(x, edge_index))
        # 4) Global mean pooling to aggregate node features per graph
        out = global_mean_pool(x, batch)
        # 5) Classification
        return self.lin(out)

"""
    GCN + Self-Attention Graph (SAG) pooling + GCN + global mean pooling.
"""
class GNNSAG(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, ratio=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.pool  = SAGPooling(hidden_dim, ratio)     # SAGPooling layer uses self-attention scores to select nodes
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        # Convolution + activation
        x = F.relu(self.conv1(x, edge_index))
        # SAG pooling: returns pruned graph
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        # Second convolution + activation
        x = F.relu(self.conv2(x, edge_index))
        # Global mean pooling
        out = global_mean_pool(x, batch)
        # Classification
        return self.lin(out)

"""
    Core DiffPool layer:
    - Learns soft cluster assignments S for nodes
    - Aggregates node embeddings Z into k clusters per graph
"""
class DiffPool(nn.Module):
    def __init__(self, in_dim, assign_dim, k):
        super().__init__()
        # GCN to compute node embeddings N_E
        self.gnn_embed  = GCNConv(in_dim, assign_dim)
        # GCN to compute assignment scores S (to k clusters)
        self.gnn_assign = GCNConv(in_dim, k)

    def forward(self, x, edge_index, batch):
        # Node embedding and assignment score computation
        N_E = F.relu(self.gnn_embed(x, edge_index))             # [N_total, H]
        S = F.softmax(self.gnn_assign(x, edge_index), dim=-1) # [N_total, K]

        # Pack nodes into dense batches for matrix ops
        NE_batch, mask = to_dense_batch(N_E, batch)    # NE_batch: [B, N_max, H]
        S_batch, _    = to_dense_batch(S, batch)    # S_batch: [B, N_max, K]

        # Build dense adjacency matrices per graph: [B, N_max, N_max]
        A = to_dense_adj(edge_index, batch)

        # Clustered feature aggregation:
        # Xp[b] = S^T[b] @ Z[b] -> [B, K, H]
        Xp = torch.matmul(S_batch.transpose(1, 2), NE_batch)

        # Clustered adjacency: Ap[b] = S^T[b] @ A[b] @ S[b] -> [B, K, K]
        Ap = torch.matmul(torch.matmul(S_batch.transpose(1, 2), A), S_batch)

        return Xp, Ap


"""
    Graph classification wrapper using DiffPool:
    - Optional initial GCN
    - DiffPool layer
    - MLP on pooled graph representation
"""
class GNNDiffPool(nn.Module):
    def __init__(self, in_feats, assign_dim, k, num_classes):
        super().__init__()
        # Preprocessing GCN
        self.pre_conv = GCNConv(in_feats, assign_dim)
        # DiffPool layer
        self.diffpool = DiffPool(assign_dim, assign_dim, k)
        # MLP head
        self.lin1 = nn.Linear(assign_dim, assign_dim)
        self.lin2 = nn.Linear(assign_dim, num_classes)

    def forward(self, x, edge_index, batch):
        # Preprocess node features
        x = F.relu(self.pre_conv(x, edge_index))

        # DiffPool -> pooled features Xp
        Xp, Ap = self.diffpool(x, edge_index, batch)
        # Xp: [B, k, assign_dim]

        # Graph-level readout: mean over pooled clusters
        out = Xp.mean(dim=1)              # [B, assign_dim]
        out = F.relu(self.lin1(out))
        # 4) Classification
        return self.lin2(out)

class GNNGlobalAttention(nn.Module):
    """
    GCN + Global Attention Pooling (GAP) + MLP
    """
    def __init__(self, in_feats, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # Learnable attention gate
        self.att_gate = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # attention-weighted pooling
        out = self.att_gate(x, batch)
        return self.lin(out)


class GNNSet2Set(nn.Module):
    """
    GCN + Set2Set readout + MLP
    """
    def __init__(self, in_feats, hidden_dim, num_classes, processing_steps=3):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # Set2Set outputs 2*hidden_dim
        self.set2set = Set2Set(hidden_dim, processing_steps)
        self.lin1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # Set2Set pooling
        out = self.set2set(x, batch)
        out = F.relu(self.lin1(out))
        return self.lin2(out)

class GNNGraclus(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        cluster = graclus(edge_index, weight=None, num_nodes=x.size(0))
        data = Data(x=x, edge_index=edge_index, batch=batch)
        pooled = max_pool(cluster, data)
        x, edge_index, batch = pooled.x, pooled.edge_index, pooled.batch
        x = F.relu(self.conv2(x, edge_index))
        out = global_mean_pool(x, batch)
        return self.lin(out)
    
class GNNDMoN(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, k, dropout=0.0):
        super().__init__()
        # Two GCN layers for embeddings
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # DMoNPooling: channels=list of input dims, k clusters, dropout
        self.dmon = DMoNPooling([hidden_dim], k, dropout)
        # Final classifier
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        # 1) Compute node embeddings
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # 2) Convert to dense formats
        X_batch, mask = to_dense_batch(x, batch)        # [B, N_max, H]
        A_batch = to_dense_adj(edge_index, batch)       # [B, N_max, N_max]
        # 3) DMoN pooling
        S, Xp, Ap, loss_s, loss_o, loss_c = self.dmon(X_batch, A_batch)
        # 4) Readout: mean over clusters
        out = Xp.mean(dim=1)                            # [B, H]
        # 5) Classification
        logits = self.lin(out)
        # Return logits and DMoN losses for regularization
        return logits
    
class GNNECPool(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, ratio=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.ec_pool = EdgePooling(hidden_dim, dropout = 1.0 - ratio)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, batch, _ = self.ec_pool(x, edge_index, batch)
        x = F.relu(self.conv2(x, edge_index))
        out = global_mean_pool(x, batch)
        return self.lin(out)

"""
    GCN + MinCut Pooling + GCN + global mean pooling
"""
class GNNMinCut(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes,
                 k=30,      # number of clusters
                 temp=1.0   # softmax temperature
                ):
        super().__init__()
        # GCN for feature embedding
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # Assignment conv
        self.assign_conv = GCNConv(hidden_dim, k)
        self.lin = nn.Linear(hidden_dim, num_classes)
        self.k = k
        self.temp = temp

    def forward(self, x, edge_index, batch):
        # 1) Embed
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # 2) Dense formats
        X_batch, mask = to_dense_batch(x, batch)        # [B, N_max, H]
        A_batch = to_dense_adj(edge_index, batch)       # [B, N_max, N_max]
        # 3) Compute assignments
        S = self.assign_conv(x, edge_index)             # [N_total, k]
        S_batch, _ = to_dense_batch(S, batch)           # [B, N_max, k]
        # 4) MinCut pooling
        Xp, Ap, mincut_loss, ortho_loss = dense_mincut_pool(
            X_batch, A_batch, S_batch, mask, temp=self.temp)
        # 5) Readout & classify
        out = Xp.mean(dim=1)                            # [B, H]
        logits = self.lin(out)
        return logits
