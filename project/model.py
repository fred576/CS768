import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import (
    GCNConv, global_mean_pool, global_add_pool, global_max_pool,
    TopKPooling, SAGPooling, GlobalAttention, Set2Set, EdgePooling, max_pool
)
from torch_geometric.nn.dense import DMoNPooling, dense_mincut_pool

class GNNGlobal(nn.Module):
    """
    Simple 2-layer GCN + global pooling for graph classification.
    Supports mean, sum, or max pooling.
    """
    def __init__(self, in_feats, hidden_dim, num_classes, pool='mean'):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool = pool
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        if self.pool == 'mean':
            out = global_mean_pool(x, batch)
        elif self.pool == 'sum':
            out = global_add_pool(x, batch)
        elif self.pool == 'max':
            out = global_max_pool(x, batch)
        elif self.pool == 'none':
            out = x
        return self.lin(out)

class GNNTopK(nn.Module):
    """
    GCN + Top-K pooling + another GCN + global mean pooling.
    """
    def __init__(self, in_feats, hidden_dim, num_classes, ratio=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.pool1 = TopKPooling(hidden_dim, ratio)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        out = global_mean_pool(x, batch)
        return self.lin(out)

class GNNSAG(nn.Module):
    """
    GCN + Self-Attention Graph (SAG) pooling + GCN + global mean pooling.
    """
    def __init__(self, in_feats, hidden_dim, num_classes, ratio=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.pool = SAGPooling(hidden_dim, ratio)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        out = global_mean_pool(x, batch)
        return self.lin(out)

class DiffPool(nn.Module):
    """
    Core DiffPool layer:
    - Learns soft cluster assignments S for nodes
    - Aggregates node embeddings Z into k clusters per graph
    """
    def __init__(self, in_dim, assign_dim, k):
        super().__init__()
        self.gnn_embed = GCNConv(in_dim, assign_dim)
        self.gnn_assign = GCNConv(in_dim, k)

    def forward(self, x, edge_index, batch):
        N_E = F.relu(self.gnn_embed(x, edge_index))
        S = F.softmax(self.gnn_assign(x, edge_index), dim=-1)
        NE_batch, mask = to_dense_batch(N_E, batch)
        S_batch, _ = to_dense_batch(S, batch)
        A = to_dense_adj(edge_index, batch)
        Xp = torch.matmul(S_batch.transpose(1, 2), NE_batch)
        Ap = torch.matmul(torch.matmul(S_batch.transpose(1, 2), A), S_batch)
        return Xp, Ap

class GNNDiffPool(nn.Module):
    """
    Graph classification wrapper using DiffPool:
    - Optional initial GCN
    - DiffPool layer
    - MLP on pooled graph representation
    """
    def __init__(self, in_feats, assign_dim, k, num_classes):
        super().__init__()
        self.pre_conv = GCNConv(in_feats, assign_dim)
        self.diffpool = DiffPool(assign_dim, assign_dim, k)
        self.lin1 = nn.Linear(assign_dim, assign_dim)
        self.lin2 = nn.Linear(assign_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.pre_conv(x, edge_index))
        Xp, Ap = self.diffpool(x, edge_index, batch)
        out = Xp.mean(dim=1)
        out = F.relu(self.lin1(out))
        return self.lin2(out)

class GNNGlobalAttention(nn.Module):
    """
    GCN + Global Attention Pooling (GAP) + MLP
    """
    def __init__(self, in_feats, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
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
        self.set2set = Set2Set(hidden_dim, processing_steps)
        self.lin1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
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
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dmon = DMoNPooling([hidden_dim], k, dropout)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        X_batch, mask = to_dense_batch(x, batch)
        A_batch = to_dense_adj(edge_index, batch)
        S, Xp, Ap, loss_s, loss_o, loss_c = self.dmon(X_batch, A_batch)
        out = Xp.mean(dim=1)
        logits = self.lin(out)
        return logits

class GNNECPool(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, ratio=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.ec_pool = EdgePooling(hidden_dim, dropout=1.0 - ratio)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, batch, _ = self.ec_pool(x, edge_index, batch)
        x = F.relu(self.conv2(x, edge_index))
        out = global_mean_pool(x, batch)
        return self.lin(out)

class GNNMinCut(nn.Module):
    """
    GCN + MinCut Pooling + GCN + global mean pooling
    """
    def __init__(self, in_feats, hidden_dim, num_classes, k=30, temp=1.0):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.assign_conv = GCNConv(hidden_dim, k)
        self.lin = nn.Linear(hidden_dim, num_classes)
        self.k = k
        self.temp = temp

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        X_batch, mask = to_dense_batch(x, batch)
        A_batch = to_dense_adj(edge_index, batch)
        S = self.assign_conv(x, edge_index)
        S_batch, _ = to_dense_batch(S, batch)
        Xp, Ap, mincut_loss, ortho_loss = dense_mincut_pool(
            X_batch, A_batch, S_batch, mask, temp=self.temp)
        out = Xp.mean(dim=1)
        logits = self.lin(out)
        return logits
