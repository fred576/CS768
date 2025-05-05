import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=dropout)  # Define dropout with configurable rate
        self.link_pred = torch.nn.Sequential(
            torch.nn.Linear(out_channels * 2, out_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),  # Add dropout in link prediction MLP
            torch.nn.Linear(out_channels, 1)
        )

    def link_prediction(self, x, edge_index):
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        x = torch.cat([x_i, x_j], dim=1)
        return self.link_pred(x)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Add dropout after first conv layer
        x = self.conv2(x, edge_index)
        return x