from model import GNNGlobal, GNNTopK, GNNSAG, GNNDiffPool, GNNGlobalAttention, GNNSet2Set, GNNDMoN, GNNECPool, GNNMinCut
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch
import torch.nn.functional as F
import time
import pandas as pd
import gc
# -----------------------------------------------------------------------------
# Looping over different datasets
    # -----------------------------------------------------------------------------
    # 4. Training & Evaluation Functions
    # -----------------------------------------------------------------------------

    # Performs one epoch of training and return average loss
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)  # Forward pass
        loss = F.cross_entropy(out, data.y)   # Compute cross-entropy loss between predictions and true labels
        loss.backward()     # Backpropagation
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)             # Return average loss per graph

"""Evaluate model accuracy on the given DataLoader."""
@torch.no_grad()   # Disables autograd tracking
def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)   # Predict class by selecting highest logit
        correct += (pred == data.y).sum().item()
    
    return correct / len(loader.dataset)

    # Train & evaluate each variant for 100 epochs
results = []
for name in ['MUTAG','PROTEINS','ENZYMES']:
# for name in ['MUTAG']:
    # -----------------------------------------------------------------------------
    # 2. Data Loading & Preparation
    # -----------------------------------------------------------------------------

    dataset = TUDataset(root='data', name=name)    # Load the graph classification dataset into a PyTorch Geometric TUDataset object
    dataset = dataset.shuffle()    # Shuffle the dataset to avoid any potential ordering bias when splitting

    # Compute the index at which to split the dataset:
    # 80% for training, 20% for testing
    split = int(0.8 * len(dataset))
    train_ds, test_ds = dataset[:split], dataset[split:]  # Split the shuffled dataset into train and test datasets

    # Wrap the training subset in a DataLoader to yield batches of 32 graphs
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # Wrap the test subset in a DataLoader to yield batches of 32 graphs
    # during evaluation; shuffling is disabled to preserve consistency
    test_loader  = DataLoader(test_ds,  batch_size=32)
    
    # -----------------------------------------------------------------------------
    # 3. Hyperparameters & Device Configuration
    # -----------------------------------------------------------------------------

    in_feats    = dataset.num_features   # Number of node features per graph
    hidden_dim  = 64                     # Hidden dimension size for internal GCN layers
    num_classes = dataset.num_classes    # Number of target classes for graph-level classification
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Compute device
    print(f"Using device: {device}")
    print(f"Dataset: {name}, Number of classes: {num_classes}, Number of features: {in_feats}")
   # Return accuracy over all graphs

    # -----------------------------------------------------------------------------
    # 5. Model Instantiation & Training Loop
    # -----------------------------------------------------------------------------

    # Define different GNN variants to compare
    variants = {
        # 'no-pool':     GNNGlobal(in_feats, hidden_dim, num_classes, pool='none'),
        # 'global-max':  GNNGlobal(in_feats, hidden_dim, num_classes, pool='max'),
        'dmon':          GNNDMoN(in_feats, hidden_dim, num_classes, k=10, dropout=0.2),
        'ecpool':        GNNECPool(in_feats, hidden_dim, num_classes, ratio=0.5),
        'mincut':        GNNMinCut(in_feats, hidden_dim, num_classes, k=10, temp = 1.0),
        'global-att':  GNNGlobalAttention(in_feats, hidden_dim, num_classes),
        'set2set':     GNNSet2Set(in_feats, hidden_dim, num_classes),
        'global-mean': GNNGlobal(in_feats, hidden_dim, num_classes, pool='mean'),
        'global-sum':  GNNGlobal(in_feats, hidden_dim, num_classes, pool='sum'),
        'topk':        GNNTopK(in_feats, hidden_dim, num_classes),
        'sag':         GNNSAG(in_feats, hidden_dim, num_classes, ratio=0.5),
        'diff':        GNNDiffPool(in_feats, assign_dim=hidden_dim, k=30, num_classes=num_classes)
    }

    print(f"For {name} dataset:")
    for pool, model in variants.items():
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        for epoch in range(1, 101):
            loss = train(model, train_loader, optimizer, device)
        total_time = time.time() - start_time
        time_per_epoch = total_time / 100

        acc = test(model, test_loader, device)

        if device.type == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
        else:
            peak_mem = None  # Could use psutil here for CPU if needed

        print(f"{pool} test acc: {acc:.4f}, time/epoch: {time_per_epoch:.4f}s, memory: {peak_mem:.2f} MB")
        torch.save(model.state_dict(), f"trained_models/{name}_{pool}_model.pth")
        results.append({
            'pool': pool,
            'dataset': name,
            's/epoch': round(time_per_epoch, 4),
            'memory_MB': round(peak_mem, 2) if peak_mem is not None else 'N/A',
            'accuracy': round(acc, 4)
        })

        # Cleanup to avoid CUDA OOM across variants
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

# Convert to DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv(f'gnn_pooling_benchmark.csv', index=False)
print(results_df)