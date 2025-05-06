from model import GNNGlobal, GNNTopK, GNNSAG, GNNDiffPool, GNNGlobalAttention, GNNSet2Set, GNNDMoN, GNNECPool, GNNMinCut
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch
import torch.nn.functional as F
import time
import pandas as pd
import gc

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch) 
        loss = F.cross_entropy(out, data.y)   
        loss.backward()     
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)
       
@torch.no_grad()
def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=1) 
        correct += (pred == data.y).sum().item()
    
    return correct / len(loader.dataset)

results = []
for name in ['MUTAG','PROTEINS','ENZYMES']:
    dataset = TUDataset(root='data', name=name)   
    dataset = dataset.shuffle()    

    split = int(0.8 * len(dataset))
    train_ds, test_ds = dataset[:split], dataset[split:]  

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    test_loader  = DataLoader(test_ds,  batch_size=32)
    
    in_feats    = dataset.num_features   
    hidden_dim  = 64                     
    num_classes = dataset.num_classes    
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Dataset: {name}, Number of classes: {num_classes}, Number of features: {in_feats}")

    variants = {
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
            peak_mem = None  

        print(f"{pool} test acc: {acc:.4f}, time/epoch: {time_per_epoch:.4f}s, memory: {peak_mem:.2f} MB")
        torch.save(model.state_dict(), f"trained_models/{name}_{pool}_model.pth")
        results.append({
            'pool': pool,
            'dataset': name,
            's/epoch': round(time_per_epoch, 4),
            'memory_MB': round(peak_mem, 2) if peak_mem is not None else 'N/A',
            'accuracy': round(acc, 4)
        })

        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

results_df = pd.DataFrame(results)
results_df.to_csv(f'gnn_pooling_benchmark.csv', index=False)
print(results_df)