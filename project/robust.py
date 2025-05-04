# robust_analysis.py
# ----------------
# Empirical robustness evaluation for GNN pooling variants on TUDatasets
# using random edge removal and feature noise.
# Integrates with your existing `model.py` definitions and main.py setup.

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from model import (
    GNNGlobal, GNNTopK, GNNSAG, GNNDiffPool,
    GNNGlobalAttention, GNNSet2Set, GNNDMoN,
    GNNECPool, GNNMinCut
)

# --------------------
# 1) Perturbation functions
# --------------------

def drop_edges(data, rho):
    """
    Randomly remove a fraction rho of edges from the graph.
    data: PyG Data object
    rho: fraction of edges to drop (0 <= rho < 1)
    """
    # convert edge_index to list of edges
    E = data.edge_index.t()  # [E, 2]
    m = E.size(0)
    keep = torch.randperm(m)[:int((1 - rho) * m)]
    new_edge_index = E[keep].t().contiguous()
    return data.__class__(
        x=data.x, edge_index=new_edge_index, y=data.y, batch=data.batch
    )


def noisy_features(data, sigma):
    """
    Add Gaussian noise N(0, sigma^2) to node features.
    """
    x_noisy = data.x + torch.randn_like(data.x) * sigma
    return data.__class__(
        x=x_noisy, edge_index=data.edge_index, y=data.y, batch=data.batch
    )

# --------------------
# 2) Robustness evaluation
# --------------------

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            # unpack if tuple
            if isinstance(out, tuple):
                out = out[0]
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.num_graphs
    return correct / total


def robustness_curve(model, test_ds, device, perturb_fn, levels, batch_size=32):
    """
    Compute accuracy of model under different perturbation levels.

    model: a GNN instance
    test_ds: list of Data objects (unbatched)
    perturb_fn: function(data, level) -> perturbed Data
    levels: list of perturbation magnitudes
    returns: list of accuracies corresponding to levels
    """
    accs = []
    for lvl in levels:
        # apply perturbation transform on the fly in loader
        perturbed = [perturb_fn(d, lvl) for d in test_ds]
        loader = DataLoader(perturbed, batch_size=batch_size)
        acc = evaluate(model, loader, device)
        accs.append(acc)
    return accs

# --------------------
# 3) Runner
# --------------------

def run_all(dataset_name='MUTAG', device=None, out_csv='robustness_results.csv'):
    """
    Runs robustness curves for all pooling variants and saves to CSV.
    Columns: model, perturbation, level, accuracy
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load and split dataset like in main.py
    dataset = TUDataset(root='data', name=dataset_name).shuffle()
    split = int(0.8 * len(dataset))
    train_ds, test_ds = dataset[:split], dataset[split:]

    # instantiate variants (choose a representative set)
    variants = {
        'dmon':    GNNDMoN(dataset.num_features, 64, dataset.num_classes, k=10, dropout=0.2),
        'ecpool':  GNNECPool(dataset.num_features, 64, dataset.num_classes, ratio=0.5),
        'mincut':  GNNMinCut(dataset.num_features, 64, dataset.num_classes, k=10, temp=1.0),
        'topk':    GNNTopK(dataset.num_features, 64, dataset.num_classes, ratio=0.5),
        'sag':     GNNSAG(dataset.num_features, 64, dataset.num_classes, ratio=0.5),
        'global':  GNNGlobal(dataset.num_features, 64, dataset.num_classes, pool='mean')
    }

    # load pre-trained weights if available
    for name, model in variants.items():
        path = f'trained_models/{dataset_name}_{name}_model.pth'
        print(path)
        if os.path.exists(path):
            print(f"Loading model weights from {path}")
            model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)

    # define perturbations and levels
    perturbations = {
        'edge_drop': drop_edges,
        'feat_noise': noisy_features
    }
    levels = {
        'edge_drop': np.linspace(0.0, 0.5, 6),   # 0%,10%,…,50%
        'feat_noise': np.linspace(0.0, 1.0, 6)    # σ=0,…,1
    }

    # collect results
    rows = []
    for pert_name, fn in perturbations.items():
        for lvl in levels[pert_name]:
            for var_name, model in variants.items():
                acc = robustness_curve(model, test_ds, device, fn, [lvl])[0]
                rows.append({
                    'model': var_name,
                    'perturbation': pert_name,
                    'level': float(lvl),
                    'accuracy': float(acc)
                })
    # save
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Robustness results written to {out_csv}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Robustness analysis for GNN pooling variants')
    parser.add_argument('--dataset', type=str, default='MUTAG')
    parser.add_argument('--out', type=str, default='robustness_results.csv')
    args = parser.parse_args()
    run_all(dataset_name=args.dataset, out_csv=args.out)
