# robust_analysis.py
# ----------------
# Improved empirical robustness evaluation for GNN pooling variants on TUDatasets
# using random edge removal and feature noise with multiple runs for statistical significance.
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
from tqdm import tqdm

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
def drop_nodes(data, rho):
    """
    Randomly remove a fraction rho of nodes from the graph.
    data: PyG Data object
    rho: fraction of nodes to drop (0 <= rho < 1)
    """
    num_nodes = data.num_nodes
    
    # Create a mask for nodes to keep
    mask = torch.rand(num_nodes) > rho
    new_x = data.x[mask]
    
    # Remap edge_index to account for removed nodes
    node_map = torch.zeros(num_nodes, dtype=torch.long, device=data.edge_index.device)
    node_map[mask] = torch.arange(mask.sum(), device=data.edge_index.device)
    new_edge_index = data.edge_index[:, mask[data.edge_index[0]] & mask[data.edge_index[1]]]
    new_edge_index = node_map[new_edge_index]
    
    return data.__class__(
        x=new_x, edge_index=new_edge_index, y=data.y, batch=data.batch
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


def robustness_curve(model, test_ds, device, perturb_fn, levels, n_runs=10, batch_size=32, seed=None):
    """
    Compute accuracy of model under different perturbation levels with multiple runs.

    model: a GNN instance
    test_ds: list of Data objects (unbatched)
    perturb_fn: function(data, level) -> perturbed Data
    levels: list of perturbation magnitudes
    n_runs: number of runs for each level
    seed: random seed for reproducibility
    returns: mean and std of accuracies corresponding to levels
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    mean_accs = []
    std_accs = []
    
    for lvl in levels:
        run_accs = []
        for run in range(n_runs):
            # apply perturbation transform with different random seeds for each run
            perturbed = [perturb_fn(d.clone(), lvl) for d in test_ds]
            loader = DataLoader(perturbed, batch_size=batch_size)
            acc = evaluate(model, loader, device)
            run_accs.append(acc)
        
        mean_accs.append(np.mean(run_accs))
        std_accs.append(np.std(run_accs))
    
    return mean_accs, std_accs

# --------------------
# 3) Runner
# --------------------

def run_all(dataset_name='PROTEINS', n_runs=10, device=None, out_csv='robustness_results.csv'):
    """
    Runs robustness curves for all pooling variants and saves to CSV.
    Columns: model, perturbation, level, mean_accuracy, std_accuracy
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Running robustness analysis on {dataset_name} with {n_runs} runs per level")

    # load and split dataset like in main.py
    dataset = TUDataset(root='data', name=dataset_name).shuffle()
    split = int(0.8 * len(dataset))
    train_ds, test_ds = dataset[:split], dataset[split:]
    
    print(f"Dataset loaded: {len(dataset)} graphs ({len(train_ds)} train, {len(test_ds)} test)")

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
        if os.path.exists(path):
            print(f"Loading model weights from {path}")
            model.load_state_dict(torch.load(path, map_location=device))
        else:
            print(f"Warning: No pre-trained weights found at {path}")
        model.to(device)

    # define perturbations and levels (more granular levels for smoother curves)
    perturbations = {
        'edge_drop': drop_edges,
        'feat_noise': noisy_features,
        'node_drop': drop_nodes

    }
    levels = {
        'edge_drop': np.linspace(0.0, 0.3, 16),  # More fine-grained levels
        'feat_noise': np.linspace(0.0, 0.5, 16),   # More fine-grained levels
        'node_drop': np.linspace(0.0, 0.3, 16)   # More fine-grained levels
    }

    # collect results
    rows = []
    for pert_name, fn in perturbations.items():
        print(f"\nRunning {pert_name} experiments:")
        for var_name, model in variants.items():
            print(f"  Model: {var_name}")
            mean_accs, std_accs = robustness_curve(
                model, test_ds, device, fn, levels[pert_name], 
                n_runs=n_runs, seed=42
            )
            
            for i, lvl in enumerate(levels[pert_name]):
                rows.append({
                    'model': var_name,
                    'perturbation': pert_name,
                    'level': float(lvl),
                    'mean_accuracy': float(mean_accs[i]),
                    'std_accuracy': float(std_accs[i])
                })
                
    # save
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nRobustness results written to {out_csv}")
    
    # print summary
    summary = df.groupby(['model', 'perturbation']).agg(
        baseline_acc=('mean_accuracy', lambda x: x.iloc[0]),
        worst_acc=('mean_accuracy', 'min'),
        drop=('mean_accuracy', lambda x: x.iloc[0] - x.min())
    ).reset_index()
    
    print("\nSummary of robustness results:")
    print(summary)
    
    # Generate visualization if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set plot style
        sns.set(style="whitegrid")
        
        # Create a plot for each perturbation type
        for pert_name in perturbations.keys():
            plt.figure(figsize=(10, 6))
            
            # Filter data for this perturbation
            pert_df = df[df['perturbation'] == pert_name]
            
            # Plot for each model
            for var_name in variants.keys():
                model_df = pert_df[pert_df['model'] == var_name]
                plt.errorbar(
                    model_df['level'], 
                    model_df['mean_accuracy'],
                    yerr=model_df['std_accuracy'],
                    label=var_name, 
                    marker='o', 
                    capsize=4
                )
            
            plt.xlabel(f"Perturbation Level ({pert_name})")
            plt.ylabel("Accuracy")
            plt.title(f"Robustness to {pert_name}")
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plot_path = f"robustness_{dataset_name}_{pert_name}.png"
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
            
    except ImportError:
        print("\nMatplotlib and/or seaborn not available. Skipping visualization.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Robustness analysis for GNN pooling variants')
    parser.add_argument('--dataset', type=str, default='MUTAG', help='Dataset name (e.g., MUTAG, PROTEINS)')
    parser.add_argument('--out', type=str, default='robustness_results.csv', help='Output CSV filename')
    parser.add_argument('--runs', type=int, default=20, help='Number of runs per perturbation level')
    args = parser.parse_args()
    
    run_all(dataset_name=args.dataset, n_runs=args.runs, out_csv=args.out)