import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import urllib.request
import tarfile
import numpy as np
from utils import get_device, accuracy

def load_planetoid_dataset(dataset_name='cora'):
    """
    Download and load Planetoid datasets (Cora, CiteSeer, PubMed).
    
    Cora: 2708 nodes, 5429 edges, 1433 features, 7 classes
    CiteSeer: 3327 nodes, 4732 edges, 3703 features, 6 classes
    PubMed: 19717 nodes, 44338 edges, 500 features, 3 classes
    """
    cache_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(cache_dir, exist_ok=True)
    
    dataset_dir = os.path.join(cache_dir, dataset_name)
    
    if not os.path.exists(dataset_dir):
        print(f"Downloading {dataset_name} dataset...")
        os.makedirs(dataset_dir, exist_ok=True)
        
        base_url = 'https://github.com/kimiyoung/planetoid/raw/master/data/'
        files = [
            f'ind.{dataset_name}.x',
            f'ind.{dataset_name}.y',
            f'ind.{dataset_name}.tx',
            f'ind.{dataset_name}.ty',
            f'ind.{dataset_name}.allx',
            f'ind.{dataset_name}.ally',
            f'ind.{dataset_name}.graph',
            f'ind.{dataset_name}.test.index'
        ]
        
        try:
            import pickle
            for fname in files:
                url = base_url + fname
                filepath = os.path.join(dataset_dir, fname)
                urllib.request.urlretrieve(url, filepath)
            print("Download complete!")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Creating synthetic dataset...")
            return create_synthetic_dataset(2708, 1433, 7, 5429)
    
    try:
        import pickle
        
        def load_pickle(name):
            with open(os.path.join(dataset_dir, f'ind.{dataset_name}.{name}'), 'rb') as f:
                try:
                    return pickle.load(f, encoding='latin1')
                except:
                    return pickle.load(f)
        
        x = load_pickle('x')
        y = load_pickle('y')
        tx = load_pickle('tx')
        ty = load_pickle('ty')
        allx = load_pickle('allx')
        ally = load_pickle('ally')
        graph = load_pickle('graph')
        
        with open(os.path.join(dataset_dir, f'ind.{dataset_name}.test.index'), 'r') as f:
            test_idx = [int(line.strip()) for line in f]
        
        if hasattr(x, 'toarray'):
            x = x.toarray()
        if hasattr(tx, 'toarray'):
            tx = tx.toarray()
        if hasattr(allx, 'toarray'):
            allx = allx.toarray()
        
        if dataset_name == 'citeseer':
            test_idx_sorted = sorted(test_idx)
            tx_extended = np.zeros((max(test_idx) - min(test_idx) + 1, tx.shape[1]))
            ty_extended = np.zeros((max(test_idx) - min(test_idx) + 1, ty.shape[1]))
            tx_extended[test_idx_sorted - min(test_idx_sorted)] = tx
            ty_extended[test_idx_sorted - min(test_idx_sorted)] = ty
            tx, ty = tx_extended, ty_extended
        
        features = np.vstack([allx, tx])
        labels = np.vstack([ally, ty])
        
        test_idx_reorder = test_idx
        features[test_idx_reorder] = features[sorted(test_idx_reorder)]
        labels[test_idx_reorder] = labels[sorted(test_idx_reorder)]
        
        num_nodes = features.shape[0]
        adj = np.zeros((num_nodes, num_nodes))
        for i, neighbors in graph.items():
            for j in neighbors:
                if i < num_nodes and j < num_nodes:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        labels = np.argmax(labels, axis=1)
        num_classes = len(np.unique(labels))
        
        X = torch.FloatTensor(features)
        adj = torch.FloatTensor(adj)
        y = torch.LongTensor(labels)
        
        return X, adj, y, num_classes
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating synthetic dataset...")
        return create_synthetic_dataset(2708, 1433, 7, 5429)


def load_karate_dataset():
    """
    Load Zachary's Karate Club dataset.
    
    Classic social network dataset:
    - Nodes: 34 members of a karate club
    - Edges: 78 friendship links
    - Classes: 2 (split into two groups after dispute)
    
    Good for quick testing due to small size.
    """
    edges = [
        (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),(0,12),(0,13),
        (0,17),(0,19),(0,21),(0,31),(1,2),(1,3),(1,7),(1,13),(1,17),(1,19),(1,21),(1,30),
        (2,3),(2,7),(2,8),(2,9),(2,13),(2,27),(2,28),(2,32),(3,7),(3,12),(3,13),(4,6),(4,10),
        (5,6),(5,10),(5,16),(6,16),(8,30),(8,32),(8,33),(9,33),(13,33),(14,32),(14,33),
        (15,32),(15,33),(18,32),(18,33),(19,33),(20,32),(20,33),(22,32),(22,33),(23,25),
        (23,27),(23,29),(23,32),(23,33),(24,25),(24,27),(24,31),(25,31),(26,29),(26,33),
        (27,33),(28,31),(28,33),(29,32),(29,33),(30,32),(30,33),(31,32),(31,33),(32,33)
    ]
    
    num_nodes = 34
    adj = torch.zeros(num_nodes, num_nodes)
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1
    
    labels = torch.tensor([0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1])
    
    X = torch.eye(num_nodes)
    
    return X, adj, labels, 2


def create_synthetic_dataset(num_nodes=1000, num_features=64, num_classes=5, num_edges=5000):
    """
    Create synthetic graph dataset for testing.
    
    Generates a random graph with cluster structure.
    """
    print(f"Creating synthetic dataset: {num_nodes} nodes, {num_features} features, {num_classes} classes")
    
    labels = torch.randint(0, num_classes, (num_nodes,))
    
    X = torch.randn(num_nodes, num_features)
    for c in range(num_classes):
        mask = labels == c
        X[mask] += torch.randn(num_features) * 2
    
    adj = torch.zeros(num_nodes, num_nodes)
    
    for c in range(num_classes):
        nodes_in_class = torch.where(labels == c)[0]
        intra_edges = min(num_edges // (num_classes * 2), len(nodes_in_class) * 3)
        for _ in range(intra_edges):
            if len(nodes_in_class) >= 2:
                perm = torch.randperm(len(nodes_in_class))[:2]
                i, j = nodes_in_class[perm[0]].item(), nodes_in_class[perm[1]].item()
                adj[i, j] = 1
                adj[j, i] = 1
    
    inter_edges = num_edges // 4
    for _ in range(inter_edges):
        i, j = torch.randint(0, num_nodes, (2,))
        if i != j:
            adj[i.item(), j.item()] = 1
            adj[j.item(), i.item()] = 1
    
    return X, adj, labels, num_classes


def load_dataset(dataset_name):
    """Load dataset by name."""
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        return load_planetoid_dataset(dataset_name)
    elif dataset_name == 'karate':
        return load_karate_dataset()
    elif dataset_name == 'synthetic':
        return create_synthetic_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_train_val_test_masks(num_nodes, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Create random train/val/test splits."""
    torch.manual_seed(seed)
    indices = torch.randperm(num_nodes)
    
    train_end = int(num_nodes * train_ratio)
    val_end = int(num_nodes * (train_ratio + val_ratio))
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True
    
    return train_mask, val_mask, test_mask


def evaluate(model, X, adj, y, mask):
    """Evaluate model on masked nodes."""
    model.eval()
    with torch.no_grad():
        out = model(X, adj)
        pred = out[mask]
        target = y[mask]
        
        loss = F.nll_loss(pred, target).item()
        acc = (pred.argmax(dim=1) == target).float().mean().item()
    
    return loss, acc


def train_gnn(args):
    """Train basic GNN model."""
    from gnn import GNN
    
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Training GNN on {args.dataset} dataset")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    X, adj, y, num_classes = load_dataset(args.dataset)
    X, adj, y = X.to(device), adj.to(device), y.to(device)
    
    train_mask, val_mask, test_mask = get_train_val_test_masks(len(y))
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)
    
    print(f"Nodes: {X.shape[0]}, Features: {X.shape[1]}, Classes: {num_classes}")
    print(f"Edges: {int(adj.sum().item() / 2)}")
    print(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
    
    model = GNN(X.shape[1], args.hidden_size, num_classes, 
                num_layers=args.num_layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        out = model(X, adj)
        loss = F.nll_loss(out[train_mask], y[train_mask])
        
        loss.backward()
        optimizer.step()
        
        if epoch % args.eval_every == 0:
            val_loss, val_acc = evaluate(model, X, adj, y, val_mask)
            print(f"Epoch {epoch}/{args.epochs}: "
                  f"Train Loss = {loss.item():.4f}, "
                  f"Val Loss = {val_loss:.4f}, "
                  f"Val Acc = {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
    
    test_loss, test_acc = evaluate(model, X, adj, y, test_mask)
    print(f"\n{'='*60}")
    print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")
    print(f"{'='*60}")


def train_gcn(args):
    """Train GCN model."""
    from gcn import GCN
    
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Training GCN on {args.dataset} dataset")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    X, adj, y, num_classes = load_dataset(args.dataset)
    X, adj, y = X.to(device), adj.to(device), y.to(device)
    
    train_mask, val_mask, test_mask = get_train_val_test_masks(len(y))
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)
    
    print(f"Nodes: {X.shape[0]}, Features: {X.shape[1]}, Classes: {num_classes}")
    print(f"Edges: {int(adj.sum().item() / 2)}")
    print(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
    
    model = GCN(X.shape[1], args.hidden_size, num_classes,
                num_layers=args.num_layers, dropout=args.dropout).to(device)
    model.preprocess(adj)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        out = model(X)
        loss = F.nll_loss(out[train_mask], y[train_mask])
        
        loss.backward()
        optimizer.step()
        
        if epoch % args.eval_every == 0:
            val_loss, val_acc = evaluate(model, X, adj, y, val_mask)
            print(f"Epoch {epoch}/{args.epochs}: "
                  f"Train Loss = {loss.item():.4f}, "
                  f"Val Loss = {val_loss:.4f}, "
                  f"Val Acc = {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
    
    test_loss, test_acc = evaluate(model, X, adj, y, test_mask)
    print(f"\n{'='*60}")
    print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")
    print(f"{'='*60}")


def train_graphsage(args):
    """Train GraphSAGE model."""
    from graphsage import GraphSAGE
    
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Training GraphSAGE ({args.aggregator}) on {args.dataset} dataset")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    X, adj, y, num_classes = load_dataset(args.dataset)
    X, adj, y = X.to(device), adj.to(device), y.to(device)
    
    train_mask, val_mask, test_mask = get_train_val_test_masks(len(y))
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)
    
    print(f"Nodes: {X.shape[0]}, Features: {X.shape[1]}, Classes: {num_classes}")
    print(f"Edges: {int(adj.sum().item() / 2)}")
    print(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
    
    model = GraphSAGE(X.shape[1], args.hidden_size, num_classes,
                      num_layers=args.num_layers, aggregator=args.aggregator,
                      sample_size=args.sample_size, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        out = model(X, adj)
        loss = F.nll_loss(out[train_mask], y[train_mask])
        
        loss.backward()
        optimizer.step()
        
        if epoch % args.eval_every == 0:
            val_loss, val_acc = evaluate(model, X, adj, y, val_mask)
            print(f"Epoch {epoch}/{args.epochs}: "
                  f"Train Loss = {loss.item():.4f}, "
                  f"Val Loss = {val_loss:.4f}, "
                  f"Val Acc = {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
    
    test_loss, test_acc = evaluate(model, X, adj, y, test_mask)
    print(f"\n{'='*60}")
    print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")
    print(f"{'='*60}")


def train_gat(args):
    """Train GAT model."""
    from gat import GAT
    
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Training GAT on {args.dataset} dataset")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    X, adj, y, num_classes = load_dataset(args.dataset)
    X, adj, y = X.to(device), adj.to(device), y.to(device)
    
    train_mask, val_mask, test_mask = get_train_val_test_masks(len(y))
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)
    
    print(f"Nodes: {X.shape[0]}, Features: {X.shape[1]}, Classes: {num_classes}")
    print(f"Edges: {int(adj.sum().item() / 2)}")
    print(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
    
    model = GAT(X.shape[1], args.hidden_size, num_classes,
                num_layers=args.num_layers, num_heads=args.num_heads,
                dropout=args.dropout, attn_dropout=args.attn_dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        out = model(X, adj)
        loss = F.nll_loss(out[train_mask], y[train_mask])
        
        loss.backward()
        optimizer.step()
        
        if epoch % args.eval_every == 0:
            val_loss, val_acc = evaluate(model, X, adj, y, val_mask)
            print(f"Epoch {epoch}/{args.epochs}: "
                  f"Train Loss = {loss.item():.4f}, "
                  f"Val Loss = {val_loss:.4f}, "
                  f"Val Acc = {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
    
    test_loss, test_acc = evaluate(model, X, adj, y, test_mask)
    print(f"\n{'='*60}")
    print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Train graph neural networks (PyTorch)')
    parser.add_argument('--model', type=str, required=True,
                       choices=['gnn', 'gcn', 'graphsage', 'gat'],
                       help='Model type to train')
    parser.add_argument('--dataset', type=str, default='cora',
                       choices=['cora', 'citeseer', 'pubmed', 'karate', 'synthetic'],
                       help='Dataset to use')
    parser.add_argument('--hidden-size', type=int, default=64,
                       help='Hidden layer size')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of GNN layers')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--eval-every', type=int, default=10,
                       help='Evaluate every N epochs')
    parser.add_argument('--aggregator', type=str, default='mean',
                       choices=['mean', 'maxpool'],
                       help='Aggregator for GraphSAGE')
    parser.add_argument('--sample-size', type=int, default=10,
                       help='Neighbor sample size for GraphSAGE')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads for GAT')
    parser.add_argument('--attn-dropout', type=float, default=0.6,
                       help='Attention dropout for GAT')
    
    args = parser.parse_args()
    
    if args.model == 'gnn':
        train_gnn(args)
    elif args.model == 'gcn':
        train_gcn(args)
    elif args.model == 'graphsage':
        train_graphsage(args)
    elif args.model == 'gat':
        train_gat(args)


if __name__ == "__main__":
    main()
