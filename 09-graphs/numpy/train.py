import numpy as np
import argparse
import os
import urllib.request
import zipfile

def load_cora_dataset():
    """
    Download and load Cora citation network dataset.
    
    Cora is a citation network where:
    - Nodes: 2708 scientific publications
    - Edges: 5429 citation links
    - Features: 1433-dim bag-of-words vectors
    - Classes: 7 categories (Case_Based, Genetic_Algorithms, Neural_Networks,
               Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory)
    
    Standard split: 140 train, 500 val, 1000 test nodes
    """
    cache_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(cache_dir, exist_ok=True)
    
    cora_dir = os.path.join(cache_dir, 'cora')
    content_file = os.path.join(cora_dir, 'cora.content')
    cites_file = os.path.join(cora_dir, 'cora.cites')
    
    if not os.path.exists(content_file):
        print("Downloading Cora dataset...")
        zip_path = os.path.join(cache_dir, 'cora.zip')
        url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
        
        try:
            urllib.request.urlretrieve(url, zip_path.replace('.zip', '.tgz'))
            import tarfile
            with tarfile.open(zip_path.replace('.zip', '.tgz'), 'r:gz') as tar:
                tar.extractall(cache_dir)
            print("Download complete!")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Creating synthetic Cora-like dataset...")
            return create_synthetic_dataset(2708, 1433, 7, 5429)
    
    idx_map = {}
    features = []
    labels = []
    label_map = {}
    
    with open(content_file, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            idx_map[parts[0]] = i
            features.append([float(x) for x in parts[1:-1]])
            label = parts[-1]
            if label not in label_map:
                label_map[label] = len(label_map)
            labels.append(label_map[label])
    
    X = np.array(features)
    y = np.array(labels)
    num_nodes = len(y)
    num_classes = len(label_map)
    
    adj = np.zeros((num_nodes, num_nodes))
    with open(cites_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] in idx_map and parts[1] in idx_map:
                i, j = idx_map[parts[0]], idx_map[parts[1]]
                adj[i, j] = 1
                adj[j, i] = 1
    
    return X, adj, y, num_classes


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
    adj = np.zeros((num_nodes, num_nodes))
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1
    
    labels = np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1])
    
    X = np.eye(num_nodes)
    
    return X, adj, labels, 2


def create_synthetic_dataset(num_nodes=500, num_features=32, num_classes=5, num_edges=2000):
    """
    Create synthetic graph dataset for testing.
    
    Generates a random graph with cluster structure.
    """
    print(f"Creating synthetic dataset: {num_nodes} nodes, {num_features} features, {num_classes} classes")
    
    labels = np.random.randint(0, num_classes, num_nodes)
    
    X = np.random.randn(num_nodes, num_features)
    for c in range(num_classes):
        mask = labels == c
        X[mask] += np.random.randn(num_features) * 2
    
    adj = np.zeros((num_nodes, num_nodes))
    
    for c in range(num_classes):
        nodes_in_class = np.where(labels == c)[0]
        intra_edges = min(num_edges // (num_classes * 2), len(nodes_in_class) * 3)
        for _ in range(intra_edges):
            if len(nodes_in_class) >= 2:
                i, j = np.random.choice(nodes_in_class, 2, replace=False)
                adj[i, j] = 1
                adj[j, i] = 1
    
    inter_edges = num_edges // 4
    for _ in range(inter_edges):
        i, j = np.random.randint(0, num_nodes, 2)
        if i != j:
            adj[i, j] = 1
            adj[j, i] = 1
    
    return X, adj, labels, num_classes


def load_dataset(dataset_name):
    """Load dataset by name."""
    if dataset_name == 'cora':
        return load_cora_dataset()
    elif dataset_name == 'karate':
        return load_karate_dataset()
    elif dataset_name == 'synthetic':
        return create_synthetic_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_train_val_test_masks(num_nodes, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Create random train/val/test splits."""
    np.random.seed(seed)
    indices = np.random.permutation(num_nodes)
    
    train_end = int(num_nodes * train_ratio)
    val_end = int(num_nodes * (train_ratio + val_ratio))
    
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    
    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True
    
    return train_mask, val_mask, test_mask


def one_hot_encode(labels, num_classes):
    """Convert labels to one-hot encoding."""
    n = len(labels)
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), labels] = 1
    return one_hot


def evaluate(model, X, adj, y_onehot, mask):
    """Evaluate model on masked nodes."""
    from utils import accuracy
    
    probs = model.forward(X, adj, training=False)
    acc = accuracy(probs[mask], y_onehot[mask])
    
    loss = -np.sum(y_onehot[mask] * np.log(probs[mask] + 1e-10)) / np.sum(mask)
    
    return loss, acc


def train_gnn(args):
    """Train basic GNN model."""
    from gnn import GNN
    
    print(f"\n{'='*60}")
    print(f"Training GNN on {args.dataset} dataset")
    print(f"{'='*60}")
    
    X, adj, y, num_classes = load_dataset(args.dataset)
    y_onehot = one_hot_encode(y, num_classes)
    train_mask, val_mask, test_mask = get_train_val_test_masks(len(y))
    
    print(f"Nodes: {X.shape[0]}, Features: {X.shape[1]}, Classes: {num_classes}")
    print(f"Edges: {int(np.sum(adj) / 2)}")
    print(f"Train: {np.sum(train_mask)}, Val: {np.sum(val_mask)}, Test: {np.sum(test_mask)}")
    
    model = GNN(X.shape[1], args.hidden_size, num_classes, 
                num_layers=args.num_layers, dropout=args.dropout)
    
    best_val_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        model.forward(X, adj, training=True)
        loss = model.backward(y_onehot, mask=train_mask)
        model.update(learning_rate=args.lr)
        
        if epoch % args.eval_every == 0:
            val_loss, val_acc = evaluate(model, X, adj, y_onehot, val_mask)
            print(f"Epoch {epoch}/{args.epochs}: "
                  f"Train Loss = {loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, "
                  f"Val Acc = {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
    
    test_loss, test_acc = evaluate(model, X, adj, y_onehot, test_mask)
    print(f"\n{'='*60}")
    print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")
    print(f"{'='*60}")


def train_gcn(args):
    """Train GCN model."""
    from gcn import GCN
    
    print(f"\n{'='*60}")
    print(f"Training GCN on {args.dataset} dataset")
    print(f"{'='*60}")
    
    X, adj, y, num_classes = load_dataset(args.dataset)
    y_onehot = one_hot_encode(y, num_classes)
    train_mask, val_mask, test_mask = get_train_val_test_masks(len(y))
    
    print(f"Nodes: {X.shape[0]}, Features: {X.shape[1]}, Classes: {num_classes}")
    print(f"Edges: {int(np.sum(adj) / 2)}")
    print(f"Train: {np.sum(train_mask)}, Val: {np.sum(val_mask)}, Test: {np.sum(test_mask)}")
    
    model = GCN(X.shape[1], args.hidden_size, num_classes,
                num_layers=args.num_layers, dropout=args.dropout)
    model.preprocess(adj)
    
    best_val_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        model.forward(X, training=True)
        loss = model.backward(y_onehot, mask=train_mask)
        model.update(learning_rate=args.lr)
        
        if epoch % args.eval_every == 0:
            val_loss, val_acc = evaluate(model, X, adj, y_onehot, val_mask)
            print(f"Epoch {epoch}/{args.epochs}: "
                  f"Train Loss = {loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, "
                  f"Val Acc = {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
    
    test_loss, test_acc = evaluate(model, X, adj, y_onehot, test_mask)
    print(f"\n{'='*60}")
    print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")
    print(f"{'='*60}")


def train_graphsage(args):
    """Train GraphSAGE model."""
    from graphsage import GraphSAGE
    
    print(f"\n{'='*60}")
    print(f"Training GraphSAGE ({args.aggregator}) on {args.dataset} dataset")
    print(f"{'='*60}")
    
    X, adj, y, num_classes = load_dataset(args.dataset)
    y_onehot = one_hot_encode(y, num_classes)
    train_mask, val_mask, test_mask = get_train_val_test_masks(len(y))
    
    print(f"Nodes: {X.shape[0]}, Features: {X.shape[1]}, Classes: {num_classes}")
    print(f"Edges: {int(np.sum(adj) / 2)}")
    print(f"Train: {np.sum(train_mask)}, Val: {np.sum(val_mask)}, Test: {np.sum(test_mask)}")
    
    model = GraphSAGE(X.shape[1], args.hidden_size, num_classes,
                      num_layers=args.num_layers, aggregator=args.aggregator,
                      sample_size=args.sample_size, dropout=args.dropout)
    
    best_val_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        model.forward(X, adj, training=True)
        loss = model.backward(y_onehot, mask=train_mask)
        model.update(learning_rate=args.lr)
        
        if epoch % args.eval_every == 0:
            val_loss, val_acc = evaluate(model, X, adj, y_onehot, val_mask)
            print(f"Epoch {epoch}/{args.epochs}: "
                  f"Train Loss = {loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, "
                  f"Val Acc = {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
    
    test_loss, test_acc = evaluate(model, X, adj, y_onehot, test_mask)
    print(f"\n{'='*60}")
    print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")
    print(f"{'='*60}")


def train_gat(args):
    """Train GAT model."""
    from gat import GAT
    
    print(f"\n{'='*60}")
    print(f"Training GAT on {args.dataset} dataset")
    print(f"{'='*60}")
    
    X, adj, y, num_classes = load_dataset(args.dataset)
    y_onehot = one_hot_encode(y, num_classes)
    train_mask, val_mask, test_mask = get_train_val_test_masks(len(y))
    
    print(f"Nodes: {X.shape[0]}, Features: {X.shape[1]}, Classes: {num_classes}")
    print(f"Edges: {int(np.sum(adj) / 2)}")
    print(f"Train: {np.sum(train_mask)}, Val: {np.sum(val_mask)}, Test: {np.sum(test_mask)}")
    
    model = GAT(X.shape[1], args.hidden_size, num_classes,
                num_layers=args.num_layers, num_heads=args.num_heads,
                dropout=args.dropout, attn_dropout=args.attn_dropout)
    
    best_val_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        model.forward(X, adj, training=True)
        loss = model.backward(y_onehot, mask=train_mask)
        model.update(learning_rate=args.lr)
        
        if epoch % args.eval_every == 0:
            val_loss, val_acc = evaluate(model, X, adj, y_onehot, val_mask)
            print(f"Epoch {epoch}/{args.epochs}: "
                  f"Train Loss = {loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, "
                  f"Val Acc = {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
    
    test_loss, test_acc = evaluate(model, X, adj, y_onehot, test_mask)
    print(f"\n{'='*60}")
    print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Train graph neural networks')
    parser.add_argument('--model', type=str, required=True,
                       choices=['gnn', 'gcn', 'graphsage', 'gat'],
                       help='Model type to train')
    parser.add_argument('--dataset', type=str, default='karate',
                       choices=['cora', 'karate', 'synthetic'],
                       help='Dataset to use')
    parser.add_argument('--hidden-size', type=int, default=16,
                       help='Hidden layer size')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of GNN layers')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--eval-every', type=int, default=10,
                       help='Evaluate every N epochs')
    parser.add_argument('--aggregator', type=str, default='mean',
                       choices=['mean', 'maxpool'],
                       help='Aggregator for GraphSAGE')
    parser.add_argument('--sample-size', type=int, default=10,
                       help='Neighbor sample size for GraphSAGE')
    parser.add_argument('--num-heads', type=int, default=4,
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
