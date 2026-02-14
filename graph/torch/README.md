# Graph Neural Networks (PyTorch)

PyTorch implementations of graph neural networks, mirroring the numpy implementations in `graph/`.

## Models

| Model | Description | File | Reference |
|-------|-------------|------|-----------|
| **GNN** | Basic Message Passing Network | `gnn.py` | - |
| **GCN** | Graph Convolutional Network | `gcn.py` | Kipf & Welling, 2017 |
| **GAT** | Graph Attention Network | `gat.py` | Veličković et al., 2018 |
| **GraphSAGE** | Sample and Aggregate | `graphsage.py` | Hamilton et al., 2017 |

## Datasets

The PyTorch version supports standard benchmark datasets:

| Dataset | Nodes | Edges | Features | Classes | Description |
|---------|-------|-------|----------|---------|-------------|
| `cora` | 2,708 | 5,429 | 1,433 | 7 | Citation network |
| `citeseer` | 3,327 | 4,732 | 3,703 | 6 | Citation network |
| `pubmed` | 19,717 | 44,338 | 500 | 3 | Citation network |
| `karate` | 34 | 78 | 34 | 2 | Social network |
| `synthetic` | 1,000 | 5,000 | 64 | 5 | Random clustered graph |

## Usage

### Training GCN

```bash
# Train on Cora (default)
python train.py --model gcn --dataset cora --epochs 200

# Train on CiteSeer with custom hyperparameters
python train.py --model gcn --dataset citeseer --hidden-size 64 --dropout 0.5 --lr 0.01

# Train on PubMed (larger dataset)
python train.py --model gcn --dataset pubmed --epochs 100 --hidden-size 128
```

### Training GAT

```bash
# Train GAT with 8 attention heads
python train.py --model gat --dataset cora --num-heads 8 --dropout 0.6

# Customize attention dropout
python train.py --model gat --dataset citeseer --attn-dropout 0.6 --epochs 200
```

### Training GraphSAGE

```bash
# Mean aggregator
python train.py --model graphsage --dataset cora --aggregator mean --sample-size 10

# Max pooling aggregator
python train.py --model graphsage --dataset citeseer --aggregator maxpool
```

### Training Basic GNN

```bash
python train.py --model gnn --dataset karate --epochs 100 --hidden-size 16
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Model type: gnn, gcn, graphsage, gat |
| `--dataset` | cora | Dataset to use |
| `--hidden-size` | 64 | Hidden layer size |
| `--num-layers` | 2 | Number of GNN layers |
| `--epochs` | 200 | Training epochs |
| `--lr` | 0.01 | Learning rate |
| `--weight-decay` | 5e-4 | L2 regularization |
| `--dropout` | 0.5 | Dropout rate |
| `--eval-every` | 10 | Evaluation frequency |
| `--aggregator` | mean | GraphSAGE aggregator |
| `--sample-size` | 10 | GraphSAGE neighbor sample size |
| `--num-heads` | 8 | GAT attention heads |
| `--attn-dropout` | 0.6 | GAT attention dropout |

## GPU Support

The models automatically detect and use the best available device:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU (fallback)

## Example: Using Models Directly

```python
import torch
from graph_torch import GCN, GAT, GraphSAGE

# Node classification with GCN
model = GCN(input_dim=1433, hidden_dim=64, output_dim=7, num_layers=2, dropout=0.5)
X = torch.randn(2708, 1433)  # Node features
adj = torch.randint(0, 2, (2708, 2708)).float()  # Adjacency matrix
model.preprocess(adj)  # Compute normalized adjacency
out = model(X)  # (2708, 7) log-softmax predictions

# Graph Attention Network
gat = GAT(input_dim=1433, hidden_dim=8, output_dim=7, num_heads=8)
out = gat(X, adj)

# GraphSAGE with sampling
sage = GraphSAGE(input_dim=1433, hidden_dim=64, output_dim=7, aggregator='mean')
out = sage(X, adj)
```

## Model Architectures

### GCN Layer
```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
```
- Symmetric normalization for spectral convolution
- O(|E|) complexity per layer

### GAT Layer
```
α_ij = softmax(LeakyReLU(a^T · [Wh_i || Wh_j]))
h_i' = σ(Σ_j α_ij · Wh_j)
```
- Learned attention weights
- Multi-head attention for stability

### GraphSAGE Layer
```
h_N(v) = AGGREGATE({h_u : u ∈ N(v)})
h_v' = σ(W · CONCAT(h_v, h_N(v)))
```
- Neighbor sampling for scalability
- Mean or max-pooling aggregation
