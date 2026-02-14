# Graph Neural Networks

Clean implementations of GNN, GCN, GraphSAGE, and GAT from scratch using NumPy.

## Models

- **GNN**: Basic Graph Neural Network with message passing
- **GCN**: Graph Convolutional Network (Kipf & Welling, 2017)
- **GraphSAGE**: Sample and Aggregate (Hamilton et al., 2017)
- **GAT**: Graph Attention Network (Veličković et al., 2018)

## Training

Use the unified `train.py` script to train any model:

### Basic GNN
```bash
python train.py --model gnn --dataset karate --hidden-size 16 --epochs 200 --lr 0.01
```

### GCN
```bash
python train.py --model gcn --dataset karate --hidden-size 16 --epochs 200 --lr 0.01 --dropout 0.5
```

### GraphSAGE
```bash
# Mean aggregator
python train.py --model graphsage --dataset karate --hidden-size 16 --epochs 200 --lr 0.01 --aggregator mean

# Max pooling aggregator
python train.py --model graphsage --dataset karate --hidden-size 16 --epochs 200 --lr 0.01 --aggregator maxpool
```

### GAT
```bash
python train.py --model gat --dataset karate --hidden-size 8 --epochs 200 --lr 0.005 --num-heads 4 --dropout 0.6
```

## Arguments

- `--model`: Model type (`gnn`, `gcn`, `graphsage`, `gat`)
- `--dataset`: Dataset to use (`cora`, `karate`, `synthetic`)
- `--hidden-size`: Hidden layer size
- `--num-layers`: Number of GNN layers
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--dropout`: Dropout rate
- `--eval-every`: Evaluate every N epochs
- `--aggregator`: Aggregator for GraphSAGE (`mean`, `maxpool`)
- `--sample-size`: Neighbor sample size for GraphSAGE
- `--num-heads`: Number of attention heads for GAT
- `--attn-dropout`: Attention dropout for GAT

## Datasets

The script supports the following datasets:
- **karate**: Zachary's Karate Club (34 nodes, 78 edges, 2 classes) - good for quick testing
- **cora**: Citation network (2708 nodes, 5429 edges, 7 classes) - standard benchmark
- **synthetic**: Random clustered graph (500 nodes, configurable)

Datasets are cached in `data/` directory after first download.

## Data Split

Data is automatically split into:
- **Train**: 60%
- **Validation**: 20%
- **Test**: 20%

## File Structure

- `train.py`: Unified training script with CLI and data loading
- `gnn.py`: Basic GNN implementation
- `gcn.py`: GCN implementation
- `graphsage.py`: GraphSAGE implementation
- `gat.py`: GAT implementation
- `utils.py`: Activation functions and utilities

## Model Explanations

### 1. Basic GNN (Message Passing Neural Network)

**Formulas:**
```
m_v = Σ_{u∈N(v)} W_msg · h_u          (aggregate neighbor messages)
h_v' = σ(W_self · h_v + m_v + b)      (update node representation)
```

**Description:**
The simplest form of GNN that learns to aggregate information from neighboring nodes. Each node collects messages from its neighbors, transforms them, and updates its own representation.

**Pros:**
- Simple and easy to understand
- Foundation for more complex architectures
- Flexible message passing framework

**Cons:**
- No normalization, can suffer from scale issues
- Limited expressiveness
- May over-smooth with many layers

### 2. GCN (Graph Convolutional Network)

**Formula:**
```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
```

Where Ã = A + I (adjacency with self-loops), D̃ = degree matrix of Ã

**Description:**
GCN approximates spectral graph convolutions using 1st-order Chebyshev polynomials. The symmetric normalization D̃^(-1/2) Ã D̃^(-1/2) prevents scale explosion and enables efficient computation.

**Pros:**
- Efficient: O(|E|) complexity per layer
- Principled: Spectral graph theory foundation
- Simple: Few hyperparameters
- Strong baseline performance

**Cons:**
- Transductive: Needs full graph at training
- Fixed receptive field per layer
- Over-smoothing with many layers (>3)
- Cannot handle dynamic graphs

### 3. GraphSAGE (Sample and Aggregate)

**Formula:**
```
h_N(v) = AGGREGATE({h_u : u ∈ S(N(v))})
h_v' = σ(W · CONCAT(h_v, h_N(v)))
```

Where S(N(v)) is a sampled subset of neighbors

**Description:**
GraphSAGE samples a fixed number of neighbors and aggregates their features. Key innovation is the CONCAT operation that explicitly preserves self-representation, enabling inductive learning.

**Pros:**
- Inductive: Works on unseen nodes/graphs
- Scalable: Fixed computation via sampling
- Flexible: Multiple aggregation strategies (mean, pool, LSTM)
- Mini-batch training possible

**Cons:**
- Sampling variance in training
- Information loss from sampling
- Hyperparameter: sample size selection
- Multiple aggregator choices to tune

### 4. GAT (Graph Attention Network)

**Formula:**
```
e_ij = LeakyReLU(a^T · [Wh_i || Wh_j])     (attention coefficients)
α_ij = softmax_j(e_ij)                      (normalized attention)
h_i' = σ(Σ_{j∈N(i)} α_ij · Wh_j)           (weighted aggregation)
```

**Description:**
GAT learns attention weights for each edge, allowing the model to focus on more important neighbors. Multi-head attention stabilizes learning and captures different aspects of the graph.

**Pros:**
- Learned attention: Adapts to graph structure
- Node-specific weights: Different importance per neighbor
- Multi-head: Stabilizes learning
- Interpretable: Attention weights show importance

**Cons:**
- O(N²) attention computation (can be sparse)
- More parameters than GCN
- Attention can be unstable without regularization
- Slower than GCN

## Performance Comparison

| Model | Complexity | Inductive | Key Feature | When to Use |
|-------|-----------|-----------|-------------|-------------|
| GNN | O(\|E\|) | No | Simple baseline | Learning, debugging |
| GCN | O(\|E\|) | No | Spectral theory | Fixed graphs, benchmarks |
| GraphSAGE | O(k^L·N) | Yes | Sampling | Large graphs, new nodes |
| GAT | O(\|E\|·F) | Yes | Attention | Heterogeneous importance |

Where k = sample size, L = layers, N = nodes, F = features

## Key Concepts

### Message Passing
All GNNs follow a message passing paradigm:
1. **Aggregate**: Collect information from neighbors
2. **Update**: Combine with self-representation
3. **Readout**: (Optional) Pool for graph-level tasks

### Over-smoothing
Deep GNNs (>3 layers) tend to make all node representations similar. Solutions:
- Residual connections
- Jumping knowledge
- Reduce layers

### Transductive vs Inductive
- **Transductive** (GCN): Train on fixed graph, cannot generalize to new nodes
- **Inductive** (GraphSAGE, GAT): Can generalize to unseen nodes and graphs
