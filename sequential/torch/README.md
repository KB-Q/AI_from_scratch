# Sequential Models (PyTorch)

PyTorch implementations of sequential models, mirroring the numpy implementations in `sequential/`.

## Models

| Model | Description | File |
|-------|-------------|------|
| **RNN** | Vanilla Recurrent Neural Network | `rnn.py` |
| **LSTM** | Long Short-Term Memory | `lstm.py` |
| **GRU** | Gated Recurrent Unit | `gru.py` |
| **BidirectionalLSTM** | Bidirectional LSTM | `lstm.py` |
| **BidirectionalGRU** | Bidirectional GRU | `gru.py` |
| **Word2Vec** | Skip-gram and CBOW embeddings | `word2vec.py` |
| **CoLES** | Contrastive Learning for Event Sequences | `coles.py` |

## Datasets

The PyTorch version supports larger datasets compared to the numpy version:

| Dataset | Description | Size |
|---------|-------------|------|
| `wikitext2` | WikiText-2 language modeling | ~2M tokens |
| `shakespeare` | Tiny Shakespeare | ~1M chars |
| `ptb` | Penn Treebank | ~900K tokens |
| `simple` | Simple repeated text | Toy dataset |

## Usage

### Training Language Models (RNN/LSTM/GRU)

```bash
# Train LSTM on WikiText-2
python train.py --model lstm --dataset wikitext2 --epochs 20 --hidden-size 256 --batch-size 64

# Train Bidirectional GRU on Shakespeare with text generation
python train.py --model bigru --dataset shakespeare --epochs 10 --generate

# Train RNN with FFN output layer
python train.py --model rnn --dataset wikitext2 --use-ffn --lr 0.001
```

### Training Word2Vec

```bash
# Skip-gram
python train.py --model word2vec --dataset wikitext2 --w2v-method skipgram --embedding-dim 128 --window-size 5

# CBOW
python train.py --model word2vec --dataset shakespeare --w2v-method cbow --epochs 10
```

### Training CoLES

```bash
# Train CoLES with GRU encoder
python train.py --model coles --dataset wikitext2 --coles-cell gru --coles-bidirectional

# Customize subsequence sampling
python train.py --model coles --coles-seq-len 100 --coles-subseq-len 20 --coles-temperature 0.1
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Model type: rnn, lstm, gru, bilstm, bigru, word2vec, coles |
| `--dataset` | wikitext2 | Dataset to use |
| `--max-chars` | 500000 | Maximum characters from dataset |
| `--seq-len` | 50 | Sequence length for RNN models |
| `--hidden-size` | 256 | Hidden layer size |
| `--epochs` | 20 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--use-ffn` | False | Use FFN output layer |
| `--generate` | False | Generate text after training |

## GPU Support

The models automatically detect and use the best available device:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU (fallback)

## Example: Using Models Directly

```python
import torch
from sequential_torch import LSTM, Word2Vec, CoLES

# Language modeling with LSTM
model = LSTM(vocab_size=100, hidden_size=256, output_size=100)
x = torch.randn(32, 50, 100)  # (batch, seq_len, vocab_size)
outputs, (h, c) = model(x)

# Word embeddings
w2v = Word2Vec(vocab_size=10000, embedding_dim=128, method='skipgram')
embedding = w2v.get_embedding(42)

# Contrastive learning
coles = CoLES(vocab_size=100, hidden_size=128, embedding_dim=64)
seq = torch.randint(0, 100, (50,))
embedding = coles.get_embedding(seq)
```
