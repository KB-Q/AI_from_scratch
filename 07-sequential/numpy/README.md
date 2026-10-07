# Sequential Models

Clean implementations of RNN, LSTM, GRU, Bidirectional variants, and Word2Vec models from scratch using NumPy.

## Models

- **SimpleRNN**: Vanilla RNN with tanh activation
- **LSTM**: Long Short-Term Memory with optional FFN output layer
- **GRU**: Gated Recurrent Unit with optional FFN output layer
- **BidirectionalLSTM**: Bidirectional LSTM processing sequences forward and backward
- **BidirectionalGRU**: Bidirectional GRU processing sequences forward and backward
- **Word2Vec**: Skip-gram and CBOW word embedding methods

## Training

Use the unified `train.py` script to train any model:

### RNN
```bash
# Without FFN output layer
python train.py --model rnn --dataset shakespeare --seq-len 30 --hidden-size 64 --epochs 30 --lr 0.01

# With FFN output layer
python train.py --model rnn --dataset shakespeare --seq-len 30 --hidden-size 64 --epochs 30 --lr 0.01 --use-ffn
```

### LSTM
```bash
python train.py --model lstm --dataset shakespeare --seq-len 25 --hidden-size 64 --epochs 30 --lr 0.005 --use-ffn
```

### GRU
```bash
# Without FFN output layer
python train.py --model gru --dataset shakespeare --seq-len 30 --hidden-size 64 --epochs 30 --lr 0.01

# With FFN output layer
python train.py --model gru --dataset shakespeare --seq-len 25 --hidden-size 64 --epochs 30 --lr 0.005 --use-ffn
```

### Bidirectional LSTM
```bash
python train.py --model bilstm --dataset shakespeare --seq-len 20 --hidden-size 32 --epochs 30 --lr 0.005
```

### Bidirectional GRU
```bash
python train.py --model bigru --dataset shakespeare --seq-len 20 --hidden-size 32 --epochs 30 --lr 0.005
```

### Word2Vec
```bash
# Skip-gram
python train.py --model word2vec --dataset shakespeare --w2v-method skipgram --embedding-dim 50 --window-size 3 --epochs 30 --lr 0.05

# CBOW
python train.py --model word2vec --dataset shakespeare --w2v-method cbow --embedding-dim 50 --window-size 3 --epochs 30 --lr 0.05
```

## Arguments

- `--model`: Model type (`rnn`, `lstm`, `gru`, `bilstm`, `bigru`, `word2vec`)
- `--dataset`: Dataset to use (`shakespeare`, `linux`, `simple`)
- `--seq-len`: Sequence length for RNN/LSTM models
- `--hidden-size`: Hidden layer size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--eval-every`: Evaluate on validation set every N epochs
- `--use-ffn`: Use FFN output layer (applies to RNN and LSTM)
- `--generate`: Generate sample text after training
- `--w2v-method`: Word2Vec method (`skipgram`, `cbow`)
- `--embedding-dim`: Embedding dimension for Word2Vec
- `--window-size`: Context window size for Word2Vec

## Datasets

The script automatically downloads datasets:
- **shakespeare**: Tiny Shakespeare corpus (~1MB, character-level text generation)
- **linux**: Linux kernel source code (~6MB, code generation)
- **timeseries**: Airline passengers dataset (~2KB, time series forecasting - monthly totals 1949-1960)
- **simple**: Small test dataset

Datasets are cached in `data/` directory after first download.

## Data Split

Data is automatically split into:
- **Train**: 80%
- **Validation**: 10%
- **Test**: 10%

Models are evaluated on test set after training to measure final performance on unseen data.

## File Structure

- `train.py`: Unified training script with CLI and data loading
- `rnn.py`: SimpleRNN implementation
- `lstm.py`: LSTM and BidirectionalLSTM implementations
- `gru.py`: GRU and BidirectionalGRU implementations
- `word2vec.py`: Word2Vec implementation
- `utils.py`: Activation functions and utilities
- `bleu.py`: BLEU score metric for evaluation

## Model Explanations

### 1. Simple RNN

**Formulas:**
```
h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b_h)
y_t = W_hy·h_t + b_y
```

**Description:**
The simplest recurrent architecture that maintains a hidden state updated at each time step. Uses tanh activation to introduce non-linearity.

**Pros:**
- Simple and easy to understand
- Fast to train due to fewer parameters
- Good baseline for sequential tasks

**Cons:**
- Suffers from vanishing/exploding gradients on long sequences
- Limited ability to capture long-term dependencies
- Struggles with sequences longer than ~10-20 steps

### 2. LSTM (Long Short-Term Memory)

**Formulas:**
```
f_t = σ(W_f·[h_{t-1}, x_t] + b_f)           # Forget gate
i_t = σ(W_i·[h_{t-1}, x_t] + b_i)           # Input gate
c̃_t = tanh(W_c·[h_{t-1}, x_t] + b_c)        # Cell candidate
o_t = σ(W_o·[h_{t-1}, x_t] + b_o)           # Output gate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t             # Cell state
h_t = o_t ⊙ tanh(c_t)                        # Hidden state
```

**Description:**
Introduces a cell state and three gates (forget, input, output) to control information flow. The cell state acts as a highway for gradients, enabling learning of long-term dependencies.

**Pros:**
- Effectively handles long-term dependencies (100+ steps)
- Mitigates vanishing gradient problem via cell state
- State-of-the-art performance on many sequential tasks
- Robust and well-studied architecture

**Cons:**
- More parameters (~4x compared to RNN) leading to slower training
- Higher computational cost
- More complex to implement and debug
- Can overfit on smaller datasets

### 3. GRU (Gated Recurrent Unit)

**Formulas:**
```
z_t = σ(W_z·[h_{t-1}, x_t] + b_z)            # Update gate
r_t = σ(W_r·[h_{t-1}, x_t] + b_r)            # Reset gate
h̃_t = tanh(W_h·[r_t ⊙ h_{t-1}, x_t] + b_h)  # Hidden candidate
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t       # Hidden state
```

**Description:**
Simplified variant of LSTM with two gates (update and reset) instead of three. Merges cell state and hidden state into a single hidden state, reducing parameters while maintaining similar performance.

**Pros:**
- Fewer parameters (~3x compared to RNN vs LSTM's 4x)
- Faster training and inference than LSTM
- Often performs comparably to LSTM on many tasks
- Easier to implement and tune
- Better on smaller datasets due to fewer parameters

**Cons:**
- Slightly less expressive than LSTM
- May underperform LSTM on very long sequences
- Less control over memory retention compared to LSTM's separate cell state

### 4. Bidirectional LSTM/GRU

**Formulas:**
```
h_fwd_t = LSTM/GRU_forward(x_1, ..., x_t)
h_bwd_t = LSTM/GRU_backward(x_T, ..., x_t)
y_t = FFN([h_fwd_t, h_bwd_t])
```

**Description:**
Processes sequences in both forward and backward directions, then combines outputs. Captures context from both past and future, useful when entire sequence is available.

**Pros:**
- Access to full sequence context (past + future)
- Superior performance on tasks with bidirectional context (e.g., NER, sentiment)
- Better feature extraction for classification tasks
- Can capture dependencies from both directions

**Cons:**
- Cannot be used for online/real-time prediction
- Approximately 2x slower due to dual processing
- 2x memory requirement for hidden states
- Not suitable for autoregressive generation tasks

### 5. Word2Vec

**Skip-gram Formula:**
```
Maximize: Σ log P(context | center_word)
P(o|c) = exp(u_o^T v_c) / Σ_w exp(u_w^T v_c)
```

**CBOW Formula:**
```
Maximize: Σ log P(center_word | context)
P(c|o_1,...,o_m) = exp(u_c^T v̄) / Σ_w exp(u_w^T v̄)
where v̄ = (1/m) Σ v_o_i
```

**Description:**
Learn distributed word representations by predicting context words (skip-gram) or center word (CBOW) in a sliding window. Captures semantic relationships in vector space.

**Pros:**
- Captures semantic similarity ("king" - "man" + "woman" ≈ "queen")
- Efficient unsupervised learning from large corpora
- Transfer learning: embeddings work across tasks
- Relatively fast training

**Cons:**
- No handling of out-of-vocabulary words
- Fixed embeddings (no contextualization)
- Requires large corpus for quality embeddings
- Skip-gram: slower but better for rare words
- CBOW: faster but smooths over distributional information

## Performance Comparison

| Model | Parameters | Speed | Long Dependencies | When to Use |
|-------|-----------|-------|------------------|-------------|
| RNN | Low | Fast | Poor | Short sequences, baseline |
| LSTM | High | Slow | Excellent | Long sequences, complex patterns |
| GRU | Medium | Medium | Very Good | Balance of speed and performance |
| BiLSTM/BiGRU | High | Slow | Excellent | Full sequence available, classification |
| Word2Vec | Medium | Fast | N/A | Static word embeddings |
