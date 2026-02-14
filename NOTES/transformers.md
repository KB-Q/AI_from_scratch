
**GPT-3** 

- uses self attention, 96 attention blocks + 96 FFNs
- each attention block - 96 attention heads (MHA)

### Toy example:

**Input**: 

- words, lets say context size 8
- converted into 12288 dimensional embeddings (individual + positional combined
- $E = [e1, e2, ... e8]$ - embedding vectors of shape: (12288, 8)

**single attention head**: 

- $W_K$  - key weight matrix of shape (128, 12288)
- $W_Q$  - query weight matrix of shape (128, 12288)
- $K = W_K • E = [k1, k2, ... k8]$ - key vectors of shape = (128, 8)
- $Q = W_Q • E = [q1, q2, ... q8]$ - query vectors of shape = (128, 8)
- $A = softmax(K^T • Q / sqrt(d_k))$ - attention matrix of shape = (8,8)
- now apply future masking (for self attention only)
- $W_V = W_{V_u} • W_{V_d}$ - value weight matrices of shapes (12288, 128) and (128, 12288)
- $V = W_V • E = [v1, v2, ... v8]$ - value vectors of shape (12288, 8)
- $E' = E + V • A$ - updated embedding vectors of shape (12288, 8)

$W_{V_u}$

**full formula**:

$$E' = E + W_{V_u} • W_{V_d} • E • softmax((W_K • E)^T • (W_Q • E) / sqrt(d_k))$$

$$E' = E + W_{V_{u}} \cdot W_{V_{d}} \cdot E \cdot \text{softmax}\left(\frac{(W_K \cdot E)^T \cdot (W_Q \cdot E)}{\sqrt{d_k}}\right)$$
