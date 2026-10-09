import numpy as np
from utils import xavier_init
from rnn import RNNCell
from lstm import LSTMCell
from gru import GRUCell

CELL_TYPES = {'rnn': RNNCell, 'lstm': LSTMCell, 'gru': GRUCell}
CELL_PARAMS = {
    'rnn': ['W_xh', 'W_hh', 'b_h'],
    'lstm': ['W_f', 'W_i', 'W_c', 'W_o', 'b_f', 'b_i', 'b_c', 'b_o'],
    'gru': ['W_z', 'W_r', 'W_h', 'b_z', 'b_r', 'b_h']
}

class CoLESEncoder:
    """
    Sequence Encoder for CoLES. Configurable: RNN/LSTM/GRU, uni/bidirectional.
    z = proj(pool(h_1, ..., h_T))  or  z = proj([pool(h_fwd), pool(h_bwd)])
    """
    def __init__(self, input_size, hidden_size, embedding_dim, cell_type='gru', 
                 bidirectional=False, pooling='mean'):
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.param_keys = CELL_PARAMS[cell_type]
        
        self.fwd_cell = CELL_TYPES[cell_type](input_size, hidden_size)
        self.bwd_cell = CELL_TYPES[cell_type](input_size, hidden_size) if bidirectional else None
        
        proj_input = hidden_size * (2 if bidirectional else 1)
        self.W_proj = xavier_init((embedding_dim, proj_input))
        self.b_proj = np.zeros((embedding_dim, 1))

    def _run_cell(self, cell, x, reverse=False):
        """Run RNN cell over sequence, return hidden states and caches."""
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1)) if self.cell_type == 'lstm' else None
        states, caches = [], []
        seq = x[::-1] if reverse else x
        for x_t in seq:
            x_t = x_t.reshape(-1, 1)
            if self.cell_type == 'lstm':
                h, c, cache = cell.forward(x_t, h, c)
            else:
                h, cache = cell.forward(x_t, h)
            states.append(h)
            caches.append(cache)
        return (states[::-1], caches[::-1]) if reverse else (states, caches)

    def forward(self, x):
        """Forward pass: encode sequence into embedding."""
        self.fwd_states, self.fwd_caches = self._run_cell(self.fwd_cell, x)
        pooled_fwd = np.mean(self.fwd_states, axis=0) if self.pooling == 'mean' else self.fwd_states[-1]
        
        if self.bidirectional:
            self.bwd_states, self.bwd_caches = self._run_cell(self.bwd_cell, x, reverse=True)
            pooled_bwd = np.mean(self.bwd_states, axis=0) if self.pooling == 'mean' else self.bwd_states[0]
            self.pooled = np.vstack([pooled_fwd, pooled_bwd])
        else:
            self.pooled = pooled_fwd
        
        return (self.W_proj @ self.pooled + self.b_proj).flatten()

    def _backprop_cell(self, cell, states, caches, d_pooled, lr):
        """Backprop through one direction."""
        T = len(caches)
        d_h = d_pooled / T if self.pooling == 'mean' else d_pooled
        dh_next = np.zeros((self.hidden_size, 1))
        dc_next = np.zeros((self.hidden_size, 1)) if self.cell_type == 'lstm' else None
        grads = {k: np.zeros_like(getattr(cell, k)) for k in self.param_keys}
        
        for t in reversed(range(T)):
            dh = (d_h + dh_next) if self.pooling == 'mean' else (d_h if t == T-1 else dh_next)
            if self.cell_type == 'lstm':
                dh_next, dc_next, g = cell.backward(dh, dc_next, caches[t])
            else:
                dh_next, g = cell.backward(dh, caches[t])
            for k in g: grads[k] += g[k]
        
        for k, v in grads.items():
            np.clip(v, -5, 5, out=v)
            setattr(cell, k, getattr(cell, k) - lr * v)

    def backward(self, dz, learning_rate=0.01):
        """Backward pass through encoder."""
        dz = dz.reshape(-1, 1)
        dW_proj, db_proj = dz @ self.pooled.T, dz
        d_pooled = self.W_proj.T @ dz
        
        if self.bidirectional:
            d_fwd, d_bwd = d_pooled[:self.hidden_size], d_pooled[self.hidden_size:]
            self._backprop_cell(self.fwd_cell, self.fwd_states, self.fwd_caches, d_fwd, learning_rate)
            self._backprop_cell(self.bwd_cell, self.bwd_states, self.bwd_caches, d_bwd, learning_rate)
        else:
            self._backprop_cell(self.fwd_cell, self.fwd_states, self.fwd_caches, d_pooled, learning_rate)
        
        np.clip(dW_proj, -5, 5, out=dW_proj)
        np.clip(db_proj, -5, 5, out=db_proj)
        self.W_proj -= learning_rate * dW_proj
        self.b_proj -= learning_rate * db_proj


class CoLES:
    """
    CoLES: Contrastive Learning for Event Sequences.
    NT-Xent loss: L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
    """
    def __init__(self, vocab_size, hidden_size, embedding_dim, temperature=0.1, 
                 subsequence_len=10, num_subsequences=2, learning_rate=0.01,
                 cell_type='gru', bidirectional=False):
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.subsequence_len = subsequence_len
        self.num_subsequences = num_subsequences
        self.lr = learning_rate
        self.encoder = CoLESEncoder(vocab_size, hidden_size, embedding_dim, 
                                    cell_type=cell_type, bidirectional=bidirectional)

    def embed_events(self, sequence):
        """Convert event indices to one-hot embeddings."""
        return np.eye(self.vocab_size)[sequence]

    def sample_subsequences(self, sequence):
        """Sample random subsequences from a sequence."""
        seq_len, subseq_len = len(sequence), self.subsequence_len
        if seq_len < subseq_len * self.num_subsequences:
            subseq_len = max(1, seq_len // self.num_subsequences)
        
        starts = list(range(max(1, seq_len - subseq_len + 1)))
        subsequences = []
        for _ in range(self.num_subsequences):
            idx = np.random.randint(0, len(starts)) if starts else 0
            start = starts.pop(idx) if starts else np.random.randint(0, max(1, seq_len - subseq_len + 1))
            subsequences.append(sequence[start:start + subseq_len])
        return subsequences

    def nt_xent_loss(self, embeddings, labels):
        """NT-Xent loss: L_i = -log(exp(sim(z_i,z_j)/τ) / Σ_{k≠i} exp(sim(z_i,z_k)/τ))"""
        embeddings = np.array(embeddings)
        n = len(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        sim = (embeddings @ embeddings.T) / (norms @ norms.T) / self.temperature
        exp_sim = np.exp(sim - sim.max(axis=1, keepdims=True))
        mask = ~np.eye(n, dtype=bool)
        
        loss = 0
        for i in range(n):
            pos_mask = (labels == labels[i]) & (np.arange(n) != i)
            if pos_mask.any():
                pos_idx = np.where(pos_mask)[0][0]
                loss -= np.log(exp_sim[i, pos_idx] / (exp_sim[i][mask[i]].sum() + 1e-10) + 1e-10)
        return loss / n

    def nt_xent_gradients(self, embeddings, labels):
        """Gradients of NT-Xent loss w.r.t. embeddings."""
        embeddings = np.array(embeddings)
        n = len(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        norm_emb = embeddings / norms
        sim = norm_emb @ norm_emb.T / self.temperature
        exp_sim = np.exp(sim - sim.max(axis=1, keepdims=True))
        mask = ~np.eye(n, dtype=bool)
        softmax = (exp_sim * mask) / ((exp_sim * mask).sum(axis=1, keepdims=True) + 1e-10)
        
        grads = np.zeros_like(embeddings)
        for i in range(n):
            pos_mask = (labels == labels[i]) & (np.arange(n) != i)
            if not pos_mask.any(): continue
            pos_idx = np.where(pos_mask)[0][0]
            for j in range(n):
                if i == j: continue
                d_sim = (norm_emb[j] - norm_emb[i] * (norm_emb[i] @ norm_emb[j])) / norms[i, 0]
                coef = (softmax[i, j] - (1 if j == pos_idx else 0)) / self.temperature
                grads[i] += coef * d_sim
        return grads / n

    def train_step(self, sequences):
        """
        Single training step on a batch of sequences.
        
        1. Sample subsequences from each sequence
        2. Encode all subsequences
        3. Compute NT-Xent loss
        4. Backpropagate gradients
        """
        all_subsequences = []
        labels = []
        
        for seq_idx, seq in enumerate(sequences):
            subseqs = self.sample_subsequences(seq)
            for subseq in subseqs:
                all_subsequences.append(subseq)
                labels.append(seq_idx)
        
        labels = np.array(labels)
        
        embeddings = []
        for subseq in all_subsequences:
            x = self.embed_events(subseq)
            z = self.encoder.forward(x)
            embeddings.append(z)
        
        embeddings = np.array(embeddings)
        
        loss = self.nt_xent_loss(embeddings, labels)
        
        gradients = self.nt_xent_gradients(embeddings, labels)
        
        for i, subseq in enumerate(all_subsequences):
            x = self.embed_events(subseq)
            self.encoder.forward(x)
            self.encoder.backward(gradients[i], learning_rate=self.lr)
        
        return loss

    def train(self, sequences, epochs=10, batch_size=4):
        """Train CoLES on a collection of sequences."""
        n_sequences = len(sequences)
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            indices = np.random.permutation(n_sequences)
            
            for i in range(0, n_sequences, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch = [sequences[idx] for idx in batch_indices]
                
                if len(batch) < 2:
                    continue
                
                loss = self.train_step(batch)
                total_loss += loss
                n_batches += 1
            
            avg_loss = total_loss / max(n_batches, 1)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    def get_embedding(self, sequence):
        """Get embedding for a sequence."""
        x = self.embed_events(sequence)
        return self.encoder.forward(x)

    def get_subsequence_embeddings(self, sequence):
        """Get embeddings for all subsequences of a sequence."""
        subseqs = self.sample_subsequences(sequence)
        embeddings = []
        for subseq in subseqs:
            x = self.embed_events(subseq)
            z = self.encoder.forward(x)
            embeddings.append(z)
        return np.array(embeddings)

