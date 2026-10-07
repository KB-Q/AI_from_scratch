import torch
import torch.nn as nn
import torch.nn.functional as F
from rnn import RNNCell
from lstm import LSTMCell
from gru import GRUCell

CELL_TYPES = {'rnn': RNNCell, 'lstm': LSTMCell, 'gru': GRUCell}

class CoLESEncoder(nn.Module):
    """
    Sequence Encoder for CoLES. Configurable: RNN/LSTM/GRU, uni/bidirectional.
    z = proj(pool(h_1, ..., h_T))  or  z = proj([pool(h_fwd), pool(h_bwd)])
    """
    def __init__(self, input_size, hidden_size, embedding_dim, cell_type='gru', 
                 bidirectional=False, pooling='mean'):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.pooling = pooling
        
        self.fwd_cell = CELL_TYPES[cell_type](input_size, hidden_size)
        self.bwd_cell = CELL_TYPES[cell_type](input_size, hidden_size) if bidirectional else None
        
        proj_input = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Linear(proj_input, embedding_dim)

    def _run_cell(self, cell, x, reverse=False):
        """Run RNN cell over sequence, return hidden states."""
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device) if self.cell_type == 'lstm' else None
        
        states = []
        indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        
        for t in indices:
            x_t = x[:, t, :]
            if self.cell_type == 'lstm':
                h, c = cell(x_t, h, c)
            else:
                h = cell(x_t, h)
            states.append(h)
        
        if reverse:
            states = states[::-1]
        
        return torch.stack(states, dim=1)

    def forward(self, x):
        """
        Forward pass: encode sequence into embedding.
        
        x: Input sequence (batch_size, seq_len, input_size)
        Returns: Embedding (batch_size, embedding_dim)
        """
        fwd_states = self._run_cell(self.fwd_cell, x)
        
        if self.pooling == 'mean':
            pooled_fwd = fwd_states.mean(dim=1)
        else:
            pooled_fwd = fwd_states[:, -1, :]
        
        if self.bidirectional:
            bwd_states = self._run_cell(self.bwd_cell, x, reverse=True)
            if self.pooling == 'mean':
                pooled_bwd = bwd_states.mean(dim=1)
            else:
                pooled_bwd = bwd_states[:, 0, :]
            pooled = torch.cat([pooled_fwd, pooled_bwd], dim=-1)
        else:
            pooled = pooled_fwd
        
        return self.proj(pooled)


class CoLES(nn.Module):
    """
    CoLES: Contrastive Learning for Event Sequences.
    NT-Xent loss: L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
    """
    def __init__(self, vocab_size, hidden_size, embedding_dim, temperature=0.1, 
                 subsequence_len=10, num_subsequences=2,
                 cell_type='gru', bidirectional=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.subsequence_len = subsequence_len
        self.num_subsequences = num_subsequences
        self.encoder = CoLESEncoder(vocab_size, hidden_size, embedding_dim, 
                                    cell_type=cell_type, bidirectional=bidirectional)

    def embed_events(self, sequence):
        """Convert event indices to one-hot embeddings."""
        return F.one_hot(sequence, num_classes=self.vocab_size).float()

    def sample_subsequences(self, sequence):
        """
        Sample random subsequences from a sequence.
        
        sequence: (batch_size, seq_len)
        Returns: List of subsequences, each (batch_size, subseq_len)
        """
        batch_size, seq_len = sequence.shape
        subseq_len = min(self.subsequence_len, seq_len)
        
        subsequences = []
        for _ in range(self.num_subsequences):
            max_start = max(1, seq_len - subseq_len + 1)
            starts = torch.randint(0, max_start, (batch_size,), device=sequence.device)
            
            subseq = torch.stack([
                sequence[b, starts[b]:starts[b] + subseq_len]
                for b in range(batch_size)
            ])
            subsequences.append(subseq)
        
        return subsequences

    def nt_xent_loss(self, embeddings, labels):
        """
        NT-Xent loss: L_i = -log(exp(sim(z_i,z_j)/τ) / Σ_{k≠i} exp(sim(z_i,z_k)/τ))
        
        embeddings: (N, embedding_dim)
        labels: (N,) - indices indicating which sequence each embedding came from
        Returns: loss value
        """
        embeddings = F.normalize(embeddings, dim=1)
        n = embeddings.shape[0]
        
        sim = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        mask = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)
        
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & mask
        
        exp_sim = torch.exp(sim)
        
        numerator = (exp_sim * pos_mask.float()).sum(dim=1)
        denominator = (exp_sim * mask.float()).sum(dim=1)
        
        has_positive = pos_mask.any(dim=1)
        loss = -torch.log(numerator / (denominator + 1e-10) + 1e-10)
        loss = loss[has_positive].mean()
        
        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=embeddings.device)

    def forward(self, sequences, labels):
        """
        Forward pass for training.
        
        sequences: List of sequences (each batch_size, seq_len)
        labels: Sequence labels (batch_size,)
        Returns: Loss value
        """
        all_embeddings = []
        all_labels = []
        
        for seq_idx, seq in enumerate(sequences):
            subseqs = self.sample_subsequences(seq.unsqueeze(0) if seq.dim() == 1 else seq)
            for subseq in subseqs:
                x = self.embed_events(subseq)
                z = self.encoder(x)
                all_embeddings.append(z)
                if seq.dim() == 1:
                    all_labels.append(labels[seq_idx:seq_idx+1])
                else:
                    all_labels.append(labels)
        
        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        return self.nt_xent_loss(embeddings, labels)

    def get_embedding(self, sequence):
        """
        Get embedding for a sequence.
        
        sequence: (seq_len,) or (batch_size, seq_len)
        Returns: Embedding (embedding_dim,) or (batch_size, embedding_dim)
        """
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0)
        x = self.embed_events(sequence)
        return self.encoder(x).squeeze(0)
