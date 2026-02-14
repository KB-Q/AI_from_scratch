"""
T5-style Transformer implementation using PyTorch.
Encoder-decoder model for sequence-to-sequence tasks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from gpt_torch import positional_encoding, MultiHeadAttention, FeedForward, TransformerBlock


class CrossAttention(nn.Module):
    """Cross-attention: queries from decoder, keys/values from encoder."""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        for layer in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, decoder_hidden, encoder_output, encoder_mask=None):
        batch_size, dec_len, _ = decoder_hidden.shape
        enc_len = encoder_output.shape[1]
        
        Q = self.W_q(decoder_hidden).view(batch_size, dec_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(encoder_output).view(batch_size, enc_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(encoder_output).view(batch_size, enc_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if encoder_mask is not None:
            scores = scores + encoder_mask.unsqueeze(1).unsqueeze(2)
        
        attn_out = F.softmax(scores, dim=-1) @ V
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, dec_len, self.d_model)
        return self.W_o(attn_out)


class T5DecoderBlock(nn.Module):
    """Decoder block: causal self-attention + cross-attention + FFN."""
    
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = CrossAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
    
    def forward(self, x, encoder_output, causal_mask=None, encoder_mask=None):
        x1 = self.ln1(x + self.self_attn(x, causal_mask))
        x2 = self.ln2(x1 + self.cross_attn(x1, encoder_output, encoder_mask))
        return self.ln3(x2 + self.ffn(x2))


class T5Model(nn.Module):
    """T5-style encoder-decoder transformer."""
    
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, max_seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        self.register_buffer('pos_embedding', positional_encoding(max_seq_len, d_model))
        
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            T5DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.output_proj.weight, std=0.02)
    
    def create_causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1) * -1e9
    
    def encode(self, input_ids, mask=None):
        x = self.token_embedding(input_ids) + self.pos_embedding[:input_ids.shape[1]]
        for block in self.encoder_blocks:
            x = block(x, mask)
        return x
    
    def decode(self, decoder_ids, encoder_output, encoder_mask=None):
        dec_len = decoder_ids.shape[1]
        x = self.token_embedding(decoder_ids) + self.pos_embedding[:dec_len]
        causal_mask = self.create_causal_mask(dec_len, decoder_ids.device)
        
        for block in self.decoder_blocks:
            x = block(x, encoder_output, causal_mask, encoder_mask)
        return self.output_proj(x)
    
    def forward(self, encoder_ids, decoder_ids, encoder_mask=None):
        encoder_output = self.encode(encoder_ids, encoder_mask)
        return self.decode(decoder_ids, encoder_output, encoder_mask)
    
    def generate(self, encoder_ids, max_new_tokens=50, start_token_id=0, eos_token_id=1, temperature=1.0):
        self.eval()
        device = encoder_ids.device
        encoder_output = self.encode(encoder_ids)
        decoder_ids = torch.full((encoder_ids.shape[0], 1), start_token_id, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.decode(decoder_ids, encoder_output)
                next_logits = logits[:, -1, :] / temperature
                next_token = torch.multinomial(F.softmax(next_logits, dim=-1), 1)
                decoder_ids = torch.cat([decoder_ids, next_token], dim=1)
                if (next_token == eos_token_id).all():
                    break
        return decoder_ids[:, 1:]  # exclude start token
    
    def save(self, filepath):
        torch.save({
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'max_seq_len': self.max_seq_len, 'num_layers': len(self.encoder_blocks),
            'state_dict': self.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, device='cpu'):
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'], d_model=checkpoint['d_model'],
            num_layers=checkpoint['num_layers'], max_seq_len=checkpoint['max_seq_len']
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        print(f"Model loaded from {filepath}")
        return model
