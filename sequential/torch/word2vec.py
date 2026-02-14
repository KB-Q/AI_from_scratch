import torch
import torch.nn as nn
import torch.nn.functional as F

class Word2Vec(nn.Module):
    """
    Word2Vec with Skip-gram and CBOW methods.
    Skip-gram: P(context|word) = softmax(W2·W1[word])
    CBOW: P(word|context) = softmax(W2·mean(W1[context]))
    """
    def __init__(self, vocab_size, embedding_dim, method='skipgram'):
        super().__init__()
        self.vocab_size = vocab_size  # Size of vocabulary
        self.embedding_dim = embedding_dim  # Dimensionality of word embeddings
        self.method = method  # Training method: 'skipgram' or 'cbow'
        self.W1 = nn.Embedding(vocab_size, embedding_dim)  # Input word embeddings
        self.W2 = nn.Linear(embedding_dim, vocab_size, bias=False)  # Context word embeddings
        
        nn.init.normal_(self.W1.weight, mean=0, std=0.001)
        nn.init.normal_(self.W2.weight, mean=0, std=0.001)

    def forward_skipgram(self, center_word, context_words):
        """
        Skip-gram forward pass: predict context from center word.
        
        center_word: Center word indices (batch_size,)
        context_words: Context word indices (batch_size, num_context)
        Returns: loss
        """
        h = self.W1(center_word)
        logits = self.W2(h)
        
        batch_size = center_word.shape[0]
        num_context = context_words.shape[1]
        
        logits_expanded = logits.unsqueeze(1).expand(-1, num_context, -1)
        logits_flat = logits_expanded.reshape(-1, self.vocab_size)
        context_flat = context_words.reshape(-1)
        
        loss = F.cross_entropy(logits_flat, context_flat)
        return loss
    
    def forward_cbow(self, context_words, center_word):
        """
        CBOW forward pass: predict center word from context.
        
        context_words: Context word indices (batch_size, num_context)
        center_word: Center word indices (batch_size,)
        Returns: loss
        """
        context_embeds = self.W1(context_words)
        h = context_embeds.mean(dim=1)
        logits = self.W2(h)
        
        loss = F.cross_entropy(logits, center_word)
        return loss
    
    def forward(self, center_word, context_words):
        """Forward pass using configured method."""
        if self.method == 'skipgram':
            return self.forward_skipgram(center_word, context_words)
        elif self.method == 'cbow':
            return self.forward_cbow(context_words, center_word)
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose 'skipgram' or 'cbow'.")

    def get_embedding(self, idx):
        """Return embedding vector for word index."""
        if isinstance(idx, int):
            idx = torch.tensor([idx], device=self.W1.weight.device)
        return self.W1(idx)
    
    @property
    def embeddings(self):
        """Return all word embeddings."""
        return self.W1.weight.detach()
