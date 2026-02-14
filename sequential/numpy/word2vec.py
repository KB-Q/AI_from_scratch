import numpy as np

class Word2Vec:
    """
    Word2Vec with Skip-gram and CBOW methods.
    Skip-gram: P(context|word) = softmax(W2·W1[word])
    CBOW: P(word|context) = softmax(W2·mean(W1[context]))
    """
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01, method='skipgram'):
        self.vocab_size = vocab_size  # Size of vocabulary
        self.embedding_dim = embedding_dim  # Dimensionality of word embeddings
        self.lr = learning_rate  # Learning rate for gradient updates
        self.method = method  # Training method: 'skipgram' or 'cbow'
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.001  # Input word embeddings
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.001  # Context word embeddings

    def softmax(self, x):
        """Numerically stable softmax: exp(x_i) / Σ exp(x_j)"""
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum() + 1e-10)

    def train_skipgram(self, corpus, window_size, epochs):
        """Train using Skip-gram method: predict context from center word."""
        for epoch in range(epochs):
            loss = 0
            num_samples = 0
            for i, target_idx in enumerate(corpus):
                context_idxs = corpus[max(0, i-window_size):i] + corpus[i+1:min(len(corpus), i+window_size+1)]
                if not context_idxs: continue
                
                h = self.W1[target_idx]
                dW1_accum = np.zeros_like(h)
                
                for ctx_idx in context_idxs:
                    u = np.dot(h, self.W2)
                    y_pred = self.softmax(u)
                    
                    e = y_pred.copy()
                    e[ctx_idx] -= 1
                    
                    dW2 = np.outer(h, e)
                    dW1 = np.dot(self.W2, e)
                    
                    np.clip(dW2, -0.1, 0.1, out=dW2)
                    np.clip(dW1, -0.1, 0.1, out=dW1)
                    
                    self.W2 -= self.lr * dW2
                    dW1_accum += dW1
                    
                    loss -= np.log(y_pred[ctx_idx] + 1e-10)
                    num_samples += 1
                
                np.clip(dW1_accum, -0.1, 0.1, out=dW1_accum)
                self.W1[target_idx] -= self.lr * dW1_accum
            
            avg_loss = loss / num_samples if num_samples > 0 else 0
            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch+1}, loss: {loss:.4f}, avg loss: {avg_loss:.4f}")
    
    def train_cbow(self, corpus, window_size, epochs):
        """Train using CBOW method: predict center word from context."""
        for epoch in range(epochs):
            loss = 0
            num_samples = 0
            for i, target_idx in enumerate(corpus):
                context_idxs = corpus[max(0, i-window_size):i] + corpus[i+1:min(len(corpus), i+window_size+1)]
                if not context_idxs: continue
                
                h = np.mean(self.W1[context_idxs], axis=0)
                u = np.dot(h, self.W2)
                y_pred = self.softmax(u)
                
                e = y_pred.copy()
                e[target_idx] -= 1
                
                dW2 = np.outer(h, e)
                dW1 = np.dot(self.W2, e)
                
                np.clip(dW2, -0.1, 0.1, out=dW2)
                np.clip(dW1, -0.1, 0.1, out=dW1)
                
                self.W2 -= self.lr * dW2
                
                grad_per_word = dW1 / len(context_idxs)
                for ctx in context_idxs:
                    self.W1[ctx] -= self.lr * grad_per_word
                    
                loss -= np.log(y_pred[target_idx] + 1e-10)
                num_samples += 1
            
            avg_loss = loss / num_samples if num_samples > 0 else 0
            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch+1}, loss: {loss:.4f}, avg loss: {avg_loss:.4f}")
    
    def train(self, corpus, window_size=2, epochs=5):
        """Train Word2Vec on corpus using either Skip-gram or CBOW."""
        if self.method == 'skipgram':
            self.train_skipgram(corpus, window_size, epochs)
        elif self.method == 'cbow':
            self.train_cbow(corpus, window_size, epochs)
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose 'skipgram' or 'cbow'.")

    def get_embedding(self, idx):
        """Return embedding vector for word index."""
        return self.W1[idx]

