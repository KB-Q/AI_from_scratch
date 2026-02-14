"""
Utility functions for data processing.
Includes helper functions for downloading and processing datasets.
"""
import numpy as np
import requests
import os
from collections import Counter


# ============================================================================
# Helper Functions
# ============================================================================

def softmax(x, axis=-1) -> np.ndarray:
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5) -> np.ndarray:
    """Layer normalization with learnable parameters."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def gelu(x) -> np.ndarray:
    """GELU activation (approximation)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def positional_encoding(max_len, d_model) -> np.ndarray:
    """Sinusoidal positional encodings."""
    pe = np.zeros((max_len, d_model))
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def cross_entropy_loss(logits, targets) -> tuple[float, np.ndarray]:
    """
    Cross-entropy loss for language modeling.
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len) with token IDs
    """
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    
    # Softmax probabilities
    probs = softmax(logits_flat, axis=-1)
    
    # Cross-entropy: -log(p[correct_class])
    log_probs = np.log(probs[np.arange(len(targets_flat)), targets_flat] + 1e-10)
    loss = -np.mean(log_probs)
    
    # Gradient of loss w.r.t. logits
    dlogits = probs.copy()
    dlogits[np.arange(len(targets_flat)), targets_flat] -= 1
    dlogits = dlogits / len(targets_flat)
    dlogits = dlogits.reshape(batch_size, seq_len, vocab_size)
    
    return loss, dlogits


# ============================================================================
# Data Processing Functions
# ============================================================================

def download_tiny_shakespeare(data_dir='data') -> str:
    """Download Tiny Shakespeare dataset."""
    os.makedirs(data_dir, exist_ok=True)
    txt_path = os.path.join(data_dir, 'input.txt')
    
    if not os.path.exists(txt_path):
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f"Downloading {url}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Saved to {txt_path}")
    
    return txt_path


def download_tinychat(data_dir='data', sample_frac=0.1) -> str:
    """
    Download TinyChat dataset from HuggingFace using datasets API.
    Dataset: https://huggingface.co/datasets/starhopp3r/TinyChat
    
    Args:
        data_dir: Directory to save data
        sample_frac: Fraction of data to use (0.1 = 10%, 1.0 = 100%)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Always create filename based on sample fraction for consistency
    txt_path = os.path.join(data_dir, f'tinychat_{int(sample_frac*100)}pct.txt')
    
    if os.path.exists(txt_path):
        print(f"Using cached data from {txt_path}")
        return txt_path
    
    print(f"Downloading TinyChat dataset from HuggingFace (using {int(sample_frac*100)}% of data)...")
    
    try:
        from datasets import load_dataset
        
        # Load dataset using HuggingFace datasets library
        dataset = load_dataset('starhopp3r/TinyChat', split='train')
        
        # Sample dataset if requested
        if sample_frac < 1.0:
            num_samples = int(len(dataset) * sample_frac)
            dataset = dataset.select(range(num_samples))
            print(f"Using {num_samples:,} samples ({int(sample_frac*100)}% of dataset)")
        
        # Extract conversations
        conversations = []
        
        # Check what columns are available
        if 'messages' in dataset.column_names:
            for item in dataset:
                messages = item['messages']
                if messages is None:
                    continue
                # Handle list of messages
                if isinstance(messages, list):
                    for msg in messages:
                        if isinstance(msg, dict) and 'content' in msg:
                            conversations.append(msg['content'])
                        elif isinstance(msg, str):
                            conversations.append(msg)
                elif isinstance(messages, str):
                    conversations.append(messages)
        elif 'conversation' in dataset.column_names:
            conversations = [str(item['conversation']) for item in dataset if item['conversation']]
        elif 'text' in dataset.column_names:
            conversations = [str(item['text']) for item in dataset if item['text']]
        else:
            # Try to extract from all text columns
            for item in dataset:
                for key, value in item.items():
                    if isinstance(value, str) and value.strip():
                        conversations.append(value)
        
        # Join conversations with newlines
        text = '\n'.join(str(conv) for conv in conversations if conv)
        
        # Save as text file
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Saved {len(conversations):,} conversations to {txt_path}")
        
    except ImportError:
        print("Error: HuggingFace datasets library is required for TinyChat.")
        print("Install with: pip install datasets")
        raise
    except Exception as e:
        print(f"Error downloading TinyChat: {e}")
        print("Make sure you have 'datasets' installed: pip install datasets")
        raise
    
    return txt_path


def load_and_clean_text(txt_path) -> str:
    """Load and clean text: lowercase, keep only alphabetic + space."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    
    # Clean: keep only letters and spaces
    words = []
    for line in text.splitlines():
        cleaned = ''.join(c if c.isalpha() or c == ' ' else '' for c in line)
        words.extend(cleaned.split())
    
    return ' '.join(words)


def build_vocab(text, max_vocab=5000) -> tuple[dict[str, int], dict[int, str]]:
    """Build word-level vocabulary."""
    words = text.split()
    counter = Counter(words)
    
    # Most common words
    vocab_words = [word for word, _ in counter.most_common(max_vocab)]
    
    # Create mappings
    word_to_id = {word: idx for idx, word in enumerate(vocab_words)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    
    return word_to_id, id_to_word


def tokenize(text, word_to_id) -> np.ndarray:
    """Convert text to token IDs."""
    words = text.split()
    # Use first token (most common) as unknown token
    return np.array([word_to_id.get(word, 0) for word in words])


def create_batches(token_ids, seq_len=32, batch_size=16) -> tuple[np.ndarray, np.ndarray]:
    """
    Create batches for next-token prediction.
    Returns inputs and targets where targets are inputs shifted by 1.
    """
    n = len(token_ids)
    sequences = []
    
    # Create overlapping sequences
    for i in range(0, n - seq_len, seq_len // 2):
        seq = token_ids[i:i + seq_len + 1]
        if len(seq) == seq_len + 1:
            sequences.append(seq)
    
    # Trim to multiple of batch_size
    num_batches = len(sequences) // batch_size
    sequences = sequences[:num_batches * batch_size]
    
    # Convert to numpy array and reshape
    sequences = np.array(sequences)
    
    # Split into inputs and targets
    inputs = sequences[:, :-1]  # All but last token
    targets = sequences[:, 1:]  # All but first token
    
    # Reshape into batches
    inputs = inputs.reshape(num_batches, batch_size, seq_len)
    targets = targets.reshape(num_batches, batch_size, seq_len)
    
    return inputs, targets