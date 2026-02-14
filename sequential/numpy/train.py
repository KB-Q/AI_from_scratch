import numpy as np
import argparse
import os
import urllib.request

def load_text_corpus(corpus_type='shakespeare'):
    """Download and load text corpus for language modeling."""
    cache_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(cache_dir, exist_ok=True)
    
    if corpus_type == 'shakespeare':
        cache_file = os.path.join(cache_dir, 'shakespeare.txt')
        if not os.path.exists(cache_file):
            print("Downloading Shakespeare dataset...")
            url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            urllib.request.urlretrieve(url, cache_file)
            print("Download complete!")
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            text = f.read()
    
    elif corpus_type == 'linux':
        cache_file = os.path.join(cache_dir, 'linux.txt')
        if not os.path.exists(cache_file):
            print("Downloading Linux kernel source dataset...")
            url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/linux/input.txt'
            urllib.request.urlretrieve(url, cache_file)
            print("Download complete!")
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            text = f.read()
    
    elif corpus_type == 'timeseries':
        cache_file = os.path.join(cache_dir, 'airline.txt')
        if not os.path.exists(cache_file):
            print("Downloading airline passengers time series dataset...")
            url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
            urllib.request.urlretrieve(url, cache_file)
            print("Download complete!")
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            text = f.read()
    
    elif corpus_type == 'simple':
        text = "the quick brown fox jumps over the lazy dog " * 100
    
    else:
        raise ValueError(f"Unknown corpus type: {corpus_type}")

    if len(text) > 10000: 
        print("Dataset size is too large, truncating text to 10000 characters")
        text = text[:10000]
    
    return text

def prepare_char_sequences(text, seq_len=40):
    """Prepare character-level sequences from text."""
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    vocab_size = len(chars)
    
    data = [char_to_idx[ch] for ch in text]
    
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+1:i+seq_len+1])
    
    return np.array(X), np.array(y), vocab_size, char_to_idx, idx_to_char

def prepare_word_corpus(text):
    """Prepare word corpus for Word2Vec training."""
    words = text.split()
    vocab = sorted(set(words))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    
    corpus = [word_to_idx[w] for w in words]
    return corpus, len(vocab), word_to_idx, idx_to_word

def split_data(X, y, train_split=0.8, val_split=0.1):
    """Split data into train/val/test sets."""
    n = len(X)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def compute_perplexity(loss):
    """Compute perplexity from cross-entropy loss."""
    return np.exp(loss)

def evaluate_model(model, X, y, vocab_size):
    """Evaluate model on test set."""
    total_loss = 0
    for i in range(len(X)):
        X_onehot = np.eye(vocab_size)[X[i]]
        y_onehot = np.eye(vocab_size)[y[i]]
        
        pred = model.forward(X_onehot)
        
        if pred.ndim == 2:
            loss = 0.5 * np.mean((pred - y_onehot)**2)
        else:
            loss = 0
            for t in range(len(y_onehot)):
                loss += 0.5 * np.sum((pred[t] - y_onehot[t])**2)
            loss /= len(y_onehot)
        
        total_loss += loss
    
    return total_loss / len(X)

def generate_text(model, seed_text, char_to_idx, idx_to_char, vocab_size, length=100):
    """Generate text using trained model."""
    generated = seed_text
    current_seq = [char_to_idx.get(ch, 0) for ch in seed_text]
    
    for _ in range(length):
        X_onehot = np.eye(vocab_size)[current_seq]
        pred = model.forward(X_onehot)
        
        if pred.ndim == 2:
            next_char_probs = pred[-1]
        else:
            next_char_probs = pred
        
        if isinstance(next_char_probs, np.ndarray) and next_char_probs.ndim > 0:
            next_idx = np.random.choice(vocab_size, p=np.abs(next_char_probs)/np.sum(np.abs(next_char_probs)))
        else:
            next_idx = np.random.randint(0, vocab_size)
        
        generated += idx_to_char[next_idx]
        current_seq = current_seq[1:] + [next_idx]
    
    return generated

def train_sequence_model(args):
    """Train RNN, LSTM, GRU, or Bidirectional variants."""
    from rnn import RNN
    from lstm import LSTM, BidirectionalLSTM
    from gru import GRU, BidirectionalGRU
    
    print(f"\n{'='*60}")
    print(f"Training {args.model.upper()} on {args.dataset} dataset")
    print(f"{'='*60}")
    
    text = load_text_corpus(args.dataset)
    print(f"Dataset size: {len(text)} characters")
    
    X, y, vocab_size, char_to_idx, idx_to_char = prepare_char_sequences(text, seq_len=args.seq_len)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    if args.model == 'rnn':
        model = RNN(vocab_size, args.hidden_size, vocab_size, use_ffn=args.use_ffn)
    elif args.model == 'lstm':
        model = LSTM(vocab_size, args.hidden_size, vocab_size, use_ffn=args.use_ffn)
    elif args.model == 'gru':
        model = GRU(vocab_size, args.hidden_size, vocab_size, use_ffn=args.use_ffn)
    elif args.model == 'bilstm':
        model = BidirectionalLSTM(vocab_size, args.hidden_size, vocab_size)
    elif args.model == 'bigru':
        model = BidirectionalGRU(vocab_size, args.hidden_size, vocab_size)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        for i in range(len(X_train)):
            X_onehot = np.eye(vocab_size)[X_train[i]]
            y_onehot = np.eye(vocab_size)[y_train[i]]
            
            model.forward(X_onehot)
            loss = model.backward(y_onehot, learning_rate=args.lr)
            total_loss += loss
        
        avg_train_loss = total_loss / len(X_train)
        
        if epoch % args.eval_every == 0:
            val_loss = evaluate_model(model, X_val, y_val, vocab_size)
            
            print(f"Epoch {epoch}/{args.epochs}: "
                  f"Train Loss = {avg_train_loss:.6f}, "
                  f"Val Loss = {val_loss:.6f}, "
                  f"Perplexity = {compute_perplexity(avg_train_loss):.2f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
    
    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}")
    test_loss = evaluate_model(model, X_test, y_test, vocab_size)
    print(f"Test Loss = {test_loss:.6f}, Test Perplexity = {compute_perplexity(test_loss):.2f}")
    
    if args.generate:
        print(f"\n{'='*60}")
        print("Generating Sample Text")
        print(f"{'='*60}")
        seed = text[:args.seq_len]
        generated = generate_text(model, seed, char_to_idx, idx_to_char, vocab_size, length=200)
        print(f"Seed: {seed}")
        print(f"Generated: {generated}")

def train_coles(args):
    """Train CoLES model for contrastive learning on event sequences."""
    from coles import CoLES
    
    print(f"\n{'='*60}")
    print(f"Training CoLES on {args.dataset} dataset")
    print(f"{'='*60}")
    
    text = load_text_corpus(args.dataset)
    corpus, vocab_size, word_to_idx, idx_to_word = prepare_word_corpus(text)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Corpus length: {len(corpus)} tokens")
    
    sequences = []
    seq_len = args.coles_seq_len
    for i in range(0, len(corpus) - seq_len, seq_len // 2):
        sequences.append(corpus[i:i+seq_len])
    
    print(f"Number of sequences: {len(sequences)}")
    print(f"Sequence length: {seq_len}")
    print(f"Subsequence length: {args.coles_subseq_len}")
    print(f"Temperature: {args.coles_temperature}")
    
    model = CoLES(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        temperature=args.coles_temperature,
        subsequence_len=args.coles_subseq_len,
        num_subsequences=args.coles_num_subseq,
        learning_rate=args.lr,
        cell_type=args.coles_cell,
        bidirectional=args.coles_bidirectional
    )
    
    model.train(sequences, epochs=args.epochs, batch_size=args.batch_size)
    
    print("\nSample sequence embeddings:")
    for i in range(min(5, len(sequences))):
        embedding = model.get_embedding(sequences[i])
        print(f"Sequence {i}: norm={np.linalg.norm(embedding):.3f}")
    
    print("\nEvaluating embedding similarity:")
    if len(sequences) >= 4:
        emb1 = model.get_embedding(sequences[0])
        emb2 = model.get_embedding(sequences[1])
        emb3 = model.get_embedding(sequences[2])
        
        sim_01 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10)
        sim_02 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3) + 1e-10)
        
        print(f"Similarity(seq0, seq1): {sim_01:.4f}")
        print(f"Similarity(seq0, seq2): {sim_02:.4f}")

def train_word2vec(args):
    """Train Word2Vec model."""
    from word2vec import Word2Vec
    
    print(f"\n{'='*60}")
    print(f"Training Word2Vec ({args.w2v_method}) on {args.dataset} dataset")
    print(f"{'='*60}")
    
    text = load_text_corpus(args.dataset)
    corpus, vocab_size, word_to_idx, idx_to_word = prepare_word_corpus(text)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Corpus length: {len(corpus)} words")
    
    model = Word2Vec(vocab_size, args.embedding_dim, method=args.w2v_method, learning_rate=args.lr)
    model.train(corpus, window_size=args.window_size, epochs=args.epochs)
    
    print("\nSample word embeddings:")
    sample_words = list(word_to_idx.keys())[:10]
    for word in sample_words:
        idx = word_to_idx[word]
        embedding = model.get_embedding(idx)
        print(f"{word}: norm={np.linalg.norm(embedding):.3f}")

def main():
    parser = argparse.ArgumentParser(description='Train sequential models')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['rnn', 'lstm', 'gru', 'bilstm', 'bigru', 'word2vec', 'coles'],
                       help='Model type to train')
    parser.add_argument('--dataset', type=str, default='shakespeare',
                       choices=['shakespeare', 'linux', 'timeseries', 'simple'],
                       help='Dataset to use')
    parser.add_argument('--seq-len', type=int, default=30,
                       help='Sequence length for RNN/LSTM')
    parser.add_argument('--hidden-size', type=int, default=64,
                       help='Hidden layer size')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--eval-every', type=int, default=5,
                       help='Evaluate every N epochs')
    parser.add_argument('--use-ffn', action='store_true',
                       help='Use FFN output layer for LSTM')
    parser.add_argument('--generate', action='store_true',
                       help='Generate text after training')
    parser.add_argument('--w2v-method', type=str, default='skipgram',
                       choices=['skipgram', 'cbow'],
                       help='Word2Vec method')
    parser.add_argument('--embedding-dim', type=int, default=50,
                       help='Embedding dimension for Word2Vec')
    parser.add_argument('--window-size', type=int, default=3,
                       help='Context window size for Word2Vec')
    parser.add_argument('--coles-seq-len', type=int, default=50,
                       help='Sequence length for CoLES')
    parser.add_argument('--coles-subseq-len', type=int, default=10,
                       help='Subsequence length for CoLES')
    parser.add_argument('--coles-num-subseq', type=int, default=2,
                       help='Number of subsequences per sequence for CoLES')
    parser.add_argument('--coles-temperature', type=float, default=0.1,
                       help='Temperature for NT-Xent loss in CoLES')
    parser.add_argument('--coles-cell', type=str, default='gru',
                       choices=['rnn', 'lstm', 'gru'],
                       help='Cell type for CoLES encoder')
    parser.add_argument('--coles-bidirectional', action='store_true',
                       help='Use bidirectional encoder for CoLES')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for CoLES training')
    
    args = parser.parse_args()
    
    if args.model == 'word2vec':
        train_word2vec(args)
    elif args.model == 'coles':
        train_coles(args)
    else:
        train_sequence_model(args)

if __name__ == "__main__":
    main()
