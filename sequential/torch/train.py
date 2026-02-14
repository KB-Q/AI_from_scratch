import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import urllib.request
import zipfile
from utils import get_device

class TextDataset(Dataset):
    """Character-level text dataset for language modeling."""
    def __init__(self, data, seq_len, vocab_size):
        self.data = data
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        x_onehot = F.one_hot(torch.tensor(x), num_classes=self.vocab_size).float()
        return x_onehot, torch.tensor(y)


class Word2VecDataset(Dataset):
    """Dataset for Word2Vec training with skip-gram/CBOW."""
    def __init__(self, corpus, window_size=2):
        self.corpus = corpus
        self.window_size = window_size
        self.pairs = self._create_pairs()
    
    def _create_pairs(self):
        pairs = []
        for i, target_idx in enumerate(self.corpus):
            start = max(0, i - self.window_size)
            end = min(len(self.corpus), i + self.window_size + 1)
            context_indices = [self.corpus[j] for j in range(start, end) if j != i]
            if context_indices:
                pairs.append((target_idx, context_indices))
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        target, context = self.pairs[idx]
        max_ctx = self.window_size * 2
        context = context[:max_ctx]
        while len(context) < max_ctx:
            context.append(context[-1] if context else 0)
        return torch.tensor(target), torch.tensor(context)


class CoLESDataset(Dataset):
    """Dataset for CoLES contrastive learning."""
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(idx)


def load_text_corpus(corpus_type='wikitext2', max_chars=None):
    """Download and load text corpus for language modeling."""
    cache_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(cache_dir, exist_ok=True)
    
    if corpus_type == 'wikitext2':
        cache_file = os.path.join(cache_dir, 'wikitext2.txt')
        if not os.path.exists(cache_file):
            print("Downloading WikiText-2 dataset...")
            url = 'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt'
            try:
                urllib.request.urlretrieve(url, cache_file)
                print("Download complete!")
            except Exception as e:
                print(f"Download failed: {e}, using fallback text")
                with open(cache_file, 'w') as f:
                    f.write("the quick brown fox jumps over the lazy dog " * 1000)
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            text = f.read()
    
    elif corpus_type == 'shakespeare':
        cache_file = os.path.join(cache_dir, 'shakespeare.txt')
        if not os.path.exists(cache_file):
            print("Downloading Shakespeare dataset...")
            url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            urllib.request.urlretrieve(url, cache_file)
            print("Download complete!")
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            text = f.read()
    
    elif corpus_type == 'ptb':
        cache_file = os.path.join(cache_dir, 'ptb.txt')
        if not os.path.exists(cache_file):
            print("Downloading Penn Treebank dataset...")
            url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt'
            try:
                urllib.request.urlretrieve(url, cache_file)
                print("Download complete!")
            except:
                url = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt'
                urllib.request.urlretrieve(url, cache_file)
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            text = f.read()
    
    elif corpus_type == 'simple':
        text = "the quick brown fox jumps over the lazy dog " * 500
    
    else:
        raise ValueError(f"Unknown corpus type: {corpus_type}")
    
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    
    return text


def prepare_char_sequences(text):
    """Prepare character-level vocabulary and data."""
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    vocab_size = len(chars)
    
    data = [char_to_idx[ch] for ch in text]
    
    return data, vocab_size, char_to_idx, idx_to_char


def prepare_word_corpus(text):
    """Prepare word corpus for Word2Vec training."""
    words = text.lower().split()
    vocab = sorted(set(words))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    
    corpus = [word_to_idx[w] for w in words]
    return corpus, len(vocab), word_to_idx, idx_to_word


def compute_perplexity(loss):
    """Compute perplexity from cross-entropy loss."""
    return torch.exp(torch.tensor(loss)).item()


def train_sequence_model(args):
    """Train RNN, LSTM, GRU, or Bidirectional variants."""
    from rnn import RNN, NativeRNN
    from lstm import LSTM, BidirectionalLSTM, NativeLSTM, NativeBidirectionalLSTM
    from gru import GRU, BidirectionalGRU, NativeGRU, NativeBidirectionalGRU
    
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Training {args.model.upper()} on {args.dataset} dataset")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    text = load_text_corpus(args.dataset, max_chars=args.max_chars)
    print(f"Dataset size: {len(text)} characters")
    
    data, vocab_size, char_to_idx, idx_to_char = prepare_char_sequences(text)
    
    train_split = int(len(data) * 0.8)
    val_split = int(len(data) * 0.9)
    
    train_data = data[:train_split]
    val_data = data[train_split:val_split]
    test_data = data[val_split:]
    
    train_dataset = TextDataset(train_data, args.seq_len, vocab_size)
    val_dataset = TextDataset(val_data, args.seq_len, vocab_size)
    test_dataset = TextDataset(test_data, args.seq_len, vocab_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
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
    elif args.model == 'native_rnn':
        model = NativeRNN(vocab_size, args.hidden_size, vocab_size, use_ffn=args.use_ffn, num_layers=args.num_layers)
    elif args.model == 'native_lstm':
        model = NativeLSTM(vocab_size, args.hidden_size, vocab_size, use_ffn=args.use_ffn, num_layers=args.num_layers)
    elif args.model == 'native_gru':
        model = NativeGRU(vocab_size, args.hidden_size, vocab_size, use_ffn=args.use_ffn, num_layers=args.num_layers)
    elif args.model == 'native_bilstm':
        model = NativeBidirectionalLSTM(vocab_size, args.hidden_size, vocab_size, num_layers=args.num_layers)
    elif args.model == 'native_bigru':
        model = NativeBidirectionalGRU(vocab_size, args.hidden_size, vocab_size, num_layers=args.num_layers)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            if args.model in ['bilstm', 'bigru']:
                outputs = model(x)
            else:
                outputs, _ = model(x)
            
            outputs = outputs.view(-1, vocab_size)
            y = y.view(-1)
            
            loss = criterion(outputs, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        if epoch % args.eval_every == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    if args.model in ['bilstm', 'bigru']:
                        outputs = model(x)
                    else:
                        outputs, _ = model(x)
                    outputs = outputs.view(-1, vocab_size)
                    y = y.view(-1)
                    val_loss += criterion(outputs, y).item()
            
            val_loss /= len(val_loader)
            
            print(f"Epoch {epoch}/{args.epochs}: "
                  f"Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, "
                  f"Perplexity = {compute_perplexity(avg_train_loss):.2f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
    
    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}")
    
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            if args.model in ['bilstm', 'bigru']:
                outputs = model(x)
            else:
                outputs, _ = model(x)
            outputs = outputs.view(-1, vocab_size)
            y = y.view(-1)
            test_loss += criterion(outputs, y).item()
    
    test_loss /= len(test_loader)
    print(f"Test Loss = {test_loss:.4f}, Test Perplexity = {compute_perplexity(test_loss):.2f}")
    
    if args.generate:
        print(f"\n{'='*60}")
        print("Generating Sample Text")
        print(f"{'='*60}")
        generated = generate_text(model, text[:args.seq_len], char_to_idx, idx_to_char, 
                                  vocab_size, device, length=200, model_type=args.model)
        print(f"Generated: {generated}")


def generate_text(model, seed_text, char_to_idx, idx_to_char, vocab_size, device, length=100, model_type='lstm'):
    """Generate text using trained model."""
    model.eval()
    generated = seed_text
    current_seq = [char_to_idx.get(ch, 0) for ch in seed_text]
    
    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([current_seq], device=device)
            x_onehot = F.one_hot(x, num_classes=vocab_size).float()
            
            if model_type in ['bilstm', 'bigru']:
                outputs = model(x_onehot)
            else:
                outputs, _ = model(x_onehot)
            
            next_char_logits = outputs[0, -1, :]
            probs = F.softmax(next_char_logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            
            generated += idx_to_char[next_idx]
            current_seq = current_seq[1:] + [next_idx]
    
    return generated


def train_word2vec(args):
    """Train Word2Vec model."""
    from word2vec import Word2Vec
    
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Training Word2Vec ({args.w2v_method}) on {args.dataset} dataset")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    text = load_text_corpus(args.dataset, max_chars=args.max_chars)
    corpus, vocab_size, word_to_idx, idx_to_word = prepare_word_corpus(text)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Corpus length: {len(corpus)} words")
    
    dataset = Word2VecDataset(corpus, window_size=args.window_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    model = Word2Vec(vocab_size, args.embedding_dim, method=args.w2v_method).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        
        for center, context in loader:
            center, context = center.to(device), context.to(device)
            
            optimizer.zero_grad()
            loss = model(center, context)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs}, Loss: {avg_loss:.4f}")
    
    print("\nSample word embeddings:")
    sample_words = list(word_to_idx.keys())[:10]
    for word in sample_words:
        idx = word_to_idx[word]
        embedding = model.get_embedding(idx)
        print(f"{word}: norm={embedding.norm().item():.3f}")


def train_coles(args):
    """Train CoLES model for contrastive learning on event sequences."""
    from coles import CoLES
    
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Training CoLES on {args.dataset} dataset")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    text = load_text_corpus(args.dataset, max_chars=args.max_chars)
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
    
    dataset = CoLESDataset(sequences)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    model = CoLES(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        temperature=args.coles_temperature,
        subsequence_len=args.coles_subseq_len,
        num_subsequences=args.coles_num_subseq,
        cell_type=args.coles_cell,
        bidirectional=args.coles_bidirectional
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        
        for seqs, labels in loader:
            seqs, labels = seqs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            all_embeddings = []
            all_labels = []
            
            for b in range(seqs.shape[0]):
                seq = seqs[b]
                label = labels[b]
                
                subseqs = model.sample_subsequences(seq.unsqueeze(0))
                for subseq in subseqs:
                    x = model.embed_events(subseq)
                    z = model.encoder(x)
                    all_embeddings.append(z)
                    all_labels.append(label.unsqueeze(0))
            
            if all_embeddings:
                embeddings = torch.cat(all_embeddings, dim=0)
                batch_labels = torch.cat(all_labels, dim=0)
                
                loss = model.nt_xent_loss(embeddings, batch_labels)
                
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs}, Loss: {avg_loss:.6f}")
    
    print("\nSample sequence embeddings:")
    model.eval()
    with torch.no_grad():
        for i in range(min(5, len(sequences))):
            seq = torch.tensor(sequences[i], device=device)
            embedding = model.get_embedding(seq)
            print(f"Sequence {i}: norm={embedding.norm().item():.3f}")


def main():
    parser = argparse.ArgumentParser(description='Train sequential models (PyTorch)')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['rnn', 'lstm', 'gru', 'bilstm', 'bigru', 
                                'native_rnn', 'native_lstm', 'native_gru', 'native_bilstm', 'native_bigru',
                                'word2vec', 'coles'],
                       help='Model type to train (native_* versions use PyTorch optimized kernels)')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                       choices=['wikitext2', 'shakespeare', 'ptb', 'simple'],
                       help='Dataset to use')
    parser.add_argument('--max-chars', type=int, default=500000,
                       help='Maximum characters to use from dataset')
    parser.add_argument('--seq-len', type=int, default=50,
                       help='Sequence length for RNN/LSTM')
    parser.add_argument('--hidden-size', type=int, default=256,
                       help='Hidden layer size')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of layers for native models')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--eval-every', type=int, default=1,
                       help='Evaluate every N epochs')
    parser.add_argument('--use-ffn', action='store_true',
                       help='Use FFN output layer')
    parser.add_argument('--generate', action='store_true',
                       help='Generate text after training')
    parser.add_argument('--w2v-method', type=str, default='skipgram',
                       choices=['skipgram', 'cbow'],
                       help='Word2Vec method')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--window-size', type=int, default=5,
                       help='Context window size for Word2Vec')
    parser.add_argument('--coles-seq-len', type=int, default=100,
                       help='Sequence length for CoLES')
    parser.add_argument('--coles-subseq-len', type=int, default=20,
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
    
    args = parser.parse_args()
    
    if args.model == 'word2vec':
        train_word2vec(args)
    elif args.model == 'coles':
        train_coles(args)
    else:
        train_sequence_model(args)


if __name__ == "__main__":
    main()
