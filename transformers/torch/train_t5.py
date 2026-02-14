"""
Training script for T5 model on CNN/DailyMail dataset.
Uses sequence-to-sequence objective for text summarization.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import argparse
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from t5_torch import T5Model


class Seq2SeqDataset(Dataset):
    """Dataset for seq2seq. Required by PyTorch DataLoader."""
    
    def __init__(self, tokenizer, split='train', src_len=512, tgt_len=128, max_samples=None, 
                 cache_dir='data'):
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.pad_id = tokenizer.token_to_id('[PAD]')
        self.bos_id = tokenizer.token_to_id('[BOS]')
        self.eos_id = tokenizer.token_to_id('[EOS]')
        
        # Try to load cached tokenized data
        cache_file = os.path.join(cache_dir, f't5_cache_{split}_{max_samples}.pt')
        if os.path.exists(cache_file):
            print(f"Loading cached {split} data from {cache_file}")
            self.examples = torch.load(cache_file)
            print(f"Loaded {len(self.examples)} examples")
            return
        
        dataset = load_dataset('cnn_dailymail', '3.0.0', split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.examples = []
        for x in tqdm(dataset, desc=f"Tokenizing {split}"):
            if x['article'].strip() and x['highlights'].strip():
                src = tokenizer.encode(x['article']).ids[:src_len-1] + [self.eos_id]
                tgt = [self.bos_id] + tokenizer.encode(x['highlights']).ids[:tgt_len-2] + [self.eos_id]
                self.examples.append((src, tgt))
        
        # Cache tokenized data
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(self.examples, cache_file)
        print(f"Cached {len(self.examples)} examples to {cache_file}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        src, tgt = self.examples[idx]
        
        # Pad sequences
        src = src + [self.pad_id] * (self.src_len - len(src))
        tgt = tgt + [self.pad_id] * (self.tgt_len - len(tgt))
        
        dec_input = tgt[:-1] + [self.pad_id]
        labels = [t if t != self.pad_id else -100 for t in tgt[1:]] + [-100]
        
        return {
            'encoder_ids': torch.tensor(src),
            'decoder_ids': torch.tensor(dec_input),
            'labels': torch.tensor(labels)
        }


def get_tokenizer(vocab_size=16000, save_path='data/t5_tokenizer.json'):
    """Train or load BPE tokenizer."""
    if os.path.exists(save_path):
        return Tokenizer.from_file(save_path)
    
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                   special_tokens=['[PAD]', '[UNK]', '[BOS]', '[EOS]'])
    
    dataset = load_dataset('cnn_dailymail', '3.0.0', split='train')
    texts = [x['article'] for x in dataset.select(range(50000))]
    tokenizer.train_from_iterator(texts, trainer=trainer)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    return tokenizer


def train(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0, 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    for batch in tqdm(loader, desc="Training"):
        enc_ids = batch['encoder_ids'].to(device)
        dec_ids = batch['decoder_ids'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(enc_ids, dec_ids)
        loss = loss_fn(logits.view(-1, model.vocab_size), labels.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n += 1
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, n = 0, 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    for batch in loader:
        enc_ids = batch['encoder_ids'].to(device)
        dec_ids = batch['decoder_ids'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(enc_ids, dec_ids)
        loss = loss_fn(logits.view(-1, model.vocab_size), labels.view(-1))
        total_loss += loss.item()
        n += 1
    return total_loss / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--src_len', type=int, default=512)
    parser.add_argument('--tgt_len', type=int, default=128)
    parser.add_argument('--vocab_size', type=int, default=16000)
    parser.add_argument('--max_train', type=int, default=10000)
    parser.add_argument('--max_val', type=int, default=1000)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tokenizer = get_tokenizer(args.vocab_size)
    vocab_size = tokenizer.get_vocab_size()
    
    train_data = Seq2SeqDataset(tokenizer, 'train', args.src_len, args.tgt_len, args.max_train)
    val_data = Seq2SeqDataset(tokenizer, 'validation', args.src_len, args.tgt_len, args.max_val)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    
    model = T5Model(vocab_size, args.d_model, args.num_layers, args.num_heads,
                    args.d_ff, max(args.src_len, args.tgt_len)).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            model.save('checkpoints/t5_best.pt')
    
    print(f"Training complete. Best val loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
