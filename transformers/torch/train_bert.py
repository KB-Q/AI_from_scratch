"""
Training script for BERT model on WikiText-2 dataset.
Uses Masked Language Modeling (MLM) objective.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import os
import argparse
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from bert_torch import BERTModel


class MLMDataset(Dataset):
    """Dataset for Masked Language Modeling. Required by PyTorch DataLoader."""
    
    def __init__(self, tokenizer, split='train', max_len=128, mask_prob=0.15, 
                 max_samples=None, cache_dir='checkpoints'):
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_id = tokenizer.token_to_id('[MASK]')
        self.pad_id = tokenizer.token_to_id('[PAD]')
        self.vocab_size = tokenizer.get_vocab_size()
        
        # Try to load cached tokenized data
        cache_file = os.path.join(cache_dir, f'bert_cache_{split}_{max_samples}.pt')
        if os.path.exists(cache_file):
            print(f"Loading cached {split} data from {cache_file}")
            self.examples = torch.load(cache_file)
            print(f"Loaded {len(self.examples)} examples")
            return
        
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        self.examples = [tokenizer.encode(x['text']).ids for x in dataset 
                        if len(x['text'].strip()) > 0]
        if max_samples:
            self.examples = self.examples[:max_samples]
        
        # Cache tokenized data
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(self.examples, cache_file)
        print(f"Cached {len(self.examples)} examples to {cache_file}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ids = self.examples[idx][:self.max_len]
        seq_len = len(ids)
        ids = ids + [self.pad_id] * (self.max_len - seq_len)
        
        input_ids, labels = ids.copy(), [-100] * self.max_len
        for i in range(seq_len):
            if np.random.random() < self.mask_prob:
                labels[i] = ids[i]
                r = np.random.random()
                if r < 0.8:
                    input_ids[i] = self.mask_id
                elif r < 0.9:
                    input_ids[i] = np.random.randint(0, self.vocab_size)
        
        mask = [1] * seq_len + [0] * (self.max_len - seq_len)
        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(labels),
            'attention_mask': torch.tensor(mask)
        }


def get_tokenizer(vocab_size=10000, save_path='checkpoints/bert_tokenizer.json'):
    """Train or load BPE tokenizer."""
    if os.path.exists(save_path):
        return Tokenizer.from_file(save_path)
    
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, 
                                   special_tokens=['[PAD]', '[UNK]', '[MASK]'])
    
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    tokenizer.train_from_iterator([x['text'] for x in dataset], trainer=trainer)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    return tokenizer


def train(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0, 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    for batch in tqdm(loader, desc="Training"):
        ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        mask = batch['attention_mask'].to(device)
        
        logits = model(ids, mask)
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
        ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        mask = batch['attention_mask'].to(device)
        
        logits = model(ids, mask)
        loss = loss_fn(logits.view(-1, model.vocab_size), labels.view(-1))
        total_loss += loss.item()
        n += 1
    return total_loss / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=None)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tokenizer = get_tokenizer(args.vocab_size)
    vocab_size = tokenizer.get_vocab_size()
    
    train_data = MLMDataset(tokenizer, 'train', args.max_seq_len, max_samples=args.max_train)
    val_data = MLMDataset(tokenizer, 'validation', args.max_seq_len, max_samples=args.max_val)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    
    model = BERTModel(vocab_size, args.d_model, args.num_layers, args.num_heads, 
                      args.d_ff, args.max_seq_len).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            model.save('checkpoints/bert_best.pt')
    
    print(f"Training complete. Best val loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
