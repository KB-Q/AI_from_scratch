"""
Unified training script for GPT, BERT, and T5 models.
Handles different model types, datasets, and training objectives.
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

from gpt_torch import GPTModel
from bert_torch import BERTModel
from t5_torch import T5Model


# ============ DATASETS ============

class LMDataset(Dataset):
    """Dataset for causal language modeling (GPT)."""
    
    def __init__(self, tokenizer, split='train', max_len=128, max_samples=None, cache_dir='data'):
        self.max_len = max_len
        self.pad_id = tokenizer.token_to_id('[PAD]')
        
        cache_file = os.path.join(cache_dir, f'gpt_cache_{split}_{max_samples}.pt')
        if os.path.exists(cache_file):
            print(f"Loading cached {split} data")
            self.examples = torch.load(cache_file)
        else:
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
            self.examples = [tokenizer.encode(x['text']).ids for x in dataset 
                            if len(x['text'].strip()) > 0]
            if max_samples:
                self.examples = self.examples[:max_samples]
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(self.examples, cache_file)
        print(f"LMDataset: {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ids = self.examples[idx][:self.max_len]
        ids = ids + [self.pad_id] * (self.max_len - len(ids))
        return {'input_ids': torch.tensor(ids)}


class MLMDataset(Dataset):
    """Dataset for masked language modeling (BERT)."""
    
    def __init__(self, tokenizer, split='train', max_len=128, mask_prob=0.15, 
                 max_samples=None, cache_dir='data'):
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_id = tokenizer.token_to_id('[MASK]')
        self.pad_id = tokenizer.token_to_id('[PAD]')
        self.vocab_size = tokenizer.get_vocab_size()
        
        cache_file = os.path.join(cache_dir, f'bert_cache_{split}_{max_samples}.pt')
        if os.path.exists(cache_file):
            print(f"Loading cached {split} data")
            self.examples = torch.load(cache_file)
        else:
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
            self.examples = [tokenizer.encode(x['text']).ids for x in dataset 
                            if len(x['text'].strip()) > 0]
            if max_samples:
                self.examples = self.examples[:max_samples]
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(self.examples, cache_file)
        print(f"MLMDataset: {len(self.examples)} examples")
    
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


class Seq2SeqDataset(Dataset):
    """Dataset for seq2seq (T5). Supports summarization and translation."""
    
    def __init__(self, tokenizer, task='summarization', split='train', 
                 src_len=256, tgt_len=64, max_samples=None, cache_dir='data'):
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.pad_id = tokenizer.token_to_id('[PAD]')
        self.bos_id = tokenizer.token_to_id('[BOS]')
        self.eos_id = tokenizer.token_to_id('[EOS]')
        self.task = task
        
        cache_file = os.path.join(cache_dir, f't5_{task}_cache_{split}_{max_samples}.pt')
        if os.path.exists(cache_file):
            print(f"Loading cached {split} data")
            self.examples = torch.load(cache_file)
        else:
            self.examples = []
            if task == 'summarization':
                dataset = load_dataset('cnn_dailymail', '3.0.0', split=split)
                if max_samples:
                    dataset = dataset.select(range(min(max_samples, len(dataset))))
                for x in tqdm(dataset, desc=f"Tokenizing {split}"):
                    if x['article'].strip() and x['highlights'].strip():
                        src = tokenizer.encode(x['article']).ids[:src_len-1] + [self.eos_id]
                        tgt = [self.bos_id] + tokenizer.encode(x['highlights']).ids[:tgt_len-2] + [self.eos_id]
                        self.examples.append((src, tgt))
            elif task == 'translation':
                dataset = load_dataset('wmt14', 'de-en', split=split)
                if max_samples:
                    dataset = dataset.select(range(min(max_samples, len(dataset))))
                for x in tqdm(dataset, desc=f"Tokenizing {split}"):
                    de = x['translation']['de'].strip()
                    en = x['translation']['en'].strip()
                    if de and en:
                        src = tokenizer.encode(de).ids[:src_len-1] + [self.eos_id]
                        tgt = [self.bos_id] + tokenizer.encode(en).ids[:tgt_len-2] + [self.eos_id]
                        self.examples.append((src, tgt))
            
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(self.examples, cache_file)
        print(f"Seq2SeqDataset ({task}): {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        src, tgt = self.examples[idx]
        src = (src + [self.pad_id] * self.src_len)[:self.src_len]
        tgt = (tgt + [self.pad_id] * self.tgt_len)[:self.tgt_len]
        
        dec_input = tgt[:-1] + [self.pad_id]
        labels = [t if t != self.pad_id else -100 for t in tgt[1:]] + [-100]
        
        return {
            'encoder_ids': torch.tensor(src),
            'decoder_ids': torch.tensor(dec_input),
            'labels': torch.tensor(labels)
        }


# ============ TOKENIZER ============

def get_tokenizer(model_type, vocab_size=10000, cache_dir='data'):
    """Get or train tokenizer for specified model type."""
    special_tokens = {
        'gpt': ['[PAD]', '[UNK]'],
        'bert': ['[PAD]', '[UNK]', '[MASK]'],
        't5': ['[PAD]', '[UNK]', '[BOS]', '[EOS]']
    }
    
    save_path = os.path.join(cache_dir, f'{model_type}_tokenizer.json')
    if os.path.exists(save_path):
        return Tokenizer.from_file(save_path)
    
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens[model_type])
    
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    tokenizer.train_from_iterator([x['text'] for x in dataset], trainer=trainer)
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer.save(save_path)
    return tokenizer


# ============ TRAINING ============

def train_gpt(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0, 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.vocab_size)  # pad_id handling
    
    for batch in tqdm(loader, desc="Training"):
        ids = batch['input_ids'].to(device)
        logits = model(ids[:, :-1])
        loss = loss_fn(logits.reshape(-1, model.vocab_size), ids[:, 1:].reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n


def train_bert(model, loader, optimizer, device):
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


def train_t5(model, loader, optimizer, device):
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
def evaluate_gpt(model, loader, device):
    model.eval()
    total_loss, n = 0, 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.vocab_size)
    
    for batch in loader:
        ids = batch['input_ids'].to(device)
        logits = model(ids[:, :-1])
        loss = loss_fn(logits.reshape(-1, model.vocab_size), ids[:, 1:].reshape(-1))
        total_loss += loss.item()
        n += 1
    return total_loss / n


@torch.no_grad()
def evaluate_bert(model, loader, device):
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


@torch.no_grad()
def evaluate_t5(model, loader, device):
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


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['gpt', 'bert', 't5'])
    parser.add_argument('--task', type=str, default='summarization', choices=['summarization', 'translation'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--max_train', type=int, default=10000)
    parser.add_argument('--max_val', type=int, default=1000)
    parser.add_argument('--cache_dir', type=str, default='data')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training {args.model.upper()} on {device}")
    
    tokenizer = get_tokenizer(args.model, args.vocab_size, args.cache_dir)
    vocab_size = tokenizer.get_vocab_size()
    
    # Create model and datasets based on type
    if args.model == 'gpt':
        model = GPTModel(vocab_size, args.d_model, args.num_layers, args.num_heads, 
                         args.d_ff, args.max_len).to(device)
        train_data = LMDataset(tokenizer, 'train', args.max_len, args.max_train, args.cache_dir)
        val_data = LMDataset(tokenizer, 'validation', args.max_len, args.max_val, args.cache_dir)
        train_fn, eval_fn = train_gpt, evaluate_gpt
        
    elif args.model == 'bert':
        model = BERTModel(vocab_size, args.d_model, args.num_layers, args.num_heads,
                          args.d_ff, args.max_len).to(device)
        train_data = MLMDataset(tokenizer, 'train', args.max_len, 0.15, args.max_train, args.cache_dir)
        val_data = MLMDataset(tokenizer, 'validation', args.max_len, 0.15, args.max_val, args.cache_dir)
        train_fn, eval_fn = train_bert, evaluate_bert
        
    elif args.model == 't5':
        model = T5Model(vocab_size, args.d_model, args.num_layers, args.num_heads,
                        args.d_ff, args.max_len).to(device)
        train_data = Seq2SeqDataset(tokenizer, args.task, 'train', args.max_len, args.max_len // 2,
                                    args.max_train, args.cache_dir)
        val_data = Seq2SeqDataset(tokenizer, args.task, 'validation', args.max_len, args.max_len // 2,
                                  args.max_val, args.cache_dir)
        train_fn, eval_fn = train_t5, evaluate_t5
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_loss = float('inf')
    save_path = os.path.join('checkpoints', f'{args.model}_best.pt')
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_fn(model, train_loader, optimizer, device)
        val_loss = eval_fn(model, val_loader, device)
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            model.save(save_path)
    
    print(f"Training complete. Best val loss: {best_loss:.4f}")
    
    # Save tokenizer path for evaluation
    tokenizer.save(os.path.join('checkpoints', f'{args.model}_tokenizer.json'))


if __name__ == '__main__':
    main()
