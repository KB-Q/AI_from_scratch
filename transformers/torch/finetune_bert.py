"""
BERT finetuning pipeline for sentiment classification.
Uses SST-2 (Stanford Sentiment Treebank) dataset.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import argparse
from datasets import load_dataset
from tokenizers import Tokenizer

from bert_torch import BERTModel


class BERTClassifier(nn.Module):
    """BERT with classification head for sentiment analysis."""
    
    def __init__(self, bert_model, num_classes=2, freeze_bert=False):
        super().__init__()
        self.bert = bert_model
        self.d_model = bert_model.d_model
        self.num_classes = num_classes
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, num_classes)
        )
        
        # Initialize
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        
        # Optionally freeze BERT weights
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (batch, seq_len)
        Returns: logits (batch, num_classes)
        """
        # Get BERT hidden states (batch, seq_len, d_model)
        hidden = self.bert.token_embedding(input_ids) + self.bert.pos_embedding[:input_ids.shape[1]]
        mask = self.bert.create_padding_mask(attention_mask)
        
        for block in self.bert.blocks:
            hidden = block(hidden, mask)
        
        # Use [CLS] token representation (first token)
        cls_hidden = hidden[:, 0, :]
        
        # Classification
        logits = self.classifier(cls_hidden)
        return logits
    
    def save(self, filepath):
        torch.save({
            'bert_state': self.bert.state_dict(),
            'classifier_state': self.classifier.state_dict(),
            'd_model': self.d_model,
            'num_classes': self.num_classes,
        }, filepath)
        print(f"Classifier saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, bert_model, device='cpu'):
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(bert_model, num_classes=checkpoint['num_classes'])
        model.bert.load_state_dict(checkpoint['bert_state'])
        model.classifier.load_state_dict(checkpoint['classifier_state'])
        model.to(device)
        print(f"Classifier loaded from {filepath}")
        return model


class SentimentDataset(Dataset):
    """SST-2 sentiment dataset."""
    
    def __init__(self, tokenizer, split='train', max_len=128, max_samples=None):
        self.max_len = max_len
        self.pad_id = tokenizer.token_to_id('[PAD]')
        
        # Load SST-2 from GLUE benchmark
        dataset = load_dataset('glue', 'sst2', split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.examples = []
        for x in dataset:
            ids = tokenizer.encode(x['sentence']).ids[:max_len]
            label = x['label']
            self.examples.append((ids, label))
        
        print(f"SentimentDataset ({split}): {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ids, label = self.examples[idx]
        seq_len = len(ids)
        ids = ids + [self.pad_id] * (self.max_len - seq_len)
        mask = [1] * seq_len + [0] * (self.max_len - seq_len)
        
        return {
            'input_ids': torch.tensor(ids),
            'attention_mask': torch.tensor(mask),
            'label': torch.tensor(label)
        }


def train(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()
    
    for batch in tqdm(loader, desc="Training"):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(ids, mask)
        loss = loss_fn(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()
    
    for batch in loader:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(ids, mask)
        loss = loss_fn(logits, labels)
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def demo_sentiment(model, tokenizer, sentences, device):
    """Demo: classify sentiment of sentences."""
    model.eval()
    labels = ['Negative', 'Positive']
    pad_id = tokenizer.token_to_id('[PAD]')
    
    print("\n" + "="*60)
    print("Sentiment Classification Demo")
    print("="*60)
    
    for sentence in sentences:
        ids = tokenizer.encode(sentence).ids[:128]
        seq_len = len(ids)
        ids = ids + [pad_id] * (128 - seq_len)
        mask = [1] * seq_len + [0] * (128 - seq_len)
        
        ids_tensor = torch.tensor([ids], device=device)
        mask_tensor = torch.tensor([mask], device=device)
        
        with torch.no_grad():
            logits = model(ids_tensor, mask_tensor)
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1).item()
        
        print(f"\nSentence: {sentence}")
        print(f"Prediction: {labels[pred]} ({probs[0][pred]:.2%})")
        print(f"Confidence: Neg={probs[0][0]:.2%}, Pos={probs[0][1]:.2%}")
        print("-"*40)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_checkpoint', type=str, default='checkpoints/bert_best.pt')
    parser.add_argument('--tokenizer', type=str, default='checkpoints/bert_tokenizer.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_val', type=int, default=None)
    parser.add_argument('--freeze_bert', action='store_true', help='Freeze BERT weights')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Finetuning BERT for sentiment on {device}")
    
    # Load tokenizer
    if not os.path.exists(args.tokenizer):
        print(f"Tokenizer not found: {args.tokenizer}")
        print("Please train BERT first: python train.py --model bert")
        return
    
    tokenizer = Tokenizer.from_file(args.tokenizer)
    
    # Load pretrained BERT
    if not os.path.exists(args.bert_checkpoint):
        print(f"BERT checkpoint not found: {args.bert_checkpoint}")
        print("Please train BERT first: python train.py --model bert")
        return
    
    bert_model = BERTModel.load(args.bert_checkpoint, device)
    
    # Create classifier
    classifier_path = 'checkpoints/bert_sentiment.pt'
    
    if args.eval_only:
        if not os.path.exists(classifier_path):
            print(f"Classifier not found: {classifier_path}")
            return
        model = BERTClassifier.load(classifier_path, bert_model, device)
    else:
        model = BERTClassifier(bert_model, num_classes=2, freeze_bert=args.freeze_bert).to(device)
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {trainable:,} trainable / {total:,} total")
        
        # Create datasets
        train_data = SentimentDataset(tokenizer, 'train', args.max_len, args.max_train)
        val_data = SentimentDataset(tokenizer, 'validation', args.max_len, args.max_val)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)
        
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        best_acc = 0
        
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train(model, train_loader, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                model.save(classifier_path)
        
        print(f"\nTraining complete. Best val accuracy: {best_acc:.2%}")
        
        # Reload best model for demo
        model = BERTClassifier.load(classifier_path, bert_model, device)
    
    # Demo
    test_sentences = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "What a terrible waste of time. The acting was awful.",
        "It was okay, nothing special but not bad either.",
        "The best film I have seen this year!",
        "I really hated this movie. So boring and predictable.",
        "A masterpiece of modern cinema.",
    ]
    demo_sentiment(model, tokenizer, test_sentences, device)


if __name__ == '__main__':
    main()
