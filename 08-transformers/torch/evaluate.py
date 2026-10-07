"""
Evaluation script for GPT, BERT, and T5 models.
Includes metrics and qualitative demos for each model type.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

from gpt_torch import GPTModel
from bert_torch import BERTModel
from t5_torch import T5Model

# Optional: pip install rouge-score sacrebleu
try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("Warning: rouge-score not installed. Run: pip install rouge-score")

try:
    import sacrebleu
    HAS_BLEU = True
except ImportError:
    HAS_BLEU = False
    print("Warning: sacrebleu not installed. Run: pip install sacrebleu")


# ============ GPT EVALUATION ============

@torch.no_grad()
def compute_perplexity(model, tokenizer, texts, device, max_len=128):
    """Compute perplexity on a list of texts."""
    model.eval()
    total_loss, total_tokens = 0, 0
    pad_id = tokenizer.token_to_id('[PAD]')
    loss_fn = nn.CrossEntropyLoss(reduction='sum', ignore_index=pad_id)
    
    for text in tqdm(texts, desc="Computing perplexity"):
        ids = tokenizer.encode(text).ids[:max_len]
        if len(ids) < 2:
            continue
        ids_tensor = torch.tensor([ids], device=device)
        
        logits = model(ids_tensor[:, :-1])
        loss = loss_fn(logits.reshape(-1, model.vocab_size), ids_tensor[:, 1:].reshape(-1))
        
        total_loss += loss.item()
        total_tokens += len(ids) - 1
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    return perplexity


def demo_gpt_generation(model, tokenizer, prompts, device, max_tokens=50, temperature=0.8):
    """Generate text completions for given prompts."""
    model.eval()
    print("\n" + "="*60)
    print("GPT Generation Demo")
    print("="*60)
    
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        seed = torch.tensor([ids], device=device)
        
        generated = model.generate(seed, max_new_tokens=max_tokens, temperature=temperature)
        text = tokenizer.decode(generated.tolist())
        
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {text}")
        print("-"*40)


# ============ BERT EVALUATION ============

@torch.no_grad()
def compute_mlm_accuracy(model, tokenizer, texts, device, max_len=128, mask_prob=0.15):
    """Compute MLM accuracy on texts."""
    model.eval()
    correct, total = 0, 0
    mask_id = tokenizer.token_to_id('[MASK]')
    pad_id = tokenizer.token_to_id('[PAD]')
    vocab_size = tokenizer.get_vocab_size()
    
    for text in tqdm(texts, desc="Computing MLM accuracy"):
        ids = tokenizer.encode(text).ids[:max_len]
        if len(ids) < 3:
            continue
        
        # Create masked version
        input_ids = ids.copy()
        labels = []
        mask_positions = []
        
        for i in range(len(ids)):
            if np.random.random() < mask_prob:
                labels.append(ids[i])
                mask_positions.append(i)
                r = np.random.random()
                if r < 0.8:
                    input_ids[i] = mask_id
                elif r < 0.9:
                    input_ids[i] = np.random.randint(0, vocab_size)
        
        if not mask_positions:
            continue
        
        ids_tensor = torch.tensor([input_ids], device=device)
        logits = model(ids_tensor)
        preds = logits[0].argmax(dim=-1)
        
        for pos, label in zip(mask_positions, labels):
            if preds[pos].item() == label:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def demo_fill_in_blank(model, tokenizer, sentences, device):
    """Demo: fill in [MASK] tokens in sentences."""
    model.eval()
    mask_id = tokenizer.token_to_id('[MASK]')
    
    print("\n" + "="*60)
    print("BERT Fill-in-the-Blank Demo")
    print("="*60)
    
    for sentence in sentences:
        # Tokenize and find mask positions
        ids = tokenizer.encode(sentence).ids
        mask_positions = [i for i, x in enumerate(ids) if x == mask_id]
        
        if not mask_positions:
            print(f"\nNo [MASK] in: {sentence}")
            continue
        
        ids_tensor = torch.tensor([ids], device=device)
        with torch.no_grad():
            logits = model(ids_tensor)
        
        # Get top-5 predictions for each mask
        result_ids = ids.copy()
        print(f"\nInput: {sentence}")
        
        for pos in mask_positions:
            probs = F.softmax(logits[0, pos], dim=-1)
            top5 = torch.topk(probs, 5)
            
            # Use top prediction
            result_ids[pos] = top5.indices[0].item()
            
            top5_words = [tokenizer.decode([idx.item()]) for idx in top5.indices]
            top5_probs = [f"{p:.3f}" for p in top5.values.tolist()]
            print(f"  Position {pos} predictions: {list(zip(top5_words, top5_probs))}")
        
        filled = tokenizer.decode(result_ids)
        print(f"Filled: {filled}")
        print("-"*40)


# ============ T5 EVALUATION ============

@torch.no_grad()
def compute_rouge(model, tokenizer, sources, references, device, max_len=128):
    """Compute ROUGE scores for summarization."""
    if not HAS_ROUGE:
        print("ROUGE not available. Install: pip install rouge-score")
        return {}
    
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    bos_id = tokenizer.token_to_id('[BOS]')
    eos_id = tokenizer.token_to_id('[EOS]')
    
    for src, ref in tqdm(zip(sources, references), desc="Computing ROUGE", total=len(sources)):
        src_ids = tokenizer.encode(src).ids[:max_len-1] + [eos_id]
        src_tensor = torch.tensor([src_ids], device=device)
        
        generated = model.generate(src_tensor, max_new_tokens=max_len//2, 
                                   start_token_id=bos_id, eos_token_id=eos_id)
        hypothesis = tokenizer.decode(generated[0].tolist())
        
        score = scorer.score(ref, hypothesis)
        for key in scores:
            scores[key].append(score[key].fmeasure)
    
    return {k: np.mean(v) for k, v in scores.items()}


@torch.no_grad()
def compute_bleu(model, tokenizer, sources, references, device, max_len=128):
    """Compute BLEU score for translation."""
    if not HAS_BLEU:
        print("BLEU not available. Install: pip install sacrebleu")
        return 0.0
    
    model.eval()
    bos_id = tokenizer.token_to_id('[BOS]')
    eos_id = tokenizer.token_to_id('[EOS]')
    
    hypotheses = []
    for src in tqdm(sources, desc="Generating translations"):
        src_ids = tokenizer.encode(src).ids[:max_len-1] + [eos_id]
        src_tensor = torch.tensor([src_ids], device=device)
        
        generated = model.generate(src_tensor, max_new_tokens=max_len//2,
                                   start_token_id=bos_id, eos_token_id=eos_id)
        hypothesis = tokenizer.decode(generated[0].tolist())
        hypotheses.append(hypothesis)
    
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


def demo_seq2seq(model, tokenizer, sources, task, device, max_len=128):
    """Demo: generate summaries or translations."""
    model.eval()
    bos_id = tokenizer.token_to_id('[BOS]')
    eos_id = tokenizer.token_to_id('[EOS]')
    
    task_name = "Summarization" if task == "summarization" else "Translation"
    print("\n" + "="*60)
    print(f"T5 {task_name} Demo")
    print("="*60)
    
    for src in sources:
        src_ids = tokenizer.encode(src).ids[:max_len-1] + [eos_id]
        src_tensor = torch.tensor([src_ids], device=device)
        
        with torch.no_grad():
            generated = model.generate(src_tensor, max_new_tokens=max_len//2,
                                       start_token_id=bos_id, eos_token_id=eos_id,
                                       temperature=0.8)
        
        output = tokenizer.decode(generated[0].tolist())
        
        print(f"\nSource: {src[:200]}..." if len(src) > 200 else f"\nSource: {src}")
        print(f"Output: {output}")
        print("-"*40)


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['gpt', 'bert', 't5'])
    parser.add_argument('--task', type=str, default='summarization', choices=['summarization', 'translation'])
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default=None, help='Path to tokenizer')
    parser.add_argument('--max_samples', type=int, default=100, help='Max samples for evaluation')
    parser.add_argument('--max_len', type=int, default=128)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluating {args.model.upper()} on {device}")
    
    # Load checkpoint and tokenizer
    checkpoint_path = args.checkpoint or f'checkpoints/{args.model}_best.pt'
    tokenizer_path = args.tokenizer or f'checkpoints/{args.model}_tokenizer.json'
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found: {tokenizer_path}")
        return
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Load model
    if args.model == 'gpt':
        model = GPTModel.load(checkpoint_path, device)
    elif args.model == 'bert':
        model = BERTModel.load(checkpoint_path, device)
    elif args.model == 't5':
        model = T5Model.load(checkpoint_path, device)
    
    # Run evaluations
    if args.model == 'gpt':
        # Load test data
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        texts = [x['text'] for x in dataset if len(x['text'].strip()) > 20][:args.max_samples]
        
        # Perplexity
        ppl = compute_perplexity(model, tokenizer, texts, device, args.max_len)
        print(f"\nPerplexity: {ppl:.2f}")
        
        # Generation demo
        prompts = [
            "The history of",
            "In the year 2024",
            "Scientists have discovered",
            "The most important thing"
        ]
        demo_gpt_generation(model, tokenizer, prompts, device)
        
    elif args.model == 'bert':
        # Load test data
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        texts = [x['text'] for x in dataset if len(x['text'].strip()) > 20][:args.max_samples]
        
        # MLM Accuracy
        accuracy = compute_mlm_accuracy(model, tokenizer, texts, device, args.max_len)
        print(f"\nMLM Accuracy: {accuracy:.2%}")
        
        # Fill-in-the-blank demo
        mask_token = "[MASK]"
        sentences = [
            f"The capital of France is {mask_token}.",
            f"Water is made of hydrogen and {mask_token}.",
            f"The {mask_token} rises in the east.",
            f"Machine {mask_token} is transforming technology."
        ]
        # Encode with mask token
        demo_fill_in_blank(model, tokenizer, sentences, device)
        
    elif args.model == 't5':
        if args.task == 'summarization':
            dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
            sources = [x['article'] for x in dataset]
            references = [x['highlights'] for x in dataset]
            
            # ROUGE scores
            rouge_scores = compute_rouge(model, tokenizer, sources[:50], references[:50], device, args.max_len)
            print(f"\nROUGE Scores:")
            for k, v in rouge_scores.items():
                print(f"  {k}: {v:.4f}")
            
            # Demo
            demo_seq2seq(model, tokenizer, sources[:3], args.task, device, args.max_len)
            
        elif args.task == 'translation':
            dataset = load_dataset('wmt14', 'de-en', split='test')
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
            sources = [x['translation']['de'] for x in dataset]
            references = [x['translation']['en'] for x in dataset]
            
            # BLEU score
            bleu = compute_bleu(model, tokenizer, sources[:50], references[:50], device, args.max_len)
            print(f"\nBLEU Score: {bleu:.2f}")
            
            # Demo
            demo_seq2seq(model, tokenizer, sources[:3], args.task, device, args.max_len)


if __name__ == '__main__':
    main()
