"""
Training script for GPT model using MLX (Apple Silicon optimized).
Much faster than NumPy version with automatic differentiation.
"""
import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
import os
import pickle
from gpt_mlx import GPTModel, cross_entropy_loss
from utils_data import (
    download_tiny_shakespeare, download_tinychat, load_and_clean_text, 
    build_vocab, tokenize, create_batches
)


def train_gpt_mlx(epochs=3, lr=0.001, batch_size=None, seq_len=32, dataset='shakespeare', sample_frac=0.1):
    """
    Train GPT model on text dataset using MLX.
    
    Args:
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size (auto-set based on dataset if None)
        seq_len: Sequence length
        dataset: 'shakespeare' or 'tinychat'
        sample_frac: Fraction of TinyChat data to use (0.1 = 10%)
    """
    
    # Auto-set batch size based on dataset
    if batch_size is None:
        batch_size = 64 if dataset.lower() == 'tinychat' else 16
    
    print("=" * 60)
    print(f"GPT Training on {dataset.title()} Dataset (MLX)")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    if dataset.lower() == 'tinychat':
        txt_path = download_tinychat(sample_frac=sample_frac)
    else:
        txt_path = download_tiny_shakespeare()
    text = load_and_clean_text(txt_path)
    word_to_id, id_to_word = build_vocab(text, max_vocab=5000)
    token_ids = tokenize(text, word_to_id)
    
    vocab_size = len(word_to_id)
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Total tokens: {len(token_ids)}")
    
    # Split into train/val
    split_idx = int(len(token_ids) * 0.9)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    
    # Create batches
    print("\n2. Creating batches...")
    train_inputs, train_targets = create_batches(train_ids, seq_len, batch_size)
    val_inputs, val_targets = create_batches(val_ids, seq_len, batch_size)
    
    print(f"   Train batches: {len(train_inputs)}")
    print(f"   Val batches: {len(val_inputs)}")
    print(f"   Batch size: {batch_size}, Sequence length: {seq_len}")
    
    # Initialize model
    print("\n3. Initializing model...")
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=512,
        max_seq_len=seq_len
    )
    print(f"   Model: d_model=128, layers=2, heads=4")
    print(f"   Using MLX (GPU accelerated on Apple Silicon)")
    
    # Initialize optimizer
    optimizer = optim.Adam(learning_rate=lr)
    
    # Loss and gradient function
    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        return cross_entropy_loss(logits, targets)
    
    # Create gradient function
    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    
    # Training loop
    print("\n4. Training...")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Training
        epoch_loss = 0
        num_batches = len(train_inputs)
        
        for batch_idx in range(num_batches):
            # Convert to MLX arrays
            batch_in = mx.array(train_inputs[batch_idx])
            batch_tgt = mx.array(train_targets[batch_idx])
            
            # Forward and backward pass
            loss, grads = loss_and_grad_fn(model, batch_in, batch_tgt)
            
            # Update parameters
            optimizer.update(model, grads)
            
            # Evaluate the loss (MLX uses lazy evaluation)
            mx.eval(model.parameters(), optimizer.state)
            
            epoch_loss += loss.item()
            
            # Print progress
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        # Validation
        val_loss = 0
        for batch_idx in range(len(val_inputs)):
            batch_in = mx.array(val_inputs[batch_idx])
            batch_tgt = mx.array(val_targets[batch_idx])
            
            loss = loss_fn(model, batch_in, batch_tgt)
            val_loss += loss.item()
        
        avg_train_loss = epoch_loss / num_batches
        avg_val_loss = val_loss / len(val_inputs)
        perplexity = np.exp(avg_val_loss)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print("-" * 60)
    
    # Generation
    print("\n5. Generating text...")
    print("-" * 60)
    
    seed_texts = ["to be", "the king", "i am"]
    
    for seed_text in seed_texts:
        # Tokenize seed
        seed_words = seed_text.split()
        seed_ids = [word_to_id.get(w, 0) for w in seed_words]
        seed_ids = np.array([seed_ids])
        
        # Generate
        generated_ids = model.generate(seed_ids, max_new_tokens=20, temperature=0.8)
        
        # Convert back to text
        generated_text = ' '.join([id_to_word.get(tid, '<unk>') for tid in generated_ids])
        
        print(f"\nSeed: '{seed_text}'")
        print(f"Generated: {generated_text}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    # Save model and vocabulary
    os.makedirs('checkpoints', exist_ok=True)
    
    model_name = f"{dataset}_{int(sample_frac*100) if dataset=='tinychat' else 100}pct_mlx"
    model_path = f"checkpoints/{model_name}_model.pkl"
    vocab_path = f"checkpoints/{model_name}_vocab.pkl"
    
    model.save(model_path)
    
    with open(vocab_path, 'wb') as f:
        pickle.dump({'word_to_id': word_to_id, 'id_to_word': id_to_word}, f)
    
    print(f"\nVocabulary saved to {vocab_path}")
    print(f"\nTo load this model later:")
    print(f"  from gpt_model_mlx import GPTModel")
    print(f"  import pickle")
    print(f"  model = GPTModel.load('{model_path}')")
    print(f"  with open('{vocab_path}', 'rb') as f:")
    print(f"      vocab = pickle.load(f)")
    
    return model, word_to_id, id_to_word


if __name__ == "__main__":
    import sys
    
    # Check for dataset argument
    dataset = 'shakespeare'  # default
    sample_frac = 0.1  # default 10% for TinyChat
    
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    if len(sys.argv) > 2:
        sample_frac = float(sys.argv[2])
    
    # Train with reasonable hyperparameters
    model, word_to_id, id_to_word = train_gpt_mlx(
        epochs=3,
        lr=0.001,
        batch_size=None,  # Auto-set based on dataset
        seq_len=32,
        dataset=dataset,
        sample_frac=sample_frac
    )
