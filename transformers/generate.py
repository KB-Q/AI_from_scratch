"""
Generate text using a saved GPT model (NumPy or MLX).
Usage: python3 generate.py <model_path> <vocab_path> <seed_text>
"""
import sys
import pickle
import numpy as np


def generate_text(model_path, vocab_path, seed_text, max_tokens=50, temperature=0.8):
    """Generate text from a saved model (auto-detects NumPy or MLX)."""
    
    # Detect model type from filename
    is_mlx = '_mlx' in model_path
    
    # Load appropriate model
    if is_mlx:
        try:
            from gpt_mlx import GPTModel
            print(f"Loading MLX model from {model_path}...")
        except ImportError:
            print("Error: MLX not installed. Install with: pip install mlx")
            return
    else:
        from gpt_np import GPTModel
        print(f"Loading NumPy model from {model_path}...")
    
    model = GPTModel.load(model_path)
    
    # Load vocabulary
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    word_to_id = vocab['word_to_id']
    id_to_word = vocab['id_to_word']
    
    print(f"Vocabulary size: {len(word_to_id)}")
    print(f"Model: d_model={model.d_model}, layers={len(model.blocks)}")
    print(f"Type: {'MLX (GPU)' if is_mlx else 'NumPy (CPU)'}")
    print()
    
    # Tokenize seed text
    seed_words = seed_text.lower().split()
    seed_ids = [word_to_id.get(w, 0) for w in seed_words]
    
    if not seed_ids:
        print("Error: Seed text is empty or contains only unknown words")
        return
    
    seed_ids = np.array([seed_ids])
    
    # Generate
    print(f"Generating from seed: '{seed_text}'")
    print("-" * 60)
    
    generated_ids = model.generate(seed_ids, max_new_tokens=max_tokens, temperature=temperature)
    
    # Convert to text
    generated_text = ' '.join([id_to_word.get(tid, '<unk>') for tid in generated_ids])
    
    print(generated_text)
    print()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 generate.py <model_path> <vocab_path> <seed_text> [max_tokens] [temperature]")
        print()
        print("Example:")
        print("  python3 generate.py checkpoints/shakespeare_100pct_model.pkl \\")
        print("                      checkpoints/shakespeare_100pct_vocab.pkl \\")
        print("                      'to be or not' 100 0.8")
        print()
        print("  python3 generate.py checkpoints/tinychat_1pct_model.pkl \\")
        print("                      checkpoints/tinychat_1pct_vocab.pkl \\")
        print("                      'hello how are you' 50 1.0")
        sys.exit(1)
    
    model_path = sys.argv[1]
    vocab_path = sys.argv[2]
    seed_text = sys.argv[3]
    
    # Parse optional arguments
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    temperature = float(sys.argv[5]) if len(sys.argv) > 5 else 0.8
    
    generate_text(model_path, vocab_path, seed_text, max_tokens, temperature)
