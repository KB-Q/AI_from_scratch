# Transformers Training Workflows

This document outlines the training setups for the different models implemented in this repository—GPT, BERT, and T5. The models are organized into PyTorch implementations (`torch/` directory) and Numpy/MLX implementations (`numpy/` directory), each with distinct data handling and training mechanics.

---

## 1. Causal Language Modeling (GPT)

The GPT models are autoregressive, predicting the next token in a sequence. We have implementations in both PyTorch and NumPy/MLX that utilize entirely different data and batching mechanisms.

### PyTorch Setup (`torch/train.py`, `torch/gpt_torch.py`)
* **Data Loading**: Uses the `LMDataset` to load the `wikitext-2-raw-v1` dataset from the Hugging Face `datasets` API. The dataset uses a custom BPE tokenizer.
* **Batching**: Standard PyTorch `DataLoader` is used. Tokenized examples are simply fetched and returned. 
* **Padding**: Shorter sequences are manually padded on the right with a `[PAD]` token IDs out to `max_len` (128).
* **Masking**: 
  * *Causal mask*: Inside `GPTModel`’s forward pass, an upper-triangular matrix filled with `-1e9` is created (`torch.triu`). This is applied onto the self-attention scores so the model can't look at future tokens.
* **Loss Function**: Built-in `nn.CrossEntropyLoss(ignore_index=model.vocab_size)`. The model's inputs are shifted right `ids[:, :-1]` and targets are shifted left `ids[:, 1:]`. The `[PAD]` mask is ignored.

### NumPy & MLX Setup (`numpy/train_np.py`, `numpy/train_mlx.py`, `utils_data.py`)
* **Data Loading**: Custom helper functions fetch either `Tiny Shakespeare` or `TinyChat` dataset. A custom vocab is built (restricted to 5,000 top words, mostly composed of lowercased alpha strings).
* **Batching**: The continuous sequence of text tokens is aggressively chopped into sequences of length `seq_len + 1` with a stride offset of `seq_len // 2` to create overlapping chunks. The arrays are then truncated down to perfect multiples of `batch_size`.
* **Padding**: **None.** Because sequences are uniformly chunked from a continuous text stream down to an exact fit, there are no padded inputs. 
* **Masking**: Similar to PyTorch, upper-triangular causal masks are added inside the custom MultiHeadAttention forward passes (`mx.triu` in MLX). 
* **Loss Function**: Custom-built `cross_entropy_loss` using highly stable `softmax` and explicit log probabilities. It computes loss as `-log(p)` of target labels and manually returns `-mean(log_probs)` along with analytic gradients (NumPy variant: `probs - 1 / N` at correct indices; MLX calculates derivatives automatically over this custom function).

---

## 2. Masked Language Modeling (BERT)

Implemented in PyTorch (`torch/train_bert.py`, `torch/bert_torch.py`). Designed as a bidirectional encoder-only model for understanding the context.

* **Data Loading**: Handled by `MLMDataset` scanning the `wikitext-2-raw-v1` text splits using BPE tokens.
* **Batching**: Iterates using PyTorch `DataLoader`.
* **Padding**: 
  * Pads up to `max_seq_len` manually. 
  * Alongside input IDs, it generates an `attention_mask` (array of `1` for original tokens, `0` for padding) which later gets processed inside the BERTModel to `-1e9` penalty values.
* **Masking**: Employs classical BERT sequence corruption. 15% of actual tokens in each sequence are chosen to be predicted. Out of this 15%:
  * 80% are replaced with the explicit `[MASK]` ID.
  * 10% are replaced with a random token from the vocab.
  * 10% remain unchanged. 
* **Loss Function**: The unmasked (unabused) positions are set to an expected label of `-100`. The loss utilizes `nn.CrossEntropyLoss(ignore_index=-100)`, meaning the optimizer completely ignores standard tokens and only propagates gradients for the 15% of corrupted target slots.

---

## 3. Seq2Seq / Encoder-Decoder (T5)

Implemented in PyTorch (`torch/train_t5.py`, `torch/t5_torch.py`). 

* **Data Loading**: Facilitated by `Seq2SeqDataset`. Can pull from either `wmt14` for translation tasks or `cnn_dailymail` for summarization. The class splits article texts into "encoder sources" and highlights as "decoder targets," explicitly bounding them with `[BOS]` and `[EOS]` tokens.
* **Batching**: Fixed lengths are strictly enforced. The source (`src_len`) gets mapped differently than the target (`tgt_len`, usually much smaller).
* **Padding**: Both sequences get independently padded to their specific constrained lengths using `[PAD]`.
* **Masking**: Two separate masks are in play. 
  * **Encoder mask**: Like BERT, the encoder attends to the whole sequence (Bidirectional) without causal masking.
  * **Decoder Causal mask**: An upper-triangular `-1e9` matrix is used on self-attention so the decoder must auto-regressively generate outcomes. It then looks at the unified encoder output through Cross-Attention. 
* **Loss Function**: The decoder labels shift the ground targets right by 1 position (skipping `[BOS]`), allowing them to predict the next word. The remaining targets after `[EOS]` are labeled `-100`. The loss incorporates PyTorch's `nn.CrossEntropyLoss(ignore_index=-100)`.
