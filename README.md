# AIML

From-scratch implementations of classic machine learning, deep learning, and recent model architectures, each paired with a derivation note. The repo is a learning resource for applied and research science interviews.

## Structure

- Core modules (standalone models, methods, and building blocks):
  - `01-dsa`: hash table, heap (Python and C++), k-d tree, Levenshtein distance, weighted union-find
  - `02-statistics`: normal and Student-t distributions, hypothesis tests, A/B testing, estimation, Monte Carlo sampling
  - `03-machine-learning`: logistic regression, SVM (primal and kernel dual), naive Bayes, k-means, bias–variance
  - `04-deep-learning`: numpy MLP, bias–variance with neural networks
  - `05-trees`: CART, AdaBoost, gradient boosting, XGBoost (regression, classification, LambdaMART ranking)
  - `07-sequential`: RNN, LSTM, GRU, word2vec, CoLES, BLEU (numpy and torch)
  - `08-transformers`: GPT (numpy, MLX, torch), BERT, T5, LoRA
  - `09-graphs`: GNN, GCN, GAT, GraphSAGE (numpy and torch)
- Derived modules (architectures and pipelines built on the core modules):
  - `21-llm-archs`: LLM evaluation notes; LLM architectures planned
  - `22-recsys`: retrieval and ranking pipeline on MovieLens-100k
  - `23-llm-audio`: minimal Qwen3-TTS
- Inside each module:
  - `scripts/`: code, with `scripts/numpy/` and `scripts/torch/` where both versions exist
  - `notes/`: derivation notes
  - `images/`, `data/`, `checkpoints/`: figures, datasets, and model weights (data and checkpoints are not tracked)
- Other folders:
  - `other-notes/`: notes not yet placed in a module
  - `viz-expts/`: architecture diagrams generated with torchview
- [`SCOPE_2026_10_05.md`](SCOPE_2026_10_05.md): planned additions as a checklist, including the planned `06-vision` and `10-reinforcement-learning` modules.
