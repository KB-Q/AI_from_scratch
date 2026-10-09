# Scope: planned additions

Modules in implementation order, one at a time.

## 1. `02-statistics`

- `distributions.py`: normal and Student-t pdf, CDF, and quantile; the t CDF via the regularised incomplete beta function (continued fraction), quantiles by bisection on the CDF.
- `hypothesis_tests.py`: t-test, permutation test, bootstrap confidence intervals; Bonferroni and Benjamini–Hochberg corrections. (Efron 1979; Benjamini 1995)
- `ab_testing.py`: power and sample-size calculation; false-positive inflation from repeated peeking (simulated); CUPED variance reduction. (Deng 2013)
- `estimation.py`: MLE and MAP; conjugate priors (Beta–Binomial, Normal–Normal); bias and variance of estimators.
- `monte_carlo.py`: inverse-CDF and rejection sampling, importance sampling, Metropolis–Hastings. (Metropolis 1953; Hastings 1970)

## 2. `03-machine-learning`

- `metrics.py`: confusion matrix, precision / recall / F1, ROC-AUC (and its Mann–Whitney U form), PR-AUC, log loss, MSE, R²; sits with `notes/clf metrics.md`. (Hanley 1982)
- `linreg.py` (currently empty): OLS via normal equations and gradient descent; ridge (closed form); lasso (coordinate descent with soft-thresholding). (Tibshirani 1996)
- `kmeans.py` (currently a stub): Lloyd's algorithm, k-means++ initialisation, inertia and silhouette for choosing k. (Arthur 2007)
- `pca.py`: PCA via covariance eigendecomposition and via SVD; explained variance; reconstruction error.
- `gmm.py`: Gaussian mixture fitted with EM; relation to k-means (soft vs. hard assignment). (Dempster 1977)
- `calibration.py`: reliability diagram, expected calibration error, temperature scaling, Platt scaling. (Guo 2017; Platt 1999)
- scikit-learn models and metrics replaced by the repo's: `Ridge` and `mean_squared_error` in `bv_example_poly.py`; accuracy and ROC-AUC in `SVM.py` and `logreg.py`.

## 3. `04-deep-learning`

- `autograd.py`: reverse-mode autodiff on a small numpy tensor (add, mul, matmul, sum, exp, log, ReLU, stable softmax cross-entropy); topological-sort backward. (Baydin 2018; Karpathy 2020, micrograd)
- `init_norm.py`: Xavier and He initialisation with activation statistics across depth; BatchNorm (train vs. eval statistics), LayerNorm, and RMSNorm with numpy backward passes. (Glorot 2010; He 2015; Ioffe 2015; Ba 2016)
- `optimizers.py`: (Kingma 2015; Loshchilov 2019; Jordan 2024; Kimi Team 2025)
  - SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW (decoupled weight decay vs. L2);
  - Muon (Newton–Schulz orthogonalisation), per-head Muon, QK-clip;
  - warmup + cosine schedule, gradient clipping;
  - compared on a 2-D test function and an MLP.
- Generative models (each learns p(x) and samples from it):
  - `vae.py`: MLP encoder/decoder on MNIST; ELBO, reparameterisation trick, KL term; latent interpolation (Kingma 2014);
  - `gan.py`: MLP generator and discriminator on MNIST; minimax vs. non-saturating generator loss; mode-collapse diagnostics (Goodfellow 2014);
  - `diffusion.py`: DDPM on 2-D toy data with an MLP denoiser; noising schedule, ε-prediction loss, ancestral sampling, classifier-free guidance (Ho 2020; Ho 2022);
  - `flow_matching.py`: conditional flow matching / rectified flow on the same 2-D data; Euler sampling with a configurable number of steps (Lipman 2023; Liu 2023).
- scikit-learn metrics in `neural_network.py` replaced by `03-machine-learning/metrics.py`.

## 4. `05-trees`

- `random_forest.py`: bagging over `core_cart.py` trees, per-split feature subsampling, out-of-bag error, impurity-based feature importance. (Breiman 2001)
- Fix the LambdaMART gradient in `metrics.py`: it has the wrong sign and uses σ(s_i − s_j) for both pair orders; for a pair where document i is more relevant, the gradient on s_i is −σ(s_j − s_i)·|ΔNDCG|. (Burges 2010)
- Fix `core_gbm.py`: it fits classification trees to binarised residuals; give it a CART regressor (in `core_cart.py`) that fits the pseudo-residuals. (Friedman 2001)
- scikit-learn metrics in `train_generic.py` and `train_xgb.py` replaced by `03-machine-learning/metrics.py`.

## 5. `08-transformers`

```
08-transformers/
  bpe.py             byte-level BPE
  sampling.py        decoding strategies
  layers/            torch building blocks, imported by derived modules
    norms.py, rope.py, ffn.py
    gqa.py, mla.py, sliding_window.py, linear_attention.py, sparse_attention.py
    moe.py, residuals.py, mtp.py
  torch/
    gpt_torch.py     (extend) KV cache
    dense_decoder.py Llama-3 / Qwen3-style decoder assembled from layers/
    vit.py           built after 06-vision
```

- Tokenisation and decoding:
  - `bpe.py`: byte-level BPE (train merges, encode, decode), compared with the `tokenizers` BPE used in `torch/train.py` (Sennrich 2016; Radford 2019);
  - `sampling.py`: greedy, temperature, top-k, top-p (nucleus), and min-p over a logits vector; used by the GPT `generate` methods (Holtzman 2020; Nguyen 2024);
  - `torch/gpt_torch.py` (extend): KV cache for incremental decoding.
- Primitives:
  - `layers/norms.py`: LayerNorm, RMSNorm, QK-norm (Ba 2016; Zhang 2019);
  - `layers/rope.py`: RoPE, partial RoPE, YaRN context extension (Su 2021; Peng 2023);
  - `layers/ffn.py`: GELU MLP, SwiGLU, clamped SwiGLU (DeepSeek-V4, GLM-5.3-Flash), SiTU-GLU (Kimi K3: soft-capped gate and up branches) (Shazeer 2020).
- Attention:
  - `layers/gqa.py`: MHA, GQA, and MQA as one class parameterised by the number of KV heads; KV cache; KV-cache bytes per token (Vaswani 2017; Shazeer 2019; Ainslie 2023);
  - `layers/mla.py`: low-rank joint KV compression and decoupled RoPE key; weight absorption at inference alongside the naive path; NoPE and sigmoid output-gate options (DeepSeek-V2 2024; Kimi K3 2026);
  - `layers/sliding_window.py`: sliding-window causal mask, rolling KV cache, learnable per-head attention sink (Beltagy 2020; Jiang 2023; Xiao 2024);
  - `layers/linear_attention.py`: kernelised linear attention → DeltaNet → Gated DeltaNet → KDA (channel-wise decay), recurrent form; KDA with ShortConv, L2-normalised q/k, lower-bounded log-decay, and an output gate (Katharopoulos 2020; Yang 2024a, 2024b; Kimi Team 2025);
  - `layers/sparse_attention.py`: DSA lightning indexer with top-k token selection; CSA (compress every m tokens over overlapping windows, indexer picks top-k entries); HCA (compress every m' tokens, dense attention over all entries); a sliding-window branch alongside CSA and HCA (DeepSeek-V3.2 2025; DeepSeek-V4 2026).
- Mixture of experts, residuals, multi-token prediction:
  - `layers/moe.py`: top-k token-choice routing with a load-balancing loss; fine-grained routed + shared experts; auxiliary-loss-free bias balancing and Quantile Balancing (K3); softmax, sigmoid, and sqrt-softplus affinities; hash routing; LatentMoE with RMSNorm before the up-projection (K3) (Shazeer 2017; Fedus 2022; Roller 2021; Dai 2024; Wang 2024; Elango 2026);
  - `layers/residuals.py`: pre-norm residual → hyper-connections (n parallel streams) → mHC (Sinkhorn–Knopp projection to a doubly-stochastic mixing matrix); AttnRes, full and block variants (Zhu 2025; Xie 2026; Kimi Team 2026);
  - `layers/mtp.py`: sequential extra-depth modules predicting tokens t+2 … t+D, with loss weighting (Gloeckle 2024; DeepSeek-V3 2024).
- Assembled models:
  - `torch/dense_decoder.py`: Llama-3 / Qwen3-style decoder (RMSNorm, RoPE, GQA with QK-norm, SwiGLU) from `layers/`; the baseline for `21-llm-archs`;
  - `torch/vit.py`: patch embedding, class token, learned position embeddings, encoder blocks from `layers/`; on CIFAR-10 (Dosovitskiy 2021).
- `notes/`: KV-cache bytes per token for each attention variant (keys and values, or compressed latents and indexer keys, summed over layers at a stated precision), at a toy config and at each released model's config.

## 6. `23-llm-audio` refactor

- `torch/qwen3_tts.py`: replace its copies of RMSNorm, RoPE, GQA, and SwiGLU with imports from `08-transformers/layers/`; keep the Qwen3-TTS-specific parts (codec, dual-track prompt, MTP head over codebooks).
- `notes/tts-qwen3.md`: rewrite as a derivation note.

## 7. `21-llm-archs`

```
21-llm-archs/
  deepseek_v3/  deepseek_v4/  deepseek_v4_1/  kimi_k3/  glm_5_3_flash/
  post_training/      (module 10)
  evals/              (module 10)
  train.py
  notes/
```

Each architecture imports `08-transformers/layers/`, implements only what is specific to it, uses a toy config, and reports total and active parameters.

- `deepseek_v3/`:
  - from core: MLA, MoE (shared + fine-grained routed experts, sigmoid affinity, auxiliary-loss-free bias), MTP, RMSNorm, SwiGLU;
  - model-specific: layer schedule with leading dense layers;
  - excluded: FP8 training, node-limited routing, DualPipe.
- `deepseek_v4/`:
  - from core: CSA, HCA, sliding-window attention with sink, partial RoPE, mHC, MoE (hash routing, sqrt-softplus affinity), clamped SwiGLU, MTP;
  - model-specific: the sliding-window / CSA / HCA layer schedule, single shared K=V head, grouped low-rank output projection, hash-routed bootstrap layers;
  - excluded: FP4/FP8 kernels, 1M-token training;
  - source: report arXiv 2606.19348; only the abstract and the Hugging Face DeepSeek-V4 docs have been read, and the CSA compressor and indexer equations need the full report.
- `deepseek_v4_1/`:
  - builds on `deepseek_v4/`;
  - model-specific: causal encoder–decoder (the decoder's global KV is projected from the final encoder hidden states); CSA2 cross-layer sharing (Full / Reindex / Reuse layer modes, hierarchical sparse indexer); Engram token-lookup memory;
  - excluded: FP4 KV cache, SWA bounded replay, DSpark speculative decoding, the single-pass mHC kernel, the vision encoder;
  - source: report arXiv 2609.19969, not read yet; the model card covers the encoder–decoder split and CSA2 in one paragraph each.
- `kimi_k3/`:
  - from core: KDA, MLA with NoPE and output gate, AttnRes (block variant), MoE (LatentMoE, Quantile Balancing), SiTU-GLU;
  - model-specific: the layer schedule (repeating groups of 3 KDA + 1 gated-MLA layer, plus a final gated-MLA layer), Stable LatentMoE assembly with 2 full-width shared experts;
  - excluded: MoonViT-V2 vision encoder, MXFP4 quantisation-aware training, Per-Head Muon (in `04-deep-learning`);
  - source: report arXiv 2607.24653; its architecture section gives every equation.
- `glm_5_3_flash/`:
  - from core: KDA, MLA with NoPE, DSA lightning indexer, mHC (4 streams), MoE (fine-grained routed experts + 1 shared expert, sigmoid affinity, auxiliary-loss-free bias), clamped SwiGLU, MTP;
  - model-specific: the layer schedule (repeating groups of 3 KDA + 1 sparse-MLA layer, plus a final KDA layer; dense FFNs in the first 3 layers); KPool indexing (indexer keys averaged over 4-token pools with learned softmax weights, top-k over pools, the trailing incomplete pool always kept); an unweighted mean that collapses the mHC streams before the output head; no positional encoding in the text stack;
  - excluded: the vision encoder, FP8 weights, cross-layer top-k sharing (in the modeling code, unused in the released config);
  - source: no technical report; the released `config.json` and the Hugging Face Transformers `glm5_next` modeling code define every component (`vukrosic/glm-5.3-flash-from-scratch` replaces KDA, DSA, and mHC with simpler stand-ins, so it is not a reference).
- `train.py`: a minimal loop that trains any architecture config on TinyShakespeare with `torch.optim.AdamW`.

## 8. `06-vision`

- `numpy/conv.py`: Conv2d forward and backward via im2col; max and average pooling; small CNN on MNIST. (LeCun 1998)
- `torch/resnet.py`: residual blocks with BatchNorm; small ResNet on CIFAR-10; plain vs. residual network at equal depth. (He 2016)
- `detection.py`: IoU, non-maximum suppression, anchor matching, mean average precision (numpy); no full detector. (Ren 2015; Everingham 2010)
- Then `08-transformers/torch/vit.py`.

## 9. `10-reinforcement-learning`

- `envs.py`: gridworld and CartPole (cart-pole dynamics integrated with Euler steps), plus a toy contextual policy setting (K candidate "responses" per "prompt") for the preference objectives.
- `bandits.py`: ε-greedy, UCB1, Thompson sampling; cumulative regret. (Auer 2002; Thompson 1933)
- `tabular.py`: value iteration, policy iteration, Q-learning, SARSA on the gridworld. (Sutton & Barto 2018)
- `dqn.py`: replay buffer, target network, double DQN on CartPole. (Mnih 2015; van Hasselt 2016)
- `policy_gradient.py`: REINFORCE with baseline → advantage actor-critic with GAE → PPO clipped objective, on CartPole. (Williams 1992; Schulman 2016, 2017)
- `grpo.py`: group-sampled actions per context, group-normalised advantages, PPO-style clipped objective with a KL penalty to a reference policy, on the toy contextual policy. (Shao 2024)
- `dpo.py`: DPO loss on preference pairs over the toy contextual policy; relation to the KL-regularised reward objective. (Rafailov 2023)

## 10. `21-llm-archs` post-training and evals

- `post_training/`: applies `10-reinforcement-learning/grpo.py` and `dpo.py` to the `08-transformers` dense decoder: sequence log-probabilities, a frozen reference copy, sampled completions, and a verifiable synthetic task such as toy arithmetic.
- `evals/`: unbiased pass@k estimator; paired bootstrap for two models on the same items (bootstrap from `02-statistics`); variance across seeds and decoding settings. (Chen 2021; Miller 2024)

## Parked and open questions

Implemented or decided only on request.

- `03-machine-learning`: k-NN; Gaussian-process regression.
- `04-deep-learning`: mixed-precision note (fp16 / bf16 ranges, loss scaling); normalising flows.
- `06-vision`: U-Net (segmentation, and the standard image-diffusion denoiser); DCGAN; CLIP-style contrastive image–text training; an end-to-end detector (YOLO / DETR).
- `07-sequential`: seq2seq with additive attention; HMM (forward–backward, Viterbi, Baum–Welch); SSMs (S4D → Mamba selective scan).
- `08-transformers`: chunkwise-parallel form of linear attention / KDA; online-softmax tiled attention (the FlashAttention algorithm, numpy); speculative decoding with the MTP module as the draft.
- `09-graphs`: node2vec; GIN and the Weisfeiler–Lehman test; link prediction; graph transformers.
- `21-llm-archs`: Bradley–Terry reward model; on-policy distillation; quantisation (int8, GPTQ-style, QAT).
- `22-recsys`: DeepFM; SASRec; MMoE; off-policy evaluation (IPS, SNIPS, doubly robust).
- `23-llm-audio`: CTC loss and decoding with a tiny acoustic model.
- Unplaced topics (whether to add them, and where):
  - causal inference: propensity scores and inverse-propensity weighting, difference-in-differences, uplift modelling;
  - time-series forecasting: ARIMA / exponential smoothing, and a neural forecaster;
  - InfoNCE as a core building block (today only inside `07-sequential` CoLES and the `22-recsys` two-tower retriever);
  - information theory: entropy, cross-entropy, KL divergence, mutual information;
  - scaling laws: compute-optimal model and data size (Kaplan 2020; Hoffmann 2022).
- Placement of `other-notes/` (ASR and TTS architectures, NLP basics, transaction embeddings).

## Out of scope

- GPU kernels (CUDA, Triton).
- Serving-efficiency features (FP4 / MXFP4 formats, KV-cache offloading and replay, paged KV cache, batching schedulers).
- Training runs, experiments, and model surgery (ablations, probing, pruning, loading released checkpoints); planned as a separate track.
- Distributed training.
- Vision encoders inside the LLM architectures; `08-transformers/torch/vit.py` covers the mechanism.
