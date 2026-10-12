# Wei Xue

Theoretical physicist at the University of Florida (dark matter, early-universe cosmology, particle physics), now working on machine learning: the language-model stack built from the tensor level up, and neural-network methods for many-body physics. Heading toward model internals and interpretability, and RL environments for safety-relevant tasks, where verifiable physics problems are a natural source of ground truth. Publications on [INSPIRE](https://inspirehep.net/authors/1051291) and [Google Scholar](https://scholar.google.com/citations?user=iaziYXMAAAAJ) · [LinkedIn](https://www.linkedin.com/in/wei-xue-34914a115/).

ML as a research instrument, not a black box. The recurring question is *which part of the model is doing the work*, so every project below carries an ablation, an exact check, or a measurement against a known roof. Numbers are reproducible from the commands in each repo.

## Language model from scratch (Aug–Oct 2026)

Following the arc of Stanford's CS336, each piece written on raw tensors and checked against the course tests.

**[lm-from-scratch](https://github.com/xueweiphy/lm-from-scratch)**
- Byte-level BPE tokenizer and a decoder-only Transformer (RMSNorm, SwiGLU, RoPE, multi-head attention, AdamW, cosine schedule, checkpointing) with no `nn.Linear` or `nn.functional` shortcuts.
- 22.7M parameters on TinyStories: validation loss **1.280** (perplexity 3.60) after 450M tokens, about 20 per parameter, in 3.8 h on one A100 MIG slice. An 11× larger token budget, nothing else changed, took the loss from 1.615 to 1.280.
- Tokenizer finding: cross-tokenizing is asymmetric; a broad vocabulary degrades gracefully on narrow text (−3%), a narrow one sharply on broad text (−25%).

**[lm-systems](https://github.com/xueweiphy/lm-systems)**
- FlashAttention-2 from scratch: tiled forward in PyTorch and as a Triton kernel with causal masking, recomputation backward; plus ring all-reduce and a minimal DDP wrapper verified against single-process training.
- Triton forward within **15–30% of PyTorch SDPA** at T = 65536 (RTX Pro 6000, bf16), 55% of the card's measured bf16 peak; 8× naive and 3.4× `torch.compile` on an A100 MIG slice. Backward still 10–30× slower than SDPA's fused kernel, the stated next step.
- PyTorch · Triton · `torch.distributed` · measured roofline and tile-size sweep.

**[lm-post-training](https://github.com/xueweiphy/lm-post-training)** (in progress)
- Reasoning RL on GSM8K with OLMo-2-1B: prompting baselines, then GRPO, RFT, Dr. GRPO and MaxRL, one commit per passing test.
- Baselines on the 1,319 test problems: a 3-shot R1-style prompt lifts format compliance from 62% to 95%, and only then does accuracy appear, **0.1% → 18.5%** (38% greedy).
- vLLM for generation · Slurm on UF HiPerGator (RTX Pro 6000 / B200).

## ML for physics (research)

**[soliton_nqs](https://github.com/xueweiphy/soliton_nqs)**
- Neural quantum state (autoregressive transformer / MLP + variational Monte Carlo) for the many-body ground state of N self-gravitating bosons, the quantum version of the Schrödinger–Poisson soliton.
- Recovers **98–100% of the exact beyond-mean-field correlation energy** where exact diagonalisation is possible (N = 8, 12; error < 0.03% vs 0.7–1.2% for Hartree); predicts a 0.37% beyond-Hartree correction at N = 20, where 3×10⁹ states rule ED out.
- Four-backbone ablation: the transformer is not what makes it work; a structured binomial head initialised at the Hartree state is decisive, and an MLP with 7× fewer parameters matches it. Validated against exact diagonalisation and Bogoliubov theory.

## Foundations (Aug 2026, reimplemented from memory after Karpathy's lectures, not typed along)

**[bpe](https://github.com/xueweiphy/bpe)**
- Byte-level BPE with GPT-2 regex pretokenization, incremental pair counts during merging, parallel pretokenization at document boundaries.
- 10,000-token vocabulary on the 2.2 GB TinyStories corpus in **1.4 min** on 10 processes, 0.9 GB main-process memory; merge ties broken as in the CS336 reference. Later folded into lm-from-scratch.

**[gpt](https://github.com/xueweiphy/gpt)**
- Character-level GPT in PyTorch: multi-head causal attention, residual blocks, LayerNorm, all hand-written; trained on 1.1 MB of arXiv hep-ph abstracts.
- 67k → 1.8M parameters (27×) with a 4× longer context, 5,000 steps each.

**[makemore](https://github.com/xueweiphy/makemore)**
- Character-level language models from bigram counting to an MLP with manual backpropagation.
- NLL 3.30 (uniform) → 2.45 (bigram) → 2.14 (3-character MLP, dev set); includes the diagnosis of a silent bug where the embedding matrix was excluded from training and the loss still fell.

**[micrograd](https://github.com/xueweiphy/micrograd)**
- Scalar reverse-mode autograd engine; forward values and gradients verified against PyTorch to 1e-6, composed operations to 1e-9 on gradients.
- Two GELU forms compared, gap pinned to 4.7e-4.

## Physics code

**[MODProducerMuon](https://github.com/xueweiphy/MODProducerMuon)**
- Data layer of the MIT Open Data project, the first physics analyses of LHC data by researchers outside the CMS collaboration: a CMSSW pipeline converting CERN Open Data AOD files to the MOD format with jets, triggers, luminosity and muons.
- The muon stream fed the first new-physics search on CMS Open Data: narrow dimuon resonances at 14–66 GeV, a pT-enhanced strategy up to 9× more sensitive than the inclusive search ([PRD 100, 015021](https://arxiv.org/abs/1902.04222)), alongside the jet-substructure and QCD splitting-function studies (2017).
- C++/Python · CMSSW · ROOT.

**[AMSfit](https://github.com/xueweiphy/AMSfit)**
- Likelihood fits of cosmic-ray data (AMS-02) to dark-matter and astrophysical models.

---

*Things in progress land here as they ship.*
