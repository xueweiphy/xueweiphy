# Wei Xue

Theoretical physicist at the University of Florida (dark matter, early-universe cosmology, particle physics), now working on machine learning: language models and their training systems, neural quantum states, and reliable scientific agents. Publications on [INSPIRE](https://inspirehep.net/authors/1051291) and [Google Scholar](https://scholar.google.com/citations?user=iaziYXMAAAAJ) · [LinkedIn](https://www.linkedin.com/in/wei-xue-34914a115/).

My interests are model internals and interpretability, and how training rewards shape an agent's behavior. I approach these questions through controlled experiments: implement the method, check it against a reference, and use ablations to identify which part of the model is doing the work. Numbers below are reproducible from the commands in each repo.

## Neural quantum states for gravitating bosons (original research)

**[soliton_nqs](https://github.com/xueweiphy/soliton_nqs)** connects my research on boson stars to neural-network methods for many-body physics.

- Autoregressive Transformer and MLP wavefunctions trained with variational Monte Carlo for the quantum ground state of N self-gravitating bosons, the quantum version of the Schrödinger–Poisson soliton.
- Recovers **98–100% of the exact beyond-mean-field correlation energy** where exact diagonalisation is possible (N = 8, 12; error < 0.03% vs 0.7–1.2% for Hartree); predicts a 0.37% beyond-Hartree correction at N = 20, where 3×10⁹ states rule exact diagonalisation out. Checked against exact diagonalisation and Bogoliubov theory.
- Four-backbone ablation: the Transformer is not what makes it work; a structured binomial head initialised at the Hartree state is decisive, and an MLP with 7× fewer parameters matches it.
- The public benchmarks use a radial-lattice model. Additional 3D-lattice checks are described in the [validation notes](https://github.com/xueweiphy/soliton_nqs/blob/main/HARTREE_CHECK.md), with that code kept separately.

## Language models from tensors to training (following Stanford CS336)

**[lm-from-scratch](https://github.com/xueweiphy/lm-from-scratch)**
- Byte-level BPE tokenizer and a decoder-only Transformer (RMSNorm, SwiGLU, RoPE, multi-head attention, AdamW, cosine schedule, checkpointing), written directly on tensors and `torch.nn.Module` without `nn.Linear`, `nn.Embedding` or `nn.functional` shortcuts, and checked against the course's reference tests.
- 22.7M parameters on TinyStories: validation loss **1.280** (perplexity 3.60) after 450M tokens, about 20 per parameter, in 3.8 h on one A100 MIG slice. Extending training from 5,000 to 55,000 steps reduced the loss from 1.615 to 1.280, and the curve is still falling.
- Architecture ablations at 5,000 steps: removing RMSNorm diverges at step 300; NoPE costs 0.09 nats (1.622 → 1.715), post-norm and a SiLU FFN about 0.03.
- Tokenizer transfer, measured in bytes per token: an OpenWebText 32K vocabulary loses about 3% on TinyStories, while a TinyStories 10K vocabulary loses 25% on OpenWebText, each relative to the corpus's own tokenizer.

**[lm-systems](https://github.com/xueweiphy/lm-systems)**
- FlashAttention-2 from scratch: tiled forward in PyTorch and as a Triton kernel with causal masking, plus a recomputation backward in compiled PyTorch; outputs and gradients checked against PyTorch SDPA.
- Forward-pass benchmarks: within **15–30% of SDPA** at T = 65536 on an RTX Pro 6000 (bf16, tile 64), reaching 55% of the card's measured bf16 peak against SDPA's 64%; **8× naive PyTorch and 3.4× `torch.compile`** at T = 16384 on an A100 MIG slice. The backward is still 10–30× slower than SDPA, the stated next step.
- Ring all-reduce in NumPy with tests, a minimal DDP wrapper, and collective benchmarks over message size and world size. Tile-size sweep, roofline analysis and raw benchmark tables are in the repo.

**[lm-post-training](https://github.com/xueweiphy/lm-post-training)** (in progress)
- Prompting baselines for OLMo-2-1B on the 1,319 GSM8K test questions with vLLM: a 3-shot R1-style prompt lifts format compliance from 62% to 95% and accuracy from **0.1% to 18.5%** at temperature 1.0; a greedy check on the first 100 questions reaches 38%.
- Implemented the on-policy GRPO components from scratch (prompt/response tokenization and masking, response log-probs, rollout rewards, group-normalized advantages, policy-gradient loss, masked aggregation, microbatched train step), each checked against reference tests. Integrating vLLM rollouts into the training loop is next; RFT, Dr. GRPO and MaxRL comparisons are planned.
- Slurm on UF HiPerGator (RTX Pro 6000 / B200).

## Foundations

Rebuilt from memory after Karpathy's lectures, Aug 2026.

| Repository | What it is |
|---|---|
| [micrograd](https://github.com/xueweiphy/micrograd) | Scalar reverse-mode autograd engine; values and gradients verified against PyTorch to 1e-6. |
| [makemore](https://github.com/xueweiphy/makemore) | Character-level models from bigram counting to an MLP with manual backpropagation; NLL 3.30 → 2.14. |
| [gpt](https://github.com/xueweiphy/gpt) | Character-level Transformer in PyTorch without `nn.Transformer`, trained on arXiv hep-ph abstracts; 67k → 1.8M parameters. |
| [bpe](https://github.com/xueweiphy/bpe) | Byte-level BPE with incremental pair counts and parallel pretokenization: 10k vocabulary on 2.2 GB of TinyStories in 1.4 min on 10 processes. Later folded into lm-from-scratch. |

## Physics code

**[MODProducerMuon](https://github.com/xueweiphy/MODProducerMuon)**
- Data layer of the MIT Open Data project, the first physics analyses of LHC data by researchers outside the CMS collaboration: a CMSSW pipeline converting CERN Open Data AOD files to the MOD format with jets, triggers, luminosity and muons.
- Its muon stream fed the first new-physics search on CMS Open Data, a dimuon-resonance search at 14–66 GeV ([PRD 100, 015021](https://arxiv.org/abs/1902.04222)), alongside the [QCD splitting-function](https://arxiv.org/abs/1704.05066) and [jet-substructure](https://arxiv.org/abs/1704.05842) studies. C++/Python · CMSSW · ROOT.

**[AMSfit](https://github.com/xueweiphy/AMSfit)**
- Likelihood fits of cosmic-ray data (AMS-02) to dark-matter and astrophysical models.

---

*Things in progress land here as they ship.*
