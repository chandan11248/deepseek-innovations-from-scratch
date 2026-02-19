# DeepSeek-V3 Pre-Training Setup

> Based on the [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

---

## Training Infrastructure

| Component | Specification |
|-----------|---------------|
| **GPUs** | 2,048 NVIDIA H800 |
| **Interconnect (intra-node)** | NVLink + NVSwitch (8 GPUs per node) |
| **Interconnect (inter-node)** | InfiniBand (IB) |
| **Precision** | FP8 mixed-precision (most GEMMs in FP8, critical ops in BF16/FP32) |
| **Total Cost** | $5.576M (~2.788M H800 GPU hours) |
| **Training Stability** | Zero irrecoverable loss spikes, zero rollbacks |

---

## Data Construction

### Corpus

| Aspect | Detail |
|--------|--------|
| **Total tokens** | 14.8 trillion |
| **Tokenizer** | Byte-level BPE, 128K vocabulary (extended from DeepSeek-V2) |
| **Languages** | Multilingual (expanded beyond English and Chinese) |
| **Data mix** | Enhanced ratio of mathematical and programming samples |
| **Processing** | Refined pipeline to minimize redundancy while maintaining diversity |
| **Packing** | Document packing for data integrity (no cross-sample attention masking) |

### Fill-in-Middle (FIM) Strategy

DeepSeek-V3 uses Fill-in-Middle during pre-training at a **10% rate** using the Prefix-Suffix-Middle (PSM) framework:

```
Structure:
    <prefix>  [beginning of code/text]
    <suffix>  [end of code/text]
    <middle>  [model must predict this part]
```

This enables the model to predict middle text based on surrounding context without compromising standard next-token prediction ability.

### Token Boundary Bias Fix

The new tokenizer combines punctuation with line breaks into single tokens. This can cause issues with few-shot prompts that lack trailing newlines. Fix: randomly split a proportion of combined tokens during training to expose the model to edge cases.

---

## Model Hyperparameters

```
┌──────────────────────────────────────────────────────────────────────┐
│                   DeepSeek-V3 Architecture                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Transformer layers:        61                                       │
│  Hidden dimension:          7,168                                    │
│  Attention heads (n_h):     128                                      │
│  Per-head dimension (d_h):  128                                      │
│                                                                      │
│  MLA Parameters:                                                     │
│    KV compression dim (d_c):     512                                 │
│    Query compression dim (d_c'): 1,536                               │
│    RoPE head dim (d_h^R):        64                                  │
│                                                                      │
│  MoE (layers 4-61):                                                  │
│    Shared experts:      1                                            │
│    Routed experts:      256                                          │
│    Active per token:    8                                            │
│    Expert hidden dim:   2,048                                        │
│    Max routing nodes:   4  (node-limited routing)                    │
│                                                                      │
│  Dense FFN (layers 1-3):                                             │
│    Standard SwiGLU FFN                                               │
│                                                                      │
│  Multi-Token Prediction:                                             │
│    Depth D = 1  (predict 1 additional token)                         │
│                                                                      │
│  Weight initialisation:    N(0, 0.006)                               │
│                                                                      │
│  Total parameters:    671B                                           │
│  Active per token:    37B                                            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Training Hyperparameters

### Optimizer

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW |
| **beta_1** | 0.9 |
| **beta_2** | 0.95 |
| **Weight decay** | 0.1 |
| **Gradient clip norm** | 1.0 |

### Learning Rate Schedule

The learning rate follows a **warmup → constant → cosine decay → two-stage cooldown** schedule:

```
Learning Rate Schedule:
                                                                    
  2.2e-4 │          ┌──────────────────────────────┐                 
         │         /│         constant              │╲                
         │        / │                               │  ╲              
         │       /  │                               │    ╲  cosine    
         │      /   │                               │     ╲  decay    
         │     /    │                               │       ╲         
  2.2e-5 │    /     │                               │         ╲───────┐
  7.3e-6 │   /      │                               │                 ├──┐
     0   │──/       │                               │                 │  │
         └──┬───────┴───────────────────────────────┴─────────────────┴──┘
           2K      ........10T tokens...........  14.3T     14.63T  14.8T
           steps                                            │       │
                                                     constant│       │constant
                                                     2.2e-5  │       │7.3e-6
                                                     (333B)  │       │(167B)
```

**Stages:**
1. **Warmup** (0 → 2K steps): Linear increase 0 → 2.2 × 10⁻⁴
2. **Constant** (~2K steps → 10T tokens): Hold at 2.2 × 10⁻⁴
3. **Cosine decay** (10T → 14.3T tokens): Decay 2.2 × 10⁻⁴ → 2.2 × 10⁻⁵
4. **Cooldown phase 1** (14.3T → 14.63T): Constant at 2.2 × 10⁻⁵ (333B tokens)
5. **Cooldown phase 2** (14.63T → 14.8T): Constant at 7.3 × 10⁻⁶ (167B tokens)

### Batch Size Schedule

| Training Phase | Batch Size | Sequence Length |
|---------------|------------|-----------------|
| First 469B tokens | 3,072 → 15,360 (gradual ramp) | 4,096 |
| Remaining tokens | 15,360 | 4,096 |

### MoE Load Balancing

| Parameter | Value (first 14.3T) | Value (final 500B) |
|-----------|---------------------|---------------------|
| Bias update speed (gamma) | 0.001 | 0.0 (frozen) |
| Balance loss weight (alpha) | 0.0001 | 0.0001 |

### Multi-Token Prediction (MTP)

| Phase | MTP Loss Weight (lambda) |
|-------|--------------------------|
| First 10T tokens | 0.3 |
| Remaining 4.8T tokens | 0.1 |

---

## FP8 Mixed-Precision Training

DeepSeek-V3 pioneers FP8 training at scale, doubling theoretical compute speed:

### What runs in FP8

| Operation | Precision |
|-----------|-----------|
| Forward pass GEMMs (Fprop) | FP8 |
| Activation backward (Dgrad) | FP8 |
| Weight backward (Wgrad) | FP8 |
| Activation caching for backward | FP8 |
| MoE expert dispatch/combine | FP8 |

### What stays in higher precision

| Component | Precision | Reason |
|-----------|-----------|--------|
| Embedding module | BF16 | Sensitivity to quantisation |
| Output head (LM head) | BF16 | Final logit accuracy |
| MoE gating modules | BF16 | Router decision accuracy |
| Normalization (RMSNorm) | BF16 | Small numerical range |
| Attention operators | BF16 | Softmax precision |
| Master weights | FP32 | Gradient accumulation accuracy |
| Weight gradients | FP32 | Optimiser stability |
| Optimiser states | BF16 | Memory efficiency |

### Fine-Grained Quantisation

To handle outliers, DeepSeek uses tile/block-wise quantisation instead of per-tensor:

```
Per-tensor quantisation (standard):
    Entire tensor → 1 scale factor → large quantisation error from outliers

Tile-wise quantisation (DeepSeek):
    Tensor split into 1×Nc tiles → each tile gets its own scale factor
    → outliers only affect their local tile, not the whole tensor

Block-wise quantisation:
    Tensor split into Nc×Nc blocks → even finer granularity
```

This keeps the relative training loss error below **0.25%** compared to BF16 baseline.

---

## Long Context Extension

After the main pre-training, DeepSeek-V3 extends context from 4K to 128K using **YaRN** (Yet another RoPE extensioN) in two phases:

| Phase | Context Length | Batch Size | Steps | Learning Rate |
|-------|---------------|------------|-------|---------------|
| 1 | 32K | 1,920 | 1,000 | 7.3 × 10⁻⁶ |
| 2 | 128K | 480 | 1,000 | 7.3 × 10⁻⁶ |

**YaRN Hyperparameters:**
- Scale s = 40
- alpha = 1, beta = 32
- Scaling factor: sqrt(t) = 0.1 × ln(s) + 1
- Applied only to the decoupled shared key (k^R)

---

## Training Timeline Summary

```
Phase 1: Pre-training                      14.8T tokens
    │                                      ~2 months
    │   ├── Warmup                          2K steps
    │   ├── Constant LR training           ~10T tokens
    │   ├── Cosine decay                   ~4.3T tokens
    │   └── Two-stage cooldown              500B tokens
    │
Phase 2: Long Context Extension            2K steps
    │   ├── 4K → 32K context               1K steps
    │   └── 32K → 128K context             1K steps
    │
Phase 3: Post-Training (see post_training.md)
    │   ├── Supervised Fine-Tuning
    │   └── Reinforcement Learning (GRPO)
    ▼
DeepSeek-V3 (Final)
```

---

## Key Design Decisions

### Why FP8 instead of BF16?

- **2× theoretical speedup** on H800 GPUs
- **40% memory reduction** for activations (cached in FP8)
- Training loss error <0.25% vs BF16 baseline — negligible quality impact
- Enabled the $5.576M training budget (vs estimated $50M+ in BF16)

### Why auxiliary-loss-free load balancing?

Traditional MoE uses an auxiliary loss to balance expert utilization, but this **hurts model quality** by adding a competing training objective. DeepSeek instead uses a bias term updated outside gradient descent:

```
Traditional:  L_total = L_main + alpha * L_balance  (hurts L_main!)

DeepSeek:     L_total = L_main  (bias adjusted separately, no interference)
              bias += gamma * tanh(violation)  (outside gradient)
```

### Why cooldown the MTP loss?

The MTP loss weight reduces from 0.3 to 0.1 in the last 4.8T tokens. This lets the model focus more on the primary next-token prediction objective in the final training stages, while still benefiting from the richer representations MTP training provides.

---

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [AdamW: Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071)
- [FIM: Fill-in-the-Middle Training](https://arxiv.org/abs/2207.14255)
