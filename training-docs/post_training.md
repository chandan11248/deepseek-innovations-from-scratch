# DeepSeek-V3 Post-Training Setup

> Based on the [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) and [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948)

---

## Post-Training Pipeline Overview

```
DeepSeek-V3-Base (pre-trained on 14.8T tokens)
         │
         ▼
┌──────────────────────────────┐
│  Stage 1: Supervised         │
│  Fine-Tuning (SFT)          │  1.5M instruction examples
│  2 epochs                    │  LR: 5e-6 → 1e-6 (cosine decay)
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Stage 2: Reinforcement      │
│  Learning (GRPO)             │  Rule-based + model-based rewards
│  Multi-domain prompts        │  No critic model needed
└──────────┬───────────────────┘
           │
           ▼
    DeepSeek-V3 (Final)
```

---

## Stage 1: Supervised Fine-Tuning (SFT)

### Dataset

**1.5 million instances** spanning multiple domains, each with tailored data creation methods.

| Domain | Creation Method |
|--------|----------------|
| **Mathematics** | R1-generated + expert-refined |
| **Code (competition problems)** | R1-generated + expert-refined |
| **Logic puzzles** | R1-generated + expert-refined |
| **Creative writing** | DeepSeek-V2.5 generated + human verified |
| **Role-play** | DeepSeek-V2.5 generated + human verified |
| **Simple Q&A** | DeepSeek-V2.5 generated + human verified |

### The R1 Distillation Pipeline (Reasoning Data)

This is one of DeepSeek's most innovative contributions. For reasoning-heavy domains, they don't just use human-written examples — they distill from their own reasoning model (DeepSeek-R1):

```
The Problem with Raw R1 Data:
┌──────────────────────────────────────────────────┐
│  R1 responses have:                              │
│    + High accuracy (strong reasoning)            │
│    - Overthinking (excessively long)             │
│    - Poor formatting                             │
│    - Verbose                                     │
│                                                  │
│  Normal responses have:                          │
│    + Clean formatting                            │
│    + Concise                                     │
│    - Lower accuracy on hard problems             │
└──────────────────────────────────────────────────┘

Goal: Combine R1's accuracy with normal responses' clarity
```

**The 4-step distillation process:**

```
Step 1: Train Domain Expert Model
        ┌─────────────────────────────────────────┐
        │  Base Model                              │
        │     │                                    │
        │     ├── SFT on domain data               │
        │     │                                    │
        │     └── RL on domain tasks               │
        │          │                               │
        │          ▼                               │
        │     Domain Expert Model                  │
        │     (e.g., Math Expert, Code Expert)     │
        └─────────────────────────────────────────┘

Step 2: Create Two Types of SFT Samples
        ┌──────────────────────────────────────────────┐
        │  For each problem:                           │
        │                                              │
        │  Type A: <problem, original_response>        │
        │          (clean, concise format)              │
        │                                              │
        │  Type B: <system_prompt, problem, R1_response>│
        │          (includes reflection + verification)│
        └──────────────────────────────────────────────┘

Step 3: RL Phase (Merging Patterns)
        ┌──────────────────────────────────────────────┐
        │  High-temperature sampling generates         │
        │  responses that BLEND both patterns:         │
        │                                              │
        │    R1 patterns: deep reasoning, verification │
        │    Original patterns: clean format, concise  │
        │                                              │
        │  After ~100s of RL steps:                    │
        │  → Model learns to use R1 reasoning style    │
        │    while maintaining clean output format     │
        └──────────────────────────────────────────────┘

Step 4: Rejection Sampling
        ┌──────────────────────────────────────────────┐
        │  Use the expert model to generate many       │
        │  candidate responses for each problem        │
        │                                              │
        │  Keep only the BEST responses                │
        │  → These become the SFT data for V3          │
        └──────────────────────────────────────────────┘
```

### The System Prompt Design

The system prompt for R1-style data includes explicit instructions for:

- **Reflection:** "Before giving your final answer, verify your reasoning"
- **Verification:** "Check your work by trying an alternative approach"
- **Structured thinking:** "Break the problem into clear steps"

This teaches the model to produce responses enriched with self-checking patterns.

### SFT Training Settings

| Parameter | Value |
|-----------|-------|
| **Epochs** | 2 |
| **Learning rate** | 5 × 10⁻⁶ → 1 × 10⁻⁶ (cosine decay) |
| **Sequence packing** | Multiple samples per sequence |
| **Masking** | Sample masking (samples are isolated and mutually invisible) |
| **Base model** | DeepSeek-V3-Base |

---

## Stage 2: Reinforcement Learning

### Reward Models

DeepSeek uses **two types** of reward models, chosen based on the problem:

#### Rule-Based Reward Model

For questions with **deterministic answers**:

```
Math Problems:
    Prompt: "What is the integral of x^2?"
    Model response: "The answer is \boxed{x^3/3 + C}"
                                     ↑
                              Extract from box, compare with ground truth
                              Reward = 1.0 (correct) or 0.0 (wrong)

Code Problems (LeetCode):
    Model generates code → Compile → Run test cases
    Reward = fraction of test cases passed

Advantages:
    + 100% reliable (no hallucinated rewards)
    + Resistant to reward hacking
    + No model needed
```

#### Model-Based Reward Model

For questions where rule-based checking is impossible:

```
Free-form answers:
    Question: "Explain quantum entanglement"
    Ground truth: [reference answer]
    → Reward model checks if response matches intent

Open-ended (no ground truth):
    Question: "Write a poem about autumn"
    → Reward model evaluates quality directly

Training of the Reward Model:
    ┌──────────────────────────────────────────────┐
    │  Base: DeepSeek-V3 SFT checkpoint            │
    │                                              │
    │  Training data: Preference pairs with        │
    │  chain-of-thought explanations               │
    │                                              │
    │  Not just "response A > B" but also          │
    │  WHY it's better (CoT reasoning to reward)   │
    │                                              │
    │  This mitigates reward hacking               │
    └──────────────────────────────────────────────┘
```

### Group Relative Policy Optimization (GRPO)

GRPO is DeepSeek's RL algorithm — a major simplification over PPO:

```
PPO (Traditional):                       GRPO (DeepSeek):
┌────────────────────┐                  ┌────────────────────┐
│  Policy Model      │                  │  Policy Model      │
│  (671B params)     │                  │  (671B params)     │
├────────────────────┤                  └─────────┬──────────┘
│  Critic/Value      │                            │
│  Model             │  ← ELIMINATED!    For each prompt:
│  (671B params!)    │                     Sample G responses
├────────────────────┤                     Score each with RM
│  Reference Model   │                     Compute RELATIVE
│  (671B params)     │                     advantages within group
├────────────────────┤                  ┌─────────┴──────────┐
│  Reward Model      │                  │  Reference Model   │
└────────────────────┘                  │  (for KL penalty)  │
                                        ├────────────────────┤
Total: 4× model size                   │  Reward Model      │
Memory intensive!                       └────────────────────┘

                                        Total: 3× model size
                                        50% less memory!
```

#### The GRPO Algorithm

```
For each prompt q:

1. SAMPLE: Generate G outputs {o_1, o_2, ..., o_G} from current policy

2. SCORE: Get rewards {r_1, r_2, ..., r_G} from reward model

3. COMPUTE ADVANTAGES (no critic needed!):
   
   A_i = (r_i - mean(r)) / std(r)
   
   This is the key insight: use the GROUP statistics as baseline
   instead of a learned value function
   
4. UPDATE POLICY: Maximize the clipped objective:
   
   L = E[ min(
       ratio_i * A_i,
       clip(ratio_i, 1-eps, 1+eps) * A_i
   ) - beta * KL(policy || reference) ]
   
   where ratio_i = pi_theta(o_i|q) / pi_old(o_i|q)
```

#### Why GRPO works

```
Traditional critic estimates value: V(s) ≈ E[R | state s]
    → Requires a HUGE model (same size as policy!)
    → Training critic is unstable
    → Critic errors compound with policy updates

GRPO replaces this with:
    Baseline = mean reward within the group
    → Zero additional parameters
    → Statistically unbiased estimator
    → No training instability from critic
    → Only needs to sample more responses (compute, not memory)
```

### Multi-Domain RL Training

DeepSeek trains with RL across multiple domains simultaneously:

| Domain | Reward Type | Purpose |
|--------|-------------|---------|
| **Coding** | Rule-based (compiler) | Improve code generation |
| **Math** | Rule-based (answer check) | Improve reasoning |
| **Writing** | Model-based | Improve creativity and helpfulness |
| **Role-play** | Model-based | Improve engagement and consistency |
| **Question answering** | Model-based | Improve accuracy and helpfulness |

This multi-domain approach:
- Aligns the model with human preferences across all use cases
- Fills gaps where SFT data is limited
- Prevents catastrophic forgetting of non-RL domains

---

## The Bigger Picture: How This Connects to Reasoning (DeepSeek-R1)

### The R1-Zero Experiment

Before building the full R1 model, DeepSeek ran a groundbreaking experiment:

```
Experiment: Can reasoning EMERGE from pure RL?

Setup:
    Model: DeepSeek-V3-Base (NO SFT, NO human demonstrations)
    RL: GRPO with rule-based rewards only (math + code)
    
    No examples of chain-of-thought reasoning provided
    No instructions to "think step by step"
    Just: "here's a problem, get the right answer, get a reward"

Result: DeepSeek-R1-Zero
    
    The model SPONTANEOUSLY developed:
    
    1. Chain-of-thought reasoning
       "Let me think about this step by step..."
    
    2. Self-verification
       "Wait, let me check: 37 × 28 = 1036. Verify: 40×28 - 3×28 = 1120 - 84 = 1036 ✓"
    
    3. Reflection and backtracking
       "Hmm, this approach isn't working. Let me try a different method..."
    
    4. Problem decomposition
       "I'll break this into three sub-problems..."
    
    5. Exploration of alternatives
       "There might be a simpler way. What if I use..."
```

**This was a paradigm shift**: reasoning doesn't need to be taught — it *can emerge* from well-designed reinforcement learning.

### From R1-Zero to R1 to V3

```
DeepSeek-V3-Base
    │
    ├── Pure RL (no SFT) ──→ R1-Zero
    │                           │ (emergent reasoning, but poor format)
    │                           │
    │                           ▼
    │                    Cold-Start SFT
    │                    (small set of curated CoT examples
    │                     to fix formatting issues)
    │                           │
    │                           ▼
    │                    Multi-stage RL
    │                    (GRPO with rule-based + model-based rewards)
    │                           │
    │                           ▼
    │                    DeepSeek-R1
    │                    (full reasoning model)
    │                           │
    │                           │ Distillation
    │                           │ (R1 generates training data for V3)
    │                           ▼
    └── SFT (1.5M examples, including R1-distilled data)
                    │
                    ▼
              GRPO RL
                    │
                    ▼
            DeepSeek-V3 (Final)
            (reasoning-capable, clean format)
```

### The `<think>` Tag Mechanism

DeepSeek-R1 uses special tags to separate reasoning from answers:

```
Input:  "Solve: Find all prime numbers p where p^2 + 2 is also prime"

Output:
<think>
I need to find prime numbers p such that p^2 + 2 is also prime.

Let me check small primes:
- p = 2: p^2 + 2 = 6. Not prime.
- p = 3: p^2 + 2 = 11. Prime! ✓
- p = 5: p^2 + 2 = 27 = 3 × 9. Not prime.
- p = 7: p^2 + 2 = 51 = 3 × 17. Not prime.

Wait, let me think about why this pattern occurs...
For any prime p > 3: p is either 1 or 2 (mod 3)
  → p^2 ≡ 1 (mod 3)
  → p^2 + 2 ≡ 0 (mod 3)
  → p^2 + 2 is divisible by 3
  → p^2 + 2 > 3, so it's NOT prime

So p = 3 is the ONLY solution!

Let me verify: 3^2 + 2 = 11. Is 11 prime? Yes ✓
</think>

The only prime number p where p^2 + 2 is also prime is **p = 3**.

For any prime p > 3, we can show that p^2 + 2 is always divisible by 3,
hence not prime. When p = 3, we get 3^2 + 2 = 11, which is prime.
```

The `<think>` content can be:
- **Hidden** from users (just show the clean answer)
- **Shown** for transparency and education
- **Used as training signal** for distillation

---

## Why DeepSeek's Post-Training Matters

| Innovation | Impact |
|-----------|--------|
| **R1 distillation into SFT data** | Best of both worlds: R1 accuracy + clean formatting |
| **GRPO (no critic)** | 50% memory savings in RL, more stable training |
| **Rule-based rewards** | Unhackable rewards for math/code, no model needed |
| **CoT reward model** | Reward model explains its reasoning → harder to hack |
| **Emergent reasoning (R1-Zero)** | Proved reasoning can emerge without explicit teaching |
| **Multi-domain RL** | Single RL run improves all capabilities simultaneously |
| **$5.576M total cost** | 10-20× cheaper than comparable models (GPT-4 class) |

---

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — Sections 5.1, 5.2
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2501.12948)
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- [DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model](https://arxiv.org/abs/2405.04434)
