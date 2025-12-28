---

# 🧠 Small Model, Big Thoughts: Reasoning with Gemma 2B & Tunix

Automated reasoning and "Chain of Thought" generation using small language models (SLMs). This project implements **Group Relative Policy Optimization (GRPO)** via Google Tunix and a novel **Consensus Voting** strategy to achieve 70% accuracy on math benchmarks.

---

## 🎯 Overview

### Problem Statement

Most small open-weight models (like Gemma 2B) struggle with complex multi-step reasoning. They typically guess answers rather than deriving them, and training them via standard RLHF is computationally expensive and unstable on limited hardware like Kaggle TPUs.

### Objectives

1. **Fine-tune Gemma 2B** using Tunix and GRPO (Group Relative Policy Optimization) on TPUs.
2. **Engineer Reward Functions** that enforce strict "Show Your Work" formats (`<reasoning>` + `<answer>`).
3. **Implement Consensus Voting** to aggregate multiple inference traces and eliminate hallucinations.
4. **Beat the Baseline** significantly on the GSM8K math benchmark.

### Why This Matters

* 📉 **Efficiency**: High-level reasoning on low-resource hardware (2B parameters).
* 🧩 **Transparency**: Models that explain their logic are safer and more trustworthy.
* ⚡ **TPU Optimization**: overcoming 9-hour runtime limits via custom relay pipelines.

---

## 📊 Dataset

**Source**: [GSM8K (Grade School Math 8K)](https://github.com/openai/grade-school-math)

### Dataset Composition:

* **Task**: Multi-step mathematical reasoning problems.
* **Format**: Text questions requiring 2-8 steps of calculation.
* **Size**: ~8.5k high-quality linguistically diverse grade school math word problems.
* **Tokens**: Optimized for context lengths up to 1024 tokens.

### Data Split:

* Training: 7,473 samples
* Test: 1,319 samples

---

## 🔬 Methodology

### Approach 1: GRPO Training (The Foundation)

**Model**: Gemma-2-2B-IT (Instruction Tuned)

**Pipeline**:

1. Load Gemma 2B on Kaggle TPU v5e-8.
2. **Tunix Implementation**: Use `GRPOLearner` to generate groups of completions.
3. **Reward Function**:
* **Format Reward**: +2.0 for using `<reasoning>` and `<answer>` tags.
* **Correctness Reward**: +25.0 for exact numerical match.
* **Format Penalty**: -15.0 for broken XML tags.


4. **Optimization**: KL-divergence penalty (Beta = 0.04) to prevent mode collapse.

**Advantages**:

* ✅ Directly optimizes for the final answer.
* ✅ Enforces structured output.

**Limitations**:

* ❌ Single-pass generation still hallucinates (58% accuracy ceiling).
* ❌ Sensitive to hyperparameter drift.

---

### Approach 2: Consensus Voting (The "Winner")

**Technique**: Inference-Time Compute / Self-Consistency

**Pipeline**:

1. Take the GRPO-finetuned model.
2. **Generate 5 Traces** for every question at varying temperatures:
* `T=0.2` (Strict/Focused)
* `T=0.4` to `T=0.8` (Balanced)
* `T=1.0` (Creative/Exploratory)


3. **Extract Answers** using Regex.
4. **Majority Vote**: The most common answer is selected as the final prediction.

**Why This Works**:

* **Error Randomness**: When the model hallucinates, it usually produces *different* wrong answers.
* **Truth Consistency**: When the model reasons correctly, it produces the *same* right answer.
* **Boost**: This filters out the noise without extra training.

---

## 📈 Results

### Quantitative Comparison

| Metric | Zero-Shot Baseline | Tunix Finetuned (Single) | **Consensus Voting (Ours)** |
| --- | --- | --- | --- |
| **Accuracy (GSM8K)** | ~30.1% | 58.0% | **69.9%** |
| **Format Compliance** | < 10% | 99.5% | **100%** |
| **Inference Time** | Fast | Fast | 5x Slower |

### Training Progress

### Key Observations

**Tunix Finetuned**:

* The model learned the XML format within 50 steps.
* Accuracy plateaued around step 600.
* "Show your work" traces became coherent and mathematically sound.

**Consensus Voting**:

* **+11.9% Accuracy Boost** just by voting.
* Corrected ~20% of "silly mistakes" where the model did the math wrong in the final step.
* Proves that 2B models have latent knowledge they struggle to access in a single try.

---

## 🛠️ Installation

### Prerequisites

* Python 3.10+
* JAX/Flax (TPU support recommended)
* Kaggle Notebook Environment (for TPU access)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/rohan2700/tunix-gemma-reasoning.git
cd tunix-gemma-reasoning

```

2. **Install Tunix & Dependencies**

```bash
pip install tunix git+https://github.com/google-deepmind/gemma.git
pip install -r requirements.txt

```

3. **Download Model Weights**

* Authenticate with Kaggle/HuggingFace to access `gemma-2-2b-it`.

---

## 🚀 Usage

### 1. Training (Multi-Session Relay)

Open `notebooks/01_train_session.ipynb`. Our custom pipeline handles the 9-hour TPU limit:

```python
from tunix.algo import GRPOLearner

# Initialize Learner
learner = GRPOLearner(
    model_path="google/gemma-2-2b-it",
    reward_funcs=[format_reward, correctness_reward],
    beta=0.04
)

# Train (Automatically saves checkpoints for Session 2)
learner.fit(dataset="gsm8k", steps=1000)

```

### 2. Inference & Consensus

```python
# Consensus Voting Logic
def generate_consensus(question, num_votes=5):
    answers = []
    # Sample at different temperatures
    for temp in [0.2, 0.4, 0.6, 0.8, 1.0]:
        trace = model.generate(question, temperature=temp)
        ans = extract_answer(trace)
        answers.append(ans)
    
    # Return most common answer
    return max(set(answers), key=answers.count)

```

---

## 🔍 Key Findings

### 1. The "Beta" Sweet Spot

* `Beta = 0.1`: Too strict, model learned nothing.
* `Beta = 0.0`: Model collapsed into gibberish.
* **`Beta = 0.04`**: Perfect balance. Allowed creativity while staying on topic.

### 2. Format is Everything

* Without strict `<reasoning>` tags, the model would output the answer immediately.
* We had to penalize "short answers" to force the model to think.

### 3. Inference-Time Compute

* Trading time for accuracy works.
* Running the model 5 times is cheaper than training a model 5x larger.

### 4. Trade-offs

| Aspect | Single Pass | Consensus Voting |
| --- | --- | --- |
| **Accuracy** | Good (58%) | Excellent (70%) |
| **Latency** | ~2s / query | ~10s / query |
| **Cost** | Low | Moderate |
| **Use Case** | Real-time Chat | Offline Analysis |

---

## 🔮 Future Work

1. **Hard Negative Mining**
* Train on examples where the model gets the *right* answer for the *wrong* reasons.


2. **Process Supervision (PRM)**
* Reward the model for each correct *step* of the math, not just the final answer.


---

## 📚 References

1. **Google Tunix**: JAX-native library for LLM post-training.
2. **DeepSeekMath**: "Group Relative Policy Optimization" (GRPO) paper.
3. **Wei et al.**: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022).
4. **Wang et al.**: "Self-Consistency Improves Chain of Thought Reasoning" (2022).

---

## 👤 Author

**Rohan Sanjay Patil**

* **Competition**: Google Tunix Hackathon 2025
* **Role**: MSc AI @ THWS / Researcher
* **Focus**: RAG, Computer vision, LLMs

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact

For questions or collaboration:

* Email: [rohansanjaypatilrsp18@gmail.com]
* GitHub: [@rohan-patil-ai]

---

**⭐ If you found this project helpful, please consider giving it a star!**
