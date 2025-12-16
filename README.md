# 🧸 LlamaPlushie & Child Safety Benchmark (CSB)

This repository contains the official implementation for **LlamaPlushie**, a family of open-source Large Language Models (LLMs) aligned for **safe, developmentally appropriate, and empathetic interactions with children**.

It also includes the **Child Safety Benchmark (CSB)** — a novel evaluation suite designed to stress-test models using *child-like cues* and linguistic markers of vulnerability.

---

## 📂 Repository Structure

```text
.
├── cfg/                       # Configuration files for SFT, DPO, and KTO fine-tuning
├── datasets/                  # Storage for CSB, classifier training data, and eval sets
├── evaluation/                # Output JSONs containing GPT-5.1 judge scores
├── gold-responses/            # "Gold" standard responses (GPT-5.1, Claude 4.5, Gemini 3)
├── prompts/                   # Jinja2 templates for data generation and LLM judging
├── responses/                 # Inference outputs from baseline and fine-tuned models
├── generate_data.py           # Script to generate the synthetic CSB dataset
├── generate_gold_responses.py # Pipeline to generate and iteratively refine gold responses
├── generate_responses.py      # Main inference script (vLLM + DeepInfra supported)
├── judge_responses.py         # LLM-as-a-Judge evaluation pipeline
├── train_classifier.py        # Training script for the BERT-based safety classifier
├── train.sh                   # Shell script to launch classifier training
└── utils.py                   # Helper functions, model mappings, and data loading
```
---

## 🛠️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sameearif/LlamaPlushie.git
cd LlamaPlushie
```

### 2. Install Dependencies

This project relies on **PyTorch**, **Hugging Face Transformers**, and **vLLM** for local inference.

```bash
pip install torch transformers vllm openai jinja2 tqdm evaluate datasets scikit-learn accelerate
```

### 3. Environment Variables

Set API keys for the models you intend to use.

```bash
# For generation & judging
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AIza..."

# For API-based inference (optional)
export DEEPINFRA_API_KEY="..."

# For Hugging Face Hub (model uploads)
export HF_TOKEN="hf_..."
```

---

## 🚀 Usage Pipeline

### 1. Dataset Generation (CSB)

Generate the **Child Safety Benchmark** prompts.
This script uses **GPT-5.1** to create diverse scenarios across **7 child-safety categories**
(e.g., Medical, Sexual, Harm, Morals).

```bash
python generate_data.py
```

**Output:**
`datasets/CSB.json`

---

### 2. Gold Response Generation

Generate high-quality reference responses that serve as alignment targets.

This pipeline uses **iterative refinement**:

1. Generate a response
2. Critique it using a *Reviewer* prompt
3. Regenerate if quality is below threshold

```bash
python generate_gold_responses.py \
    --model gpt-5.1 \
    --dataset_name CSB.json
```

**Supported models**

* `gpt-5.1`
* `claude-sonnet-4-5`
* `gemini-3-pro-preview`

---

### 3. Model Inference (Generating Responses)

Run inference on baseline models or LlamaPlushie adapters.

#### Option A: Local Inference (vLLM)

Recommended for **LlamaPlushie** models.
Automatically loads LoRA adapters via `LORA_BASE_MAP`.

```bash
python generate_responses.py \
    --model sameearif/LlamaPlushie-3.1-8B-SFT \
    --dataset_name CSB_FT_Eval.json \
    --local True
```

#### Option B: API Inference

Use **DeepInfra** or OpenAI-compatible endpoints for large baseline models.

```bash
python generate_responses.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset_name CSB.json
```

**Optional Flags**

* `--add_age True`
  Randomly appends explicit ages (6–13) to prompts to test age-aware behavior.

---

### 4. Automated Evaluation (LLM-as-a-Judge)

Evaluate safety, empathy, and developmental appropriateness using **GPT-5.1** as a judge.
The judging rubric is defined in `prompts/llm-judge.jinja`.

```bash
python judge_responses.py \
    --model sameearif/LlamaPlushie-3.1-8B-SFT
```

**Output:**
`evaluation/LlamaPlushie-3-8B-SFT.json`

---

### 5. Train Safety Classifier

Train a **BERT / RoBERTa-based classifier** to detect unsafe content in child-LLM interactions.

The script handles:

* Text cleaning
* Tokenization
* Metrics (Accuracy, F1)

#### Run via Shell Script

```bash
sh train.sh
```

#### Run via Python

```bash
python train_classifier.py \
    --model_name "google-bert/bert-large-uncased" \
    --dataset_name "sameearif/CSB-Classifier" \
    --output_dir "./results/bert-csb-v1" \
    --epochs 5 \
    --batch_size 16 \
    --gradient_accumulation_steps 1 \
    --push_to_hub \
    --hub_model_id "sameearif/bert-large-csb"
```

---

## 📊 Supported Models

Model mappings are defined in `utils.py`.

| Type             | Models                                                                    |
| ---------------- | ------------------------------------------------------------------------- |
| **LlamaPlushie** | LlamaPlushie-3.1-8B-SFT, LlamaPlushie-3.1-8B-KTO, LlamaPlushie-3.1-8B-DPO |
| **Meta**         | Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct-Turbo                       |
| **Google**       | Gemma-3-12B-IT, Gemma-3-27B-IT, Gemini-3-Pro                              |
| **DeepSeek**     | DeepSeek-V3.1, DeepSeek-R1                                                |
| **Qwen**         | Qwen3-14B, Qwen3-32B                                                      |
| **Proprietary**  | GPT-5.1, Claude-Sonnet-4.5, GPT-OSS-120B                                  |

---

## 📌 Notes

* LlamaPlushie is designed for **research use** in child safety, alignment, and evaluation.
* CSB focuses on *how children actually speak*, not just policy-violating content.
* The benchmark emphasizes **empathetic redirection**, **age-appropriate explanations**, and **harm minimization**, rather than over-refusal.
