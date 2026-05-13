# Phase-LLM: Multi-Agent Generative Reasoning for Phase Prediction in Multicomponent Alloys

---

## Overview

Phase-LLM is a multi-agent generative reasoning framework for phase prediction in multicomponent alloys (e.g., MPESAs).  
The project follows a practical 3-stage pipeline:

1. **Data construction** with multi-agent Chain-of-Thought generation.
2. **SFT training** with LoRA and merged model export.
3. **RL refinement** (GRPO) plus final inference/evaluation.

Target labels:

- `Only FCC and L12 phases form`
- `Other phases form in addition to FCC and L12`
- `L12 phase does not form`

---

## Repository Structure (Current)

```text
Phase-LLM-Open-Source/
├── 0_Original_Data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── 1_Data_Construction/
│   ├── prompts/
│   ├── build_cot_multi_agent.py
│   ├── build_cot_fewshot.py
│   ├── build_cot_zeroshot.py
│   ├── convert_to_training_format.py
│   ├── split_sft_rl.py
│   ├── call_llm.py
│   └── output_data/
├── 2_Training_SFT/
│   ├── sft_config.yaml
│   ├── merge_lora.yaml
│   ├── run_sft.sh
│   ├── environment_sft.yml
│   ├── requirements_sft.txt
│   ├── data/
│   └── saves/
├── 3_Training_RL/
│   ├── grpo_config.yaml
│   ├── prepare_rl_data.py
│   ├── run_grpo.sh
│   ├── open_r1/
│   ├── accelerate_configs/
│   ├── environment_rl.yml
│   ├── requirements_rl.txt
│   ├── data/
│   └── output/
├── 4_Inference_and_Evaluation/
│   ├── run_inference.py
│   ├── eval_accuracy.py
│   └── output_data/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Environment Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for SFT/RL/inference)
- Conda (recommended for isolated SFT/RL environments)

### Install

```bash
git clone https://github.com/ioioiioo12138/Phase-LLM.git
cd Phase-LLM

pip install -r requirements.txt
conda env create -f 2_Training_SFT/environment_sft.yml
conda env create -f 3_Training_RL/environment_rl.yml
```

### API variables (for Stage 1 data construction)

```bash
export TEACHER_API_KEY="your-api-key"
export TEACHER_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export TEACHER_MODEL="qwen-plus"
```

---

## Workflow

### 1) Data Construction

```bash
cd 1_Data_Construction

# Main multi-agent data generation
python build_cot_multi_agent.py

# Optional variants
python build_cot_zeroshot.py
python build_cot_fewshot.py

# Convert and split for SFT / RL
python convert_to_training_format.py
python split_sft_rl.py
```

Prompt templates are in `1_Data_Construction/prompts/`.

### 2) SFT Training (LoRA)

```bash
cd 2_Training_SFT
bash run_sft.sh

# Merge LoRA into a standalone model
llamafactory-cli export merge_lora.yaml
```

Key files:

- `sft_config.yaml`: SFT training config
- `merge_lora.yaml`: merge/export config
- `saves/`: checkpoints and merged model outputs

### 3) RL Training (GRPO)

```bash
cd 3_Training_RL
python prepare_rl_data.py
bash run_grpo.sh
```

Key files:

- `grpo_config.yaml`: GRPO config
- `open_r1/`: RL codebase integration
- `output/`: RL checkpoints and logs

### 4) Inference and Evaluation

```bash
cd 4_Inference_and_Evaluation
python run_inference.py
python eval_accuracy.py
```

Default outputs are written under `4_Inference_and_Evaluation/output_data/`.

---

## Practical Notes

- Keep SFT and RL environments separated (`environment_sft.yml` vs `environment_rl.yml`).
- If you compare models, save predictions to different output files to avoid overwrite confusion.
- For reproducible comparisons, fix model path, prompt, input set, and decoding parameters.
```

