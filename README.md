# IHDec: Divergence-Steered Contrastive Decoding for Securing Multi-Turn Instruction Hierarchies

<!-- [Paper]() | [ArXiv]() -->

## Abstract

Large Language Models (LLMs) often fail to maintain instruction hierarchies (IH) when processing multi-source inputs with varying role-level priorities, paradoxically adhering to lower-priority directives during conflicts. While existing defenses mitigate this issue, they are largely restricted to single-turn scenarios and require expensive fine-tuning.
In this paper, we formalize this failure mode in multi-turn contexts via a Jensen-Shannon Divergence (JSD) framework, uncovering a pervasive role-influence inversion phenomenon where subordinate inputs override superior roles.
To rectify this without training, we propose **IHDec** (**I**nstruction **H**ierarchy-steered **Dec**oding). IHDec leverages JSD to automatically detect token-level hierarchy violations and dynamically executes contrastive decoding to suppress misaligned subordinate roles.
Extensive evaluations demonstrate that IHDec outperforms training-based baselines in multi-turn conflicts while fully preserving general response quality. Furthermore, IHDec strengthens safety against adversarial prompt injections and exhibits a robust scaling synergy with larger models.

## Overview

**IHDec** is a training-free decoding method that enforces instruction hierarchy in large language models at inference time. Without any fine-tuning, IHDec detects when a model is about to violate the intended priority order among instruction types (system > user > assistant) and applies a corrective logit bias to steer generation back toward hierarchy-compliant outputs.

### How It Works

At each generation step, IHDec computes the **JSD-based influence** of each instruction segment (system, user, assistant) by running batched forward passes with attention column-blocking ablations:

```
influence(role) = JSD( P_full || P_{no_role} )
```

When a hierarchy violation is detected (e.g., `influence(user) > influence(system)`), IHDec computes a correction delta and applies it to the next-token logits:

```
logit_final = logit_full + β × decay^t × Δ_final
```

The method requires no gradient updates and is compatible with any autoregressive LLM (Llama 3.1, Qwen3, etc.).


> **Note:** Model checkpoints and evaluation datasets are **not included** in this repository due to file size. You must provide them yourself (see setup instructions below).

## Setup

### 1. Clone and install

```bash
git clone <this-repo>
cd IHDec/torchtune

conda create -n IHDec python=3.10
conda activate IHDec
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 2. Download a model checkpoint

IHDec supports any Llama 3.1 or Qwen3 checkpoint. For Llama 3.1 8B Instruct:

```bash
tune download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output-dir ./pretrained_models/llama3_1_8B_Instruct \
    --hf-token <YOUR_HF_TOKEN>
```

For Qwen3 models, use `--model-type qwen` and `--device auto` flags when running `ihdec.py` directly.

### 3. Prepare evaluation data

#### IHEval

Download [IHEval](https://github.com/microsoft/IHEval) and place the desired JSON files under `torchtune/data/iheval/`. IHEval contains multiple domains with various subsets; we evaluate on the **rule-following** and **safety** domains. See the IHEval repository for the full list of available datasets.

#### MT-Bench 101

Download [MT-Bench 101](https://huggingface.co/datasets/mtbench101/mt-bench-101) and convert it using the provided script:

```bash
cd torchtune/recipes
python convert_mtbench101.py --src /path/to/mtbench101.jsonl
```

This generates `torchtune/data/mt_bench_101.json`.

## Running Evaluations

### IHEval

Run IHDec on a single IHEval dataset from `torchtune/`:

```bash
# Usage: ./run_iheval.sh <gpu_id> <dataset_name> [beta] [decay]
# Rule-following
./run_iheval.sh 0 both_conflict_default_request 1300 0.97
./run_iheval.sh 1 first_conflict_default_request 1300 0.97
# Safety
./run_iheval.sh 2 hijack_weak 1300 0.97
./run_iheval.sh 3 extract_strong 1300 0.97
```

Results are saved to `torchtune/output/iheval_ihdec/`.

#### Scoring IHEval responses

After generating responses, score them using `eval_iheval.py`. This requires cloning the [IHEval](https://github.com/microsoft/IHEval) repository:

```bash
git clone https://github.com/microsoft/IHEval torchtune/IHEval
```

Then run from `torchtune/recipes/`:

```bash
# Rule-following
python eval_iheval.py \
    --input    ../output/iheval_ihdec/both_conflict_default_request_beta1300_decay0.97.json \
    --dataset  both_conflict_default_request

# Safety
python eval_iheval.py \
    --input    ../output/iheval_ihdec/hijack_weak_beta1300_decay0.97.json \
    --dataset  hijack_weak
```

Supported `--dataset` values:
- Rule-following: `both_conflict_default_request`, `first_conflict_default_request`, `single_conflict_request`
- Safety: `hijack_strong`, `hijack_weak`, `extract_strong`, `extract_weak`

### MT-Bench 101

Run from `torchtune/`:

```bash
# Usage: ./run_mtbench101.sh <gpu_id> [beta] [decay]
./run_mtbench101.sh 0 1300 0.97
```

Results are saved to `torchtune/output/mtbench101_ihdec/`.

Key hyperparameters:
| Argument | Default | Description |
|---|---|---|
| `--beta` | `1300` | Correction strength |
| `--beta-decay` | `0.97` | Per-step decay of correction |
| `--max-tokens` | `1024` | Max new tokens to generate |

### Running `ihdec.py` directly

```bash
cd torchtune/recipes
CUDA_VISIBLE_DEVICES=0 python ihdec.py \
    --input-file  ../data/mt_bench_101.json \
    --output-file ../output/my_run.json \
    --beta        1300 \
    --beta-decay  0.97 \
    --max-tokens  1024
```

For Qwen models:

```bash
CUDA_VISIBLE_DEVICES=0 python ihdec.py \
    --model-type qwen \
    --model-path ../pretrained_models/qwen3_8b \
    --device auto \
    --input-file  ../data/mt_bench_101.json \
    --output-file ../output/qwen_run.json \
    --beta        1300 \
    --beta-decay  0.97
```

## Citation

```bibtex
@article{liu2025ihdec,
  title   = {IHDec: Divergence-Steered Contrastive Decoding for Securing Multi-Turn Instruction Hierarchies},
  author  = {Liu, Nicole Geumheon and Jang, Haeun and Jun, Yonghyun and Lee, Hwanhee},
  year    = {2026},
}
```

## Acknowledgements

This codebase builds on [ISE](https://github.com/tongwu2020/ISE) (Wu et al., ICLR 2025), which is itself built on [torchtune](https://github.com/pytorch/torchtune). Both are licensed under the BSD 3-Clause License; the torchtune license is retained in `torchtune/LICENSE`.

## License

IHDec is released under the [MIT License](LICENSE). Code from torchtune is licensed under the BSD 3-Clause License; see [`torchtune/LICENSE`](torchtune/LICENSE) for details.
