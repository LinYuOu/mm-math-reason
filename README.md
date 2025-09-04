# MM Math Reasoning Project

[![🤗 Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-huggingface-blue)](https://huggingface.co/datasets/TencentBAC/TBAC-VLR1-7B-SFT-DATA/tree/main)
[![🤗 Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-huggingface-blue)](https://huggingface.co/collections/TencentBAC/vlr1-6892c119672e8d0780e1d288)
[**论文**](https://arxiv.org/pdf/2509.03321) (了解更多技术细节)

## Project Overview
This project focuses on advancing mathematical reasoning capabilities in multimodal AI models. Built upon [ms-swift](https://github.com/modelscope/ms-swift.git), it provides:
- Evaluation pipelines for math verification and answer matching
- Training scripts for supervised fine-tuning (SFT) and reinforcement learning
- Tools for processing JSONL datasets and generating model responses

## Performance
| Model                              | **Average** | **MathVista** | **MathVision** | **MathVerse** | **DynaMath** | **LogicVista** |
| :--------------------------------: | :---------: | :-----------: | :------------: | :-----------: | :----------: | :------------: |
| Qwen2.5-VL-7B                      | 40.5        | 68.0          | 25.7           | 45.5          | 21.8         | 41.2           |
| VLAA-Thinker-Qwen2.5-7B            | 42.7        | 68.0          | 26.4           | 48.2          | 22.4         | 48.5           |
| VL-Rethinker-7B                    | 41.8        | 73.7          | 28.4           | 46.4          | 17.8         | 42.7           |
| TBAC-VLR1-7B-RL                    | 41.3        | 70.1          | 25.4           | 43.4          | 19.0         | 48.4           |
| TBAC-VLR1-7B-SFT                   | 41.8        | 65.1          | 28.5           | 49.1          | 20.6         | 45.5           |
| TBAC-VLR1-7B                       | **43.4**    | 66.7          | **31.4**       | **50.1**      | **22.6**     | 46.4           |

## Installation
### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/modelscope/mm_math_reason.git
   cd mm_math_reason
   ```
2. Install dependencies:
   ```bash
   source scripts/install.sh
   ```
For a complete list of required packages, see `requirements.txt` in the project root directory.

## Dataset
Download datasets from [HuggingFace](https://huggingface.co/datasets/TencentBAC/TBAC-VLR1-7B-SFT-DATA/tree/main):

### File List
- `SFT.jsonl`: Supervised Fine-Tuning dataset
- `MM_DATA_RL.jsonl`: Reinforcement Learning dataset (multimodal, contains both images and text)
- `TEXT_DATA_RL.jsonl`: Reinforcement Learning dataset (text-only)
- `benchmarks.tar.gz`: Benchmark evaluation datasets
- `dataset_images.zip`: Image assets for multimodal datasets

### Setup Instructions
1. Extract image assets:
   ```bash
   unzip dataset_images.zip -d dataset_images
   ```
2. Extract benchmark datasets:
   ```bash
   tar -xzf benchmarks.tar.gz -C eval/Benchmarks
   ```

## Usage
### Evaluation
1. Generate model responses:
   ```bash
   bash eval/generate_res.sh
   ```
2. Evaluate a JSONL file with math_verify:
   ```bash
   bash eval.sh
   ```

### Training
```bash
# Supervised Fine-Tuning
bash scripts/train_sft.sh

# Reinforcement Learning
bash scripts/train_rl.sh
```

## Project Structure
- `eval/`: Evaluation scripts and math verification tools
  - `eval.sh`: Main evaluation pipeline
  - `eval_from_jsonl_with_mathverify.py`: JSONL processor with math verification
- `scripts/`: Training configurations
  - SFT and RL training scripts
  - Hyperparameter configurations

## License
MIT License
