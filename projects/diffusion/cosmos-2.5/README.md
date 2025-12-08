# World Simulation with Video Foundation Models for Physical AI

## ✨ Introduction

NVIDIA Cosmos™ is a platform purpose-built for physical AI, featuring state-of-the-art generative world foundation models (WFMs), 
robust guardrails, and an accelerated data processing and curation pipeline. Designed specifically for real-world systems, 
Cosmos enables developers to rapidly advance physical AI applications such as autonomous vehicles (AVs), robots, and video analytics AI agents.

This project refactors the training and inference code for Cosmos-Predict2.5 and Cosmos-Transfer2.5, adopting the [diffusers](https://github.com/huggingface/diffusers) code style. 
This makes the code more concise and readable, and easier to use.

The original projects are:

- **Cosmos-Predict2.5**: [https://github.com/nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5)
- **Cosmos-Transfer2.5**: [https://github.com/nvidia-cosmos/cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5)

## ⚡ Installation

We recommend using a fresh conda environment for installation.

```bash
conda create -n cosmos2_5 python=3.11.10 -y
conda activate cosmos2_5

pip3 install giga-train
pip3 install giga-datasets

git clone https://github.com/open-gigaai/giga-models.git
cd giga-models
pip3 install -e .

cd projects/diffusion/cosmos-2.5/
```

## 🚀 Quick Start

### 1. Download Models

You can use the following script to download and convert the models, including the text encoder and VAE:

```bash
python scripts/download_and_conversion.py --save-dir /path/to/cosmos2_5/ --token TOKEN
```

### 2. Data Preparation

Organize your video data with corresponding text prompts:

```
raw_data/
├── 0.mp4                # Video file 0
├── 0.txt                # Prompt for video file 0
├── 1.mp4                # Video file 1
├── 1.txt                # Prompt for video file 1
├── ...
```

Pack the data and extract prompt embeddings:

```bash
python scripts/pack_data.py \
  --video-dir /path/to/raw_data/ \
  --save-dir /path/to/packed_data/
```

You can further use the `--transfer-mode` to generate depth or segmentation (seg) control information:

```bash
python scripts/pack_data.py \
  --video-dir /path/to/raw_data/ \
  --save-dir /path/to/packed_data/ \
  --transfer-mode depth
```

### 3. Training

Before starting training, ensure your video data is packed and prompt embeddings are extracted as shown in the previous steps.

```
# Train cosmos-predict2.5
python scripts/train.py --config configs.cosmos_predict25.config

# Train cosmos-predict2.5 with action control
python scripts/train.py --config configs.cosmos_predict25_action.config

# Train cosmos-transfer2.5 with depth/seg/edge/blur control
python scripts/train.py --config configs.cosmos_transfer25.config
```

### 4. Inference

Below are example commands for running inference with cosmos-predict2.5 or cosmos-transfer2.5. 
You can generate videos using either a single GPU or multiple GPUs. Adjust `--gpu-ids` according to your setup.

Use this command to run inference cosmos-predict2.5:

```bash
python scripts/inference.py \
  --data-path assets/base/data.json \
  --save-dir outputs/base \
  --transformer-model-path /path/to/cosmos2_5/models--nvidia--Cosmos-Predict2.5-2B/base/post-trained/transformer \
  --text-encoder-model-path /path/to/cosmos2_5/text_encoder \
  --vae-model-path /path/to/cosmos2_5/vae \
  --gpu_ids 0
```

Use this command to run inference cosmos-predict2.5 with action control:

```bash
python scripts/inference_action.py \
  --data-path assets/action/data.json \
  --save-dir outputs/action \
  --transformer-model-path /path/to/cosmos2_5/models--nvidia--Cosmos-Predict2.5-2B/robot/action-cond/transformer \
  --text-encoder-model-path /path/to/cosmos2_5/text_encoder \
  --vae-model-path /path/to/cosmos2_5/vae \
  --gpu_ids 0
```

Use this command to run inference cosmos-transfer2.5 with depth/seg/edge/blur control:

```bash
python scripts/inference_transfer.py \
  --data-path assets/transfer/data.json \
  --save-dir outputs/transfer/depth \
  --transformer-model-path /path/to/cosmos2_5/models--nvidia--Cosmos-Transfer2.5-2B/general/transformer \
  --controlnet-model-path /path/to/cosmos2_5/models--nvidia--Cosmos-Transfer2.5-2B/general/controlnet/depth \
  --text-encoder-model-path /path/to/cosmos2_5/text_encoder \
  --vae-model-path /path/to/cosmos2_5/vae \
  --mode depth \
  --gpu_ids 0
```

After running inference, the generated videos will be saved in the directory specified by `--save-dir`.
