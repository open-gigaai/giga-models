# Pi0 / Pi0.5: PyTorch Reproduction for Vision-Language-Action Robot Control

## ✨ Introduction

This project provides a clean PyTorch reproduction of Pi0 and Pi0.5 built on top of LeRobot-style datasets(v2.1) and aligned with the OpenPI design. It supports end-to-end training and inference with better performance and lightweight deployment. In particular:

- Leverages OpenPI's architectural insights while using standard PyTorch toolchains.
- Integrates seamlessly with LeRobot-format datasets for plug-and-play data loading and normalization.
- Provides both training and inference pipelines, plus conversion scripts to bring official JAX checkpoints into PyTorch.

The original papers are:

- **π₀: A Vision-Language-Action Flow Model for General Robot Control** ([https://arxiv.org/html/2410.24164v1](https://arxiv.org/html/2410.24164v1))
- **π₀.₅: a Vision-Language-Action Model with Open-World Generalization** ([https://arxiv.org/html/2504.16054](https://arxiv.org/html/2504.16054))

## 📈 Advantages vs OpenPI

Internal benchmarks confirm that this PyTorch reproduction achieves significant performance gains.

### Training time comparison

- **Hardware**: 8× NVIDIA H20 GPUs (96GB VRAM per unit).
- **Protocol**: All benchmarks utilize a constant global batch size of 256 (32 samples per GPU) and are trained for 50,000 steps.
- **Dataloader**: 16 workers per GPU.
- **Optimization**: 
  - FSDP2 for model and optimizer sharding
  - Mixed-precision training (FP32 weights/gradients, BF16 most computations/activations)
  - EMA weights to improve generalization ability and stability
  - TorchDynamo to optimize training
  - Activation checkpointing applied to blocks

| Model         | OpenPI-JAX-Pi0 | OpenPI-JAX-Pi0.5 | Ours-Torch-Pi0 | Ours-Torch-Pi0.5 |
| ------------- |----------------|------------------|----------------|------------------|
| Training time | 58 h           | 67 h             | 50 h           | 61 h             |

* Ours-Torch-Pi0.5 model supports a **maximum training batch size of 192**, significantly surpassing the official implementation's cap of 32.
* Ours-Torch-Pi0.5 model supports full training on GPUs with **40GB VRAM or less**, a significant reduction from the official version's larger memory requirements.

### Inference time comparison

- **Hardware**: 1× NVIDIA H20 GPUs (96GB VRAM per unit).
- **Protocol**: Data transformations without image resizing; Warmup for 1–2 steps to compile kernels.
- **Optimization**: Improve inference throughput and latency with SDPA attention and TorchDynamo (using max-autotune mode and fullgraph).

| Model          | OpenPI-JAX-Pi0 | OpenPI-JAX-Pi0.5 | Ours-Torch-Pi0 | Ours-Torch-Pi0.5 |
| -------------- |----------------|------------------|----------------|------------------|
| Inference time | 79.47 ms       | 90.05 ms         | 72.0 ms        | 89.4 ms          |

## ⚡ Installation

We recommend a fresh conda environment.

```bash
conda create -n giga_pi0 python=3.11.10 -y
conda activate giga_pi0

git clone https://github.com/open-gigaai/giga-models.git
cd giga-models
pip3 install -e .

pip3 install giga-train
pip3 install giga-datasets
pip3 install lerobot==0.3.2
```

## 🚀 Quick Start

### 1. Data preparation (LeRobot format) and normalization

If your dataset is already in LeRobot format, compute normalization stats for `observation.state` and `action` using our script:

```bash
cd projects/vla/pi0/

python scripts/compute_norm_stats.py \
  --data-paths /path/to/lerobot_dataset1 /path/to/lerobot_dataset2 \
  --output-path /path/to/norm_stats.json \
  --sample-rate 1.0 \
  --action-chunk 50 \
  --action-dim 32

```

Then point your training config to the produced `norm_stats.json` (see examples in `configs`).

### 2. Download official OpenPI checkpoints and convert to PyTorch

Use OpenPI's downloader to fetch JAX checkpoints, then convert using our script.

Download using gsutil command:

```bash
gsutil cp -r gs://openpi-assets/checkpoints/pi0_base /path/to/local
gsutil cp -r gs://openpi-assets/checkpoints/pi05_base /path/to/local
```

> or use the downloader script from your OpenPI codebase and environment

Convert to PyTorch (Pi0):

```bash
python scripts/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /path/to/pi0_base/params \
  --precision float32 \
  --tokenizer-id google/paligemma-3b-pt-224 \
  --output-path /path/to/torch_pi0_base \
```

Convert to PyTorch (Pi0.5):

```bash
python scripts/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /path/to/pi05_base/params \
  --precision float32 \
  --tokenizer-id google/paligemma-3b-pt-224 \
  --output-path /path/to/torch_pi05_base \
  --pi05-enabled
```

### 3. Training

We provide ready-to-use configs for Pi0 and Pi0.5. Adjust `gpu_ids`, `batch_size_per_gpu`, `data_paths`, and `norm_stats_path` as needed.

Logs, configs and checkpoints will be stored at the path `project_dir`

Pi0 training (FSDP2):

```bash
python scripts/train.py --config configs.pi0.config
```

Pi0.5 training (FSDP2):

```bash
python scripts/train.py --config configs.pi05.config
```

### 4. Inference

Run Pi0/Pi0.5 inference on a LeRobot dataset and optionally visualize predictions.

```bash
python scripts/inference.py \
  --model-path /path/to/torch_pi0_base \
  --tokenizer-model-path google/paligemma-3b-pt-224 \
  --data-path /path/to/lerobot_dataset \
  --norm-stats-path /path/to/norm_stats.json \
  --vis-output-path /tmp/pi0_vis \
  --device cuda:0 \
  --action-chunk 50 \
  --original-action-dim 14
```

### 5. Robot deployment


* Run the server:

  ```bash
  python scripts/inference_server.py \
    --model-path /path/to/torch_pi0_or_pi05_base \
    --tokenizer-model-path google/paligemma-3b-pt-224 \
    --norm-stats-path /path/to/norm_stats.json \
    --original-action-dim 14
  ```

* Run the client:

  ```bash
  python scripts/inference_client.py
  ```

This is a minimal client example. It generates random observations to demonstrate the end-to-end request/response flow with the server. You can copy the relevant client code onto your robot and replace the random inputs with real onboard sensor data (e.g., cameras, proprioception) and your robot's control interface. Ensure input shapes and field names remain consistent with the server's expectations.

Make sure the host and port are the same in both server and client.


## 🤝 Acknowledgements

We gratefully acknowledge the authors and maintainers of

- [OpenPI](https://github.com/Physical-Intelligence/openpi) for original Pi0/Pi0.5 design and JAX reference
- [LeRobot](https://github.com/huggingface/lerobot) for dataset, tooling, and PyTorch implementations of Pi0/Pi0_FAST
- [Transformers](https://github.com/huggingface/transformers) for PaliGemma and related vision-language components

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

______________________________________________________________________

If you use this codebase or the converted weights, please also cite the original Pi0/Pi0.5 work.
