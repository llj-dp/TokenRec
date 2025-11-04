# Multi-GPU Training Guide

This guide explains how to use the multi-GPU training feature for the LLM-based recommendation downstream task in TokenRec.

## Overview

The multi-GPU training feature allows you to train the T5-based recommendation model across multiple GPUs on a single machine using PyTorch's DataParallel. This can significantly speed up training by distributing the workload across multiple GPUs.

## Requirements

- Multiple NVIDIA GPUs on the same machine
- PyTorch with CUDA support
- All dependencies listed in `requirements.txt`

## Usage

### Single GPU Training (Default)

To train on a single GPU (e.g., GPU 0):

```bash
cd code
python main.py --dataset=LastFM --cuda=0
```

### Multi-GPU Training

To train on multiple GPUs, use the `--use_multi_gpu` flag and specify GPU IDs:

#### Example 1: Use GPUs 0 and 1
```bash
cd code
python main.py --dataset=LastFM --use_multi_gpu --gpu_ids=0,1
```

#### Example 2: Use GPUs 0, 1, 2, and 3
```bash
cd code
python main.py --dataset=LastFM --use_multi_gpu --gpu_ids=0,1,2,3
```

#### Example 3: Use specific GPUs (e.g., 1 and 3)
```bash
cd code
python main.py --dataset=LastFM --use_multi_gpu --gpu_ids=1,3
```

### Full Training Pipeline with Multi-GPU

#### Complete pipeline with VQ training and multi-GPU LLM training
```bash
cd code
python main.py --dataset=LastFM --vq --train_vq --vq_model=MQ --n_token=256 --n_book=3 --use_multi_gpu --gpu_ids=0,1
```

#### Train from checkpoint with multi-GPU
```bash
cd code
python main.py --dataset=LastFM --n_token=256 --n_book=3 --train_from_checkpoint --use_multi_gpu --gpu_ids=0,1,2
```

## New Command-Line Arguments

### `--use_multi_gpu`
- **Type**: Flag (boolean)
- **Default**: False
- **Description**: Enable multi-GPU training using PyTorch DataParallel
- **Usage**: Add this flag to enable multi-GPU mode

### `--gpu_ids`
- **Type**: String
- **Default**: '0,1'
- **Description**: Comma-separated list of GPU IDs to use for training
- **Usage**: `--gpu_ids=0,1,2,3` (uses GPUs 0, 1, 2, and 3)
- **Note**: The first GPU in the list will be set as the primary device

## How It Works

1. **Device Setup**: When `--use_multi_gpu` is enabled, the code parses the GPU IDs from `--gpu_ids` and sets up the CUDA devices accordingly.

2. **Model Wrapping**: Both the T5 model and the linear projection layer are wrapped with `nn.DataParallel`, which automatically distributes the batch across the specified GPUs.

3. **Training**: During training, each batch is split across the GPUs, and gradients are automatically synchronized.

4. **Saving**: When saving checkpoints, the code automatically unwraps the DataParallel models to save the underlying model weights.

5. **Loading**: Checkpoints saved from multi-GPU training can be loaded for both single-GPU and multi-GPU inference/training.

## Performance Considerations

- **Batch Size**: When using multiple GPUs, you may want to increase the batch size proportionally. For example, if you use 2 GPUs with a batch size of 128, each GPU will process 64 samples per batch.

- **GPU Memory**: Ensure all GPUs have sufficient memory for the model and batch size you're using.

- **GPU Balance**: DataParallel works best when all GPUs are of the same type and have similar performance characteristics.

## Troubleshooting

### CUDA Out of Memory
- Reduce the batch size: `--batch=64` or `--batch=32`
- Reduce the sequence length: `--source_length=256`

### GPUs Not Detected
- Check that CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify GPU IDs: `nvidia-smi` to see available GPUs

### Performance Not Scaling Linearly
- This is expected behavior for DataParallel due to communication overhead
- Consider using DistributedDataParallel (DDP) for better scaling with 4+ GPUs

## Example Workflow

Here's a complete workflow for training with multi-GPU:

```bash
# 1. Navigate to code directory
cd code

# 2. Train VQ model (single GPU is fine for this step)
python main.py --dataset=LastFM --vq --train_vq --vq_model=MQ --n_token=256 --n_book=3

# 3. Train LLM recommendation model with multi-GPU
python main.py --dataset=LastFM --n_token=256 --n_book=3 --use_multi_gpu --gpu_ids=0,1,2,3 --batch=256 --epochs=100

# 4. Evaluate (can use single GPU for evaluation)
python main.py --dataset=LastFM --no_train --cuda=0
```

## Notes

- The VQ training phase (`--vq --train_vq`) currently runs on a single GPU. Multi-GPU support is primarily for the LLM training phase.
- Evaluation (`--no_train`) runs on a single GPU by default for consistency.
- All training parameters (learning rate, weight decay, etc.) work the same with multi-GPU as with single-GPU training.
