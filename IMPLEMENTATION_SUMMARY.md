# Multi-GPU Training Implementation Summary

## Overview
This document summarizes the changes made to enable multi-GPU training support for TokenRec's LLM-based recommendation downstream task.

## Problem Statement
The original TokenRec implementation only supported single GPU training. Users requested the ability to:
1. Train on multiple GPUs on a single machine
2. Specify which GPUs to use
3. Achieve faster training through parallel processing

## Solution
Implemented PyTorch's DataParallel approach for multi-GPU training, which:
- Is easy to use and requires minimal code changes
- Automatically distributes batches across multiple GPUs
- Works well for 2-4 GPUs on a single machine
- Maintains backward compatibility with single-GPU training

## Changes Made

### 1. Command-Line Arguments (parse.py)
Added two new arguments:
- `--use_multi_gpu`: Flag to enable multi-GPU training
- `--gpu_ids`: Comma-separated list of GPU IDs (e.g., "0,1,2,3")

### 2. Device Selection (main.py)
- Implemented logic to parse GPU IDs and setup CUDA devices
- Set CUDA_VISIBLE_DEVICES environment variable
- Pass gpu_ids to training function
- Maintain backward compatibility with single-GPU mode

### 3. Model Training (train.py)
Modified the `backbone()` function to:
- Accept optional `gpu_ids` parameter
- Wrap T5 model and linear projection with `nn.DataParallel`
- Handle `_shift_right` method calls on potentially wrapped models
- Properly save/load models by unwrapping DataParallel when needed

### 4. Documentation
Created comprehensive documentation:
- `MULTI_GPU_USAGE.md`: Detailed usage guide with examples
- Updated `README.md`: Added multi-GPU section with quick examples
- `test_multi_gpu.py`: Test script to verify configuration
- `IMPLEMENTATION_SUMMARY.md`: This file

### 5. Repository Hygiene
- Added `.gitignore` to exclude `__pycache__` and build artifacts
- Removed cached Python bytecode files

## Key Implementation Details

### DataParallel Wrapping
```python
if gpu_ids is not None and len(gpu_ids) > 1:
    t5 = nn.DataParallel(t5, device_ids=gpu_ids)
    linear_projection = nn.DataParallel(linear_projection, device_ids=gpu_ids)
```

### Model Saving (unwrap DataParallel)
```python
t5_to_save = t5.module if hasattr(t5, 'module') else t5
linear_projection_to_save = linear_projection.module if hasattr(linear_projection, 'module') else linear_projection
```

### Accessing Model Methods
```python
t5_module = t5.module if hasattr(t5, 'module') else t5
decoder_input_ids = t5_module._shift_right(decoder_input_ids)
```

## Usage Examples

### Single GPU (Original Behavior)
```bash
python main.py --dataset=LastFM --cuda=0
```

### Multi-GPU with 2 GPUs
```bash
python main.py --dataset=LastFM --use_multi_gpu --gpu_ids=0,1
```

### Multi-GPU with Custom Selection
```bash
python main.py --dataset=LastFM --use_multi_gpu --gpu_ids=1,3 --batch=256
```

### Full Pipeline with Multi-GPU
```bash
python main.py --dataset=LastFM --vq --train_vq --vq_model=MQ --n_token=256 --n_book=3 --use_multi_gpu --gpu_ids=0,1,2,3
```

## Testing
The implementation has been verified to:
- ✓ Parse command-line arguments correctly
- ✓ Set up devices properly for both single and multi-GPU modes
- ✓ Wrap models with DataParallel when appropriate
- ✓ Handle model saving/loading correctly
- ✓ Maintain backward compatibility with single-GPU training
- ✓ Include comprehensive documentation

## Performance Considerations

### When to Use Multi-GPU
- Training large models (T5-base or larger)
- Large batch sizes (128+)
- Long training runs (many epochs)
- Available GPUs have similar specifications

### Expected Speedup
- 2 GPUs: ~1.5-1.8x speedup
- 4 GPUs: ~2.5-3.5x speedup
(Actual speedup depends on model size, batch size, and GPU interconnect)

### Batch Size Recommendations
With DataParallel, each GPU processes a portion of the batch:
- Single GPU with batch=128 → 2 GPUs with batch=256
- Single GPU with batch=64 → 4 GPUs with batch=256

## Limitations
- DataParallel uses multi-threading and has overhead for gradient synchronization
- Best suited for 2-4 GPUs; for more GPUs, consider DistributedDataParallel (DDP)
- All GPUs must be on the same machine
- VQ training phase remains single-GPU (could be extended in future)

## Future Enhancements
Potential improvements for future versions:
1. Implement DistributedDataParallel (DDP) for better scaling with 4+ GPUs
2. Support multi-node training across multiple machines
3. Add mixed precision training (AMP) for additional speedup
4. Extend multi-GPU support to VQ training phase

## Backward Compatibility
All changes maintain full backward compatibility:
- Default behavior (no flags) uses single GPU as before
- Existing scripts and commands work unchanged
- Checkpoints are compatible between single and multi-GPU modes

## Branch
The multi-GPU implementation is available on the `mutil-GPU` branch (note: branch name follows user's spelling preference).

## Technical Notes
- Primary device is always the first GPU in the gpu_ids list
- CUDA_VISIBLE_DEVICES is set to constrain which GPUs PyTorch can see
- Model state_dict is saved without DataParallel wrapper for compatibility
- All model method calls check for wrapped module using hasattr
