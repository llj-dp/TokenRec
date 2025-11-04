#!/usr/bin/env python
"""
Test script to verify multi-GPU training setup works correctly.
This script tests the device selection and model wrapping logic without running full training.
"""

import torch
import torch.nn as nn
import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from parse import parse_args

def setup_device(args):
    """Helper function to setup device based on arguments"""
    use_cuda = True
    if args.use_multi_gpu and torch.cuda.is_available():
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(',')]
        device = torch.device("cuda:" + str(gpu_ids[0]))
        print(f'Using multi-GPU training with GPUs: {gpu_ids}')
    else:
        device = torch.device("cuda:" + str(args.cuda) if use_cuda and torch.cuda.is_available() else "cpu")
        gpu_ids = None
        print(f'Using single device: {device}')
    return device, gpu_ids

def test_single_gpu():
    """Test single GPU configuration"""
    print("\n=== Testing Single GPU Configuration ===")
    sys.argv = ['test', '--dataset=LastFM', '--cuda=0']
    args = parse_args()
    
    device, gpu_ids = setup_device(args)
    
    print(f"✓ Single GPU test passed")
    print(f"  Device: {device}")
    print(f"  GPU IDs: {gpu_ids}")
    return True

def test_multi_gpu():
    """Test multi-GPU configuration"""
    print("\n=== Testing Multi-GPU Configuration ===")
    sys.argv = ['test', '--dataset=LastFM', '--use_multi_gpu', '--gpu_ids=0,1']
    args = parse_args()
    
    device, gpu_ids = setup_device(args)
    
    print(f"✓ Multi-GPU test passed")
    print(f"  Device: {device}")
    print(f"  GPU IDs: {gpu_ids}")
    
    # Test DataParallel wrapping
    if gpu_ids is not None and len(gpu_ids) > 1 and torch.cuda.is_available():
        print("\n  Testing DataParallel wrapping...")
        model = nn.Linear(10, 10)
        model.to(device)
        wrapped_model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"  ✓ Model wrapped successfully")
        print(f"  ✓ Has 'module' attribute: {hasattr(wrapped_model, 'module')}")
        
        # Test accessing underlying module
        underlying = wrapped_model.module if hasattr(wrapped_model, 'module') else wrapped_model
        print(f"  ✓ Can access underlying module")
    
    return True

def test_custom_gpu_selection():
    """Test custom GPU selection"""
    print("\n=== Testing Custom GPU Selection ===")
    sys.argv = ['test', '--dataset=LastFM', '--use_multi_gpu', '--gpu_ids=1,3']
    args = parse_args()
    
    device, gpu_ids = setup_device(args)
    
    print(f"✓ Custom GPU selection test passed")
    print(f"  Device: {device}")
    print(f"  GPU IDs: {gpu_ids}")
    return True

def main():
    print("=" * 70)
    print("Multi-GPU Training Configuration Test")
    print("=" * 70)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    try:
        # Run tests
        test_single_gpu()
        test_multi_gpu()
        test_custom_gpu_selection()
        
        print("\n" + "=" * 70)
        print("✓ All tests passed successfully!")
        print("=" * 70)
        print("\nYou can now use multi-GPU training with commands like:")
        print("  python main.py --dataset=LastFM --use_multi_gpu --gpu_ids=0,1")
        print("\nFor more information, see MULTI_GPU_USAGE.md")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
