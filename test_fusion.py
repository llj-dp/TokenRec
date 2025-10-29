"""
Test script to verify the GNNLLMFusion module functionality
"""
import torch
import sys
sys.path.append('/home/runner/work/TokenRec/TokenRec/code')
from model import GNNLLMFusion

def test_fusion_module():
    """Test the basic functionality of GNNLLMFusion module"""
    print("Testing GNNLLMFusion module...")
    
    # Test parameters
    batch_size = 4
    llm_dim = 64
    gnn_dim = 64
    hidden_dim = 128
    num_heads = 4
    
    # Create fusion module
    fusion = GNNLLMFusion(
        llm_dim=llm_dim,
        gnn_dim=gnn_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads
    )
    
    print(f"✓ Fusion module created successfully")
    print(f"  - LLM dim: {llm_dim}")
    print(f"  - GNN dim: {gnn_dim}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Attention heads: {num_heads}")
    
    # Create sample inputs
    llm_output = torch.randn(batch_size, llm_dim)
    user_gnn_emb = torch.randn(batch_size, gnn_dim)
    
    print(f"\n✓ Created sample inputs")
    print(f"  - LLM output shape: {llm_output.shape}")
    print(f"  - User GNN embedding shape: {user_gnn_emb.shape}")
    
    # Forward pass
    fusion.eval()
    with torch.no_grad():
        fused_output = fusion(llm_output, user_gnn_emb)
    
    print(f"\n✓ Forward pass successful")
    print(f"  - Fused output shape: {fused_output.shape}")
    print(f"  - Expected shape: ({batch_size}, {gnn_dim})")
    
    # Verify output shape
    assert fused_output.shape == (batch_size, gnn_dim), \
        f"Output shape mismatch: {fused_output.shape} vs ({batch_size}, {gnn_dim})"
    
    print(f"\n✓ Output shape verification passed")
    
    # Test with training mode
    fusion.train()
    fused_output_train = fusion(llm_output, user_gnn_emb)
    
    print(f"\n✓ Training mode forward pass successful")
    
    # Test gradient flow
    loss = fused_output_train.sum()
    loss.backward()
    
    # Check if gradients exist
    has_grads = any(p.grad is not None for p in fusion.parameters())
    assert has_grads, "No gradients computed"
    
    print(f"✓ Gradient computation successful")
    
    # Count parameters
    total_params = sum(p.numel() for p in fusion.parameters())
    trainable_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    
    print(f"\n✓ Parameter count:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    
    return True

if __name__ == "__main__":
    try:
        test_fusion_module()
        print("\n✅ GNNLLMFusion module is working correctly!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
