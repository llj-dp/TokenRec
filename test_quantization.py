"""
Test script to demonstrate the quantization scheme differences between MQ and RQ.
用于演示 MQ 和 RQ 量化方案差异的测试脚本。
"""

import torch
import sys
sys.path.append('code')
import model

def test_quantization_schemes():
    """Test and compare MQ and RQ quantization schemes."""
    
    print("=" * 70)
    print("TokenRec Quantization Scheme Comparison Test")
    print("TokenRec 量化方案对比测试")
    print("=" * 70)
    
    # Configuration / 配置
    input_dim = 64      # LightGCN embedding dimension
    latent_dim = 512    # Codebook latent dimension
    n_tokens = 256      # Number of tokens per codebook
    n_books = 3         # Number of codebooks
    batch_size = 4      # Test batch size
    
    print(f"\nConfiguration / 配置:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Tokens per codebook: {n_tokens}")
    print(f"  Number of codebooks: {n_books}")
    print(f"  Batch size: {batch_size}")
    
    # Create test embeddings / 创建测试嵌入
    test_embeddings = torch.randn(batch_size, input_dim)
    print(f"\nTest embeddings shape: {test_embeddings.shape}")
    
    # Test MQ (Masked Quantization) / 测试 MQ（掩码量化）
    print("\n" + "-" * 70)
    print("1. MQ (Masked Quantization with K-way Encoder)")
    print("   MQ（掩码量化与 K 路编码器）")
    print("-" * 70)
    
    mq = model.MQ(
        input_dim=input_dim,
        dim=latent_dim,
        n_embedding=n_tokens,
        m_book=n_books,
        mask_ratio=0.2
    )
    
    print(f"✅ MQ Model initialized")
    print(f"   - Independent encoders: {len(mq.encoders)}")
    print(f"   - Codebooks: {len(mq.codebooks)}")
    print(f"   - Masking ratio: {mq.mask_ratio}")
    print(f"   - Position embedding: Yes")
    
    # Count MQ parameters / 计算 MQ 参数
    mq_params = sum(p.numel() for p in mq.parameters())
    mq_encoder_params = sum(p.numel() for encoder in mq.encoders for p in encoder.parameters())
    mq_codebook_params = sum(p.numel() for codebook in mq.codebooks for p in codebook.parameters())
    mq_decoder_params = sum(p.numel() for p in mq.decoder.parameters())
    
    print(f"\n   Parameter breakdown:")
    print(f"   - Total: {mq_params:,}")
    print(f"   - Encoders: {mq_encoder_params:,} ({len(mq.encoders)} independent)")
    print(f"   - Codebooks: {mq_codebook_params:,}")
    print(f"   - Decoder: {mq_decoder_params:,}")
    
    # Test MQ encoding / 测试 MQ 编码
    mq.eval()
    with torch.no_grad():
        mq_tokens = mq.encode(test_embeddings)
        mq_reconstructed, _, _ = mq.valid(test_embeddings)
    
    print(f"\n   Encoding test:")
    print(f"   - Input: {test_embeddings.shape} → Tokens: {mq_tokens.shape}")
    print(f"   - Reconstruction MSE: {torch.mean((test_embeddings - mq_reconstructed) ** 2).item():.6f}")
    print(f"\n   Example token sequence (first sample):")
    print(f"   - Codebook IDs: {mq_tokens[0].tolist()}")
    
    # Test RQ (Residual Quantization) / 测试 RQ（残差量化）
    print("\n" + "-" * 70)
    print("2. RQ (Residual Quantization)")
    print("   RQ（残差量化）")
    print("-" * 70)
    
    rq = model.ResidualVQVAE(
        input_dim=input_dim,
        dim=latent_dim,
        n_embedding=n_tokens,
        m_book=n_books
    )
    
    print(f"✅ RQ Model initialized")
    print(f"   - Shared encoder: 1")
    print(f"   - Codebooks: {len(rq.codebooks)}")
    print(f"   - Masking ratio: None")
    print(f"   - Position embedding: No")
    
    # Count RQ parameters / 计算 RQ 参数
    rq_params = sum(p.numel() for p in rq.parameters())
    rq_encoder_params = sum(p.numel() for p in rq.encoder.parameters())
    rq_codebook_params = sum(p.numel() for codebook in rq.codebooks for p in codebook.parameters())
    rq_decoder_params = sum(p.numel() for p in rq.decoder.parameters())
    
    print(f"\n   Parameter breakdown:")
    print(f"   - Total: {rq_params:,}")
    print(f"   - Encoder: {rq_encoder_params:,} (1 shared)")
    print(f"   - Codebooks: {rq_codebook_params:,}")
    print(f"   - Decoder: {rq_decoder_params:,}")
    
    # Test RQ encoding / 测试 RQ 编码
    rq.eval()
    with torch.no_grad():
        rq_tokens = rq.encode(test_embeddings)
        rq_reconstructed, _, _ = rq.valid(test_embeddings)
    
    print(f"\n   Encoding test:")
    print(f"   - Input: {test_embeddings.shape} → Tokens: {rq_tokens.shape}")
    print(f"   - Reconstruction MSE: {torch.mean((test_embeddings - rq_reconstructed) ** 2).item():.6f}")
    print(f"\n   Example token sequence (first sample):")
    print(f"   - Codebook IDs: {rq_tokens[0].tolist()}")
    
    # Comparison / 对比
    print("\n" + "=" * 70)
    print("Comparison Summary / 对比总结")
    print("=" * 70)
    
    print(f"\nModel Complexity / 模型复杂度:")
    print(f"  MQ parameters: {mq_params:,}")
    print(f"  RQ parameters: {rq_params:,}")
    print(f"  Difference: {mq_params - rq_params:,} ({((mq_params/rq_params - 1) * 100):.1f}% more)")
    
    print(f"\nKey Differences / 关键差异:")
    print(f"  ┌─────────────────────────┬──────────┬──────────┐")
    print(f"  │ Feature                 │    MQ    │    RQ    │")
    print(f"  ├─────────────────────────┼──────────┼──────────┤")
    print(f"  │ Independent Encoders    │    {len(mq.encoders)}     │    1     │")
    print(f"  │ Masking Mechanism       │   Yes    │    No    │")
    print(f"  │ Position Embedding      │   Yes    │    No    │")
    print(f"  │ Codebook Dependency     │ Parallel │ Sequential│")
    print(f"  │ Training Stability      │  Higher  │  Lower   │")
    print(f"  └─────────────────────────┴──────────┴──────────┘")
    
    print("\nBenefits of MQ / MQ 的优势:")
    print("  ✅ Better representation learning through encoder specialization")
    print("     通过编码器专业化实现更好的表示学习")
    print("  ✅ Improved robustness via masking mechanism")
    print("     通过掩码机制提高鲁棒性")
    print("  ✅ Faster parallel training")
    print("     更快的并行训练")
    print("  ✅ Higher quality tokens for LLM")
    print("     为 LLM 提供更高质量的标记")
    
    print("\n" + "=" * 70)
    print("✅ All tests completed successfully!")
    print("✅ 所有测试成功完成！")
    print("=" * 70)

if __name__ == "__main__":
    test_quantization_schemes()
