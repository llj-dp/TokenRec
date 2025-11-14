# User and Item Quantization Scheme in TokenRec

## Overview / 概述

TokenRec implements a sophisticated **Vector Quantized Variational AutoEncoder (VQ-VAE)** approach to tokenize both user and item embeddings from LightGCN into discrete codebook IDs. This document explains the quantization scheme, modifications, and benefits.

TokenRec 实现了一种复杂的**向量量化变分自编码器（VQ-VAE）**方法，将 LightGCN 的用户和物品嵌入向量量化为离散的码本 ID。本文档解释了量化方案、修改内容和优势。

## Background / 背景

In traditional recommendation systems, user and item representations are continuous embeddings. TokenRec introduces a novel paradigm:
1. **Learn continuous embeddings** using LightGCN (Graph Neural Network)
2. **Quantize embeddings** into discrete tokens using VQ-VAE
3. **Feed tokens** to T5 Large Language Model for generative recommendation

在传统推荐系统中，用户和物品表示是连续嵌入向量。TokenRec 引入了一种新颖的范式：
1. **学习连续嵌入** - 使用 LightGCN（图神经网络）
2. **量化嵌入** - 使用 VQ-VAE 将嵌入转换为离散标记
3. **输入标记** - 将标记输入 T5 大语言模型进行生成式推荐

## Two Quantization Approaches / 两种量化方法

TokenRec implements two quantization schemes:

### 1. RQ - Residual Quantization / 残差量化

**Traditional approach** with residual encoding:

```
Input Embedding (64-dim)
    ↓
Single Encoder → Latent (512-dim)
    ↓
Codebook 1 → Residual
    ↓
Codebook 2 → Residual
    ↓
Codebook 3 → Residual
    ↓
Sum all codebook embeddings
    ↓
Decoder → Reconstructed Embedding
```

**Characteristics / 特点:**
- ✅ Single encoder for all codebooks / 所有码本共享一个编码器
- ✅ Each codebook encodes the residual from previous / 每个码本编码前一个的残差
- ⚠️ Codebooks are dependent / 码本之间存在依赖关系
- ⚠️ No masking mechanism / 没有掩码机制

### 2. MQ - Masked Quantization (K-way Encoder) / 掩码量化（K 路编码器）

**Novel approach** introduced in TokenRec:

```
Input Embedding (64-dim) + Position Embedding
    ↓
Apply Mask (20% random masking on specific regions)
    ↓
╔═══════════════╦═══════════════╦═══════════════╗
║  Encoder 1    ║  Encoder 2    ║  Encoder 3    ║
║  (for Book 1) ║  (for Book 2) ║  (for Book 3) ║
╚═══════════════╩═══════════════╩═══════════════╝
       ↓                ↓                ↓
   Latent 1         Latent 2         Latent 3
       ↓                ↓                ↓
   Codebook 1       Codebook 2       Codebook 3
       ↓                ↓                ↓
       └────────────────┴────────────────┘
                        ↓
                Sum all codebook embeddings
                        ↓
                Single Decoder
                        ↓
            Reconstructed Embedding
```

**Key Innovations / 关键创新:**

#### a) K-way Encoder / K 路编码器
- **Separate encoder for each codebook** / 每个码本有独立的编码器
- Each encoder specializes in different aspects / 每个编码器专注于不同方面
- Parallel processing instead of sequential / 并行处理而非顺序处理

#### b) Masking Mechanism / 掩码机制
```python
# Progressive masking for each codebook
mask_ratio = 0.2  # 20% masking
for m in range(m_book):
    mask = x.bernoulli_(mask_ratio).bool()
    mask[:m*patch] = 0      # Release previous regions
    mask[(m+1)*patch:] = 0  # Release future regions
    x = torch.masked_fill(x, mask, 0)
```

- **Regional masking**: Only masks specific portions for each encoder / 区域掩码：每个编码器只掩码特定部分
- **Progressive learning**: Forces encoders to learn from incomplete information / 渐进学习：强制编码器从不完整信息学习
- **Robustness**: Improves generalization / 鲁棒性：提高泛化能力

#### c) Position Embedding / 位置嵌入
```python
self.pos = nn.Embedding(1, input_dim)
x += self.pos.weight  # Add position information
```
- Adds positional context to embeddings / 为嵌入添加位置上下文
- Helps encoders understand structure / 帮助编码器理解结构

## Architecture Details / 架构细节

### MQ Module Structure / MQ 模块结构

```python
class MQ(nn.Module):
    Components / 组件:
    
    1. Multiple Encoders (one per codebook) / 多个编码器（每个码本一个）:
       Input (64-dim) → 128 → 256 → 512 (latent)
       - BatchNorm + Dropout for regularization
       - ReLU activation
    
    2. Multiple Codebooks (learnable) / 多个码本（可学习）:
       - n_embedding: 256 codewords (default)
       - dim: 512-dimensional embeddings
       - Initialized uniformly
    
    3. Single Decoder (shared) / 单一解码器（共享）:
       512 (latent) → 256 → 128 → 64 (reconstructed)
       - BatchNorm + Dropout
       - ReLU activation
    
    4. Position Embedding / 位置嵌入:
       - 1 learnable position vector (64-dim)
```

## Training Process / 训练过程

### VQ-VAE Training / VQ-VAE 训练

```python
Training Loop / 训练循环:
1. K-means Initialization (every 50 epochs):
   - Cluster latent codes
   - Update codebook centers
   - Improves codebook utilization

2. Loss Function / 损失函数:
   L_total = L_reconstruction + λ_emb * L_embedding
   
   Where / 其中:
   - L_reconstruction: MSE between input and output
   - L_embedding: MSE between encoder output and codebook
   - Straight-through estimator for gradients
```

### Hyperparameters / 超参数

```python
codebook_dim = 512        # Latent dimension
n_token = 256            # Number of codewords per codebook
m_book = 3               # Number of codebooks
mask_ratio = 0.2         # Masking ratio for MQ
lr = 1e-3                # Learning rate
batch_size = 512         # Training batch size
```

## Benefits of MQ over RQ / MQ 相对于 RQ 的优势

### 1. Independent Learning / 独立学习
- **RQ**: Sequential, codebooks are dependent / 顺序的，码本相互依赖
- **MQ**: Parallel, each encoder learns independently / 并行的，每个编码器独立学习
- **Benefit**: More diverse and complementary representations / 优势：更多样化和互补的表示

### 2. Masked Auto-Encoding / 掩码自编码
- **RQ**: No masking / 无掩码
- **MQ**: Regional masking (20%) / 区域掩码（20%）
- **Benefit**: Better generalization and robustness / 优势：更好的泛化能力和鲁棒性

### 3. Position Awareness / 位置感知
- **RQ**: No position information / 无位置信息
- **MQ**: Learnable position embedding / 可学习的位置嵌入
- **Benefit**: Structural understanding of embeddings / 优势：对嵌入结构的理解

### 4. Encoder Specialization / 编码器专业化
- **RQ**: Single encoder for all / 所有码本共用一个编码器
- **MQ**: K separate encoders / K 个独立编码器
- **Benefit**: Each captures different semantic aspects / 优势：每个捕获不同的语义方面

### 5. Training Stability / 训练稳定性
- **RQ**: Gradient flow through residual chain / 梯度流经残差链
- **MQ**: Direct gradient to each encoder / 直接梯度到每个编码器
- **Benefit**: More stable and faster convergence / 优势：更稳定和更快的收敛

## Quantization Workflow / 量化工作流程

### Step 1: Train LightGCN / 训练 LightGCN
```bash
# Pre-trained embeddings are provided
user_emb: [num_users, 64]
item_emb: [num_items, 64]
```

### Step 2: Train VQ-VAE / 训练 VQ-VAE
```bash
cd code
python main.py --dataset=LastFM --vq --train_vq --vq_model=MQ --n_token=256 --n_book=3
```

**What happens / 发生了什么:**
1. Load LightGCN embeddings / 加载 LightGCN 嵌入
2. Train user VQ-VAE with MQ / 使用 MQ 训练用户 VQ-VAE
3. Train item VQ-VAE with MQ / 使用 MQ 训练物品 VQ-VAE
4. Save trained models / 保存训练的模型

### Step 3: Encode to Codebook IDs / 编码为码本 ID
```python
# For each user/item embedding:
codeword_idx = vq_model.encode(embedding)
# Returns: [m_book] tensor with codebook IDs
# e.g., user_123 → [45, 128, 201] (3 tokens)
```

### Step 4: Generate Token Sequences / 生成标记序列
```
User Sequence / 用户序列:
[user_tokens] [item_1_tokens] [item_2_tokens] ... [item_k_tokens]

Example / 示例:
[45, 128, 201] [12, 88, 199] [23, 45, 167] ... → Predict next item
```

## Performance Impact / 性能影响

### Quantization Quality / 量化质量
```
Reconstruction Error (MSE) / 重建误差:
- MQ: Lower reconstruction error
- MQ: Better preserves semantic information
- MQ: More compact and discriminative codes
```

### Downstream Recommendation / 下游推荐
```
Impact on Metrics / 对指标的影响:
✅ NDCG@10: Improved with MQ
✅ Hit@10: Better generalization
✅ Cold-start: More robust representations
```

## Implementation Example / 实现示例

### Using MQ for Quantization / 使用 MQ 进行量化

```python
import torch
from code import model

# Initialize MQ
user_vq = model.MQ(
    input_dim=64,      # LightGCN embedding dimension
    dim=512,           # Latent codebook dimension
    n_embedding=256,   # Number of codewords per book
    m_book=3,          # Number of codebooks
    mask_ratio=0.2     # Masking ratio
)

# Load pre-trained weights
user_vq.load_state_dict(torch.load('checkpoints/vq/user-MQ-lgn-LastFM-64.pth'))
user_vq.eval()

# Encode user embedding to codebook IDs
user_embedding = torch.randn(1, 64)  # Example user embedding
user_tokens = user_vq.encode(user_embedding)  # Returns [1, 3] tensor

print(f"User tokens: {user_tokens}")
# Output: User tokens: tensor([[45, 128, 201]])
```

## Comparison Table / 对比表

| Feature / 特性 | RQ (Residual) | MQ (Masked K-way) |
|----------------|---------------|-------------------|
| Encoders / 编码器 | 1 shared | K independent |
| Codebook dependency / 码本依赖 | Sequential | Parallel |
| Masking / 掩码 | ❌ No | ✅ Yes (20%) |
| Position embedding / 位置嵌入 | ❌ No | ✅ Yes |
| Training / 训练 | Slower | Faster |
| Reconstruction / 重建 | Good | Better |
| Generalization / 泛化 | Good | Better |
| Complexity / 复杂度 | Lower | Higher |
| Performance / 性能 | Baseline | **Recommended** |

## Why This Modification? / 为什么要这样修改？

### Problem with Traditional VQ-VAE / 传统 VQ-VAE 的问题
1. **Single encoder bottleneck** / 单编码器瓶颈
   - All codebooks share one encoder
   - Limited representation capacity

2. **Sequential dependency** / 顺序依赖
   - Later codebooks depend on earlier ones
   - Error propagation

3. **No self-supervision** / 无自监督
   - No masking or denoising
   - Overfitting risk

### MQ Solution / MQ 解决方案
1. **K-way parallel encoding** / K 路并行编码
   - Each encoder specializes
   - Higher capacity

2. **Independent codebooks** / 独立码本
   - No dependency chain
   - More robust

3. **Masked auto-encoding** / 掩码自编码
   - Self-supervised learning
   - Better generalization

## Conclusion / 结论

The **MQ (Masked Quantization with K-way Encoder)** approach represents a significant improvement over traditional Residual Quantization for user and item tokenization in recommendation systems. Key advantages include:

MQ（掩码量化与 K 路编码器）方法相比传统的残差量化，在推荐系统的用户和物品标记化方面有显著改进。主要优势包括：

- ✅ **Better representation learning** through independent encoders / 通过独立编码器实现更好的表示学习
- ✅ **Improved robustness** via masking mechanism / 通过掩码机制提高鲁棒性
- ✅ **Faster training** with parallel architecture / 通过并行架构加速训练
- ✅ **Higher quality** discrete tokens for LLM / 为 LLM 提供更高质量的离散标记
- ✅ **Enhanced downstream performance** in recommendation tasks / 在推荐任务中提升下游性能

This quantization scheme is crucial for bridging continuous GNN embeddings with discrete LLM tokens, enabling the generative recommendation paradigm in TokenRec.

这种量化方案对于连接连续的 GNN 嵌入和离散的 LLM 标记至关重要，使 TokenRec 中的生成式推荐范式成为可能。

## References / 参考文献

1. **TokenRec Paper**: "TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation" (arXiv:2406.10450)
2. **VQ-VAE**: "Neural Discrete Representation Learning" (van den Oord et al., 2017)
3. **Masked Autoencoders**: "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2022)
4. **LightGCN**: "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" (He et al., 2020)

---

**Author**: GitHub Copilot Agent  
**Date**: 2025-11-14  
**Status**: Complete ✅
