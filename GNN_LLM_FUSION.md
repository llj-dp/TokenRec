# GNN-LLM Deep Fusion Enhancement

## Overview

This enhancement adds a deep fusion mechanism that integrates LLM (Large Language Model) outputs with GNN (Graph Neural Network) features for improved recommendation performance.

## Problem Statement

The original TokenRec system:
- Uses LightGCN to generate user and item embeddings
- Tokenizes these embeddings via VQ-VAE into codebook IDs
- Feeds tokenized sequences to T5 LLM
- Projects LLM output to item embedding space for prediction

**Issue**: The final prediction only uses the LLM's output. There's no deep fusion between:
- The semantic understanding from LLM (learned from tokenized sequences)
- The structural information from GNN (user/item graph embeddings)

## Solution: GNNLLMFusion Module

### Architecture

The `GNNLLMFusion` module implements a sophisticated fusion mechanism with:

1. **Feature Projection**: Projects LLM and GNN features to a common hidden dimension
2. **Cross-Attention**: Multi-head attention where LLM features attend to user GNN features
3. **Adaptive Gating**: Learned gates to balance LLM and attended GNN contributions
4. **Feature Combination**: Concatenates gated features with user features
5. **Output Transformation**: Projects fused features back to item embedding space
6. **Residual Connection**: Preserves original LLM information via weighted residual

### Key Components

```python
class GNNLLMFusion(nn.Module):
    - llm_proj: Projects LLM output to hidden dimension
    - user_gnn_proj: Projects user GNN embedding to hidden dimension
    - cross_attention: Multi-head attention (4 heads by default)
    - gate_llm: Gating mechanism for adaptive fusion
    - fusion_transform: Transform combined features to output space
    - residual_weight: Learnable weight for residual connection
```

### Forward Pass

```
Input: llm_output [batch, llm_dim], user_gnn_emb [batch, gnn_dim]

1. Project to common space:
   llm_feat = llm_proj(llm_output)
   user_feat = user_gnn_proj(user_gnn_emb)

2. Cross-attention:
   attn_output = cross_attention(query=llm_feat, key=user_feat, value=user_feat)

3. Adaptive gating:
   gate = sigmoid(linear([llm_feat, attn_output]))
   gated_feat = gate * llm_feat + (1 - gate) * attn_output

4. Feature fusion:
   fusion_input = [gated_feat, user_feat]
   fused_output = fusion_transform(fusion_input)

5. Residual connection:
   final_output = α * fused_output + (1 - α) * llm_output

Output: fused_output [batch, gnn_dim]
```

## Integration Points

### Training (train.py)

1. **Initialization**: Create fusion module alongside T5 and projection layer
2. **Forward Pass**: 
   - Extract user GNN embeddings from user_emb[user_id]
   - Get LLM output from projection layer
   - Apply fusion: `predicts = fusion_module(llm_output, user_gnn_emb)`
3. **Optimization**: Include fusion parameters in optimizer
4. **Checkpointing**: Save fusion module state_dict

### Testing (test.py)

1. **Loading**: Load fusion module from checkpoint
2. **Fallback**: Gracefully handle missing fusion checkpoint
3. **Inference**: Apply same fusion mechanism as training

### Utilities (utils.py)

Added `group_model_params_fusion()` to properly group parameters from:
- T5 model
- Projection layer  
- Fusion module

## Benefits

1. **Semantic + Structural**: Combines LLM's semantic understanding with GNN's structural knowledge
2. **Adaptive Fusion**: Learns to balance contributions based on context
3. **Attention Mechanism**: Captures relevant user features for each prediction
4. **Residual Learning**: Preserves original LLM information
5. **Backward Compatible**: Can load models without fusion (falls back to standard prediction)

## Usage

### Training from Scratch
```bash
cd code
python main.py --dataset=LastFM --vq --train_vq --vq_model=MQ --n_token=256 --n_book=3
```

### Training from Checkpoint
```bash
python main.py --dataset=LastFM --n_token=256 --n_book=3 --train_from_checkpoint
```

### Evaluation
```bash
python main.py --dataset=LastFM --no_train
```

## File Changes

- **model.py**: Added `GNNLLMFusion` class
- **train.py**: Integrated fusion in training and validation loops
- **test.py**: Integrated fusion in testing with fallback
- **utils.py**: Added `group_model_params_fusion()` utility

## Future Enhancements

1. **Item-level Fusion**: Extend fusion to also incorporate item GNN features
2. **Multi-hop Attention**: Use multiple attention layers for deeper interaction
3. **Dynamic Fusion Weights**: Make fusion weights instance-dependent
4. **Graph Structure**: Directly incorporate graph structure into fusion
5. **Contrastive Learning**: Add contrastive loss between fused and original representations

## References

- Original TokenRec Paper: [arXiv:2406.10450](https://arxiv.org/pdf/2406.10450)
- Multi-head Attention: "Attention Is All You Need" (Vaswani et al., 2017)
- LightGCN: "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" (He et al., 2020)
