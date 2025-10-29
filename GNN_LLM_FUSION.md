# GNN-LLM Deep Fusion Enhancement

## Overview

This enhancement adds a deep fusion mechanism that integrates LLM (Large Language Model) outputs with **both user and item** GNN (Graph Neural Network) features for improved recommendation performance.

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

1. **Feature Projection**: Projects LLM, user GNN, and item GNN features to a common hidden dimension
2. **Dual Cross-Attention**: 
   - Multi-head attention where LLM features attend to user GNN features
   - Multi-head attention where LLM features attend to item GNN features
3. **Adaptive Gating**: Separate learned gates to balance LLM with user and item GNN contributions
4. **Feature Combination**: Concatenates all gated features (user, item, and LLM)
5. **Output Transformation**: Projects fused features back to item embedding space
6. **Residual Connection**: Preserves original LLM information via weighted residual

### Key Components

```python
class GNNLLMFusion(nn.Module):
    - llm_proj: Projects LLM output to hidden dimension
    - user_gnn_proj: Projects user GNN embedding to hidden dimension
    - item_gnn_proj: Projects item GNN embedding to hidden dimension
    - user_cross_attention: Multi-head attention for user features (4 heads by default)
    - item_cross_attention: Multi-head attention for item features (4 heads by default)
    - gate_user: Gating mechanism for adaptive user fusion
    - gate_item: Gating mechanism for adaptive item fusion
    - fusion_transform: Transform combined features to output space
    - residual_weight: Learnable weight for residual connection
```

### Forward Pass

```
Input: 
  - llm_output [batch, llm_dim]: LLM predicted features
  - user_gnn_emb [batch, gnn_dim]: User GNN embeddings
  - item_gnn_emb [batch, gnn_dim]: Item GNN embeddings (optional)

1. Project to common space:
   llm_feat = llm_proj(llm_output)
   user_feat = user_gnn_proj(user_gnn_emb)
   item_feat = item_gnn_proj(item_gnn_emb)  # if provided

2. User Cross-attention:
   user_attn = user_cross_attention(query=llm_feat, key=user_feat, value=user_feat)
   user_gate = sigmoid(linear([llm_feat, user_attn]))
   user_gated = user_gate * llm_feat + (1 - user_gate) * user_attn

3. Item Cross-attention (if item features provided):
   item_attn = item_cross_attention(query=llm_feat, key=item_feat, value=item_feat)
   item_gate = sigmoid(linear([llm_feat, item_attn]))
   item_gated = item_gate * llm_feat + (1 - item_gate) * item_attn

4. Feature fusion:
   fusion_input = [user_gated, item_gated, llm_feat]  # or [user_gated, user_feat, llm_feat] if no item
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
   - Extract target item GNN embeddings from item_emb[target_id]
   - Get LLM output from projection layer
   - Apply fusion: `predicts = fusion_module(llm_output, user_gnn_emb, target_item_gnn_emb)`
3. **Optimization**: Include fusion parameters in optimizer
4. **Checkpointing**: Save fusion module state_dict

### Testing (test.py)

1. **Loading**: Load fusion module from checkpoint
2. **Fallback**: Gracefully handle missing fusion checkpoint
3. **Inference**: Apply same fusion mechanism as training with both user and item embeddings

### Utilities (utils.py)

Added `group_model_params_fusion()` to properly group parameters from:
- T5 model
- Projection layer  
- Fusion module

## Benefits

1. **Semantic + Structural**: Combines LLM's semantic understanding with GNN's structural knowledge from both user and item perspectives
2. **Dual Perspective Fusion**: Separately processes user preferences and item characteristics before combining
3. **Adaptive Fusion**: Learns to balance contributions from LLM, user, and item based on context
4. **Attention Mechanism**: Captures relevant user and item features for each prediction
5. **Residual Learning**: Preserves original LLM information
6. **Backward Compatible**: Can load models without fusion (falls back to standard prediction)
7. **Flexible**: Item features are optional - works with only user features for backward compatibility

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
