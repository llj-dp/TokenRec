# TokenRec
A LLM-based Recommender System with user&amp;item Tokenizers and a generative retrieval paradigm. The overall framework of the proposed TokenRec, which consists of the masked vector-quantized tokenizer with a K-way encoder for item ID tokenization and the generative retrieval paradigm for recommendation generation. Our paper is available at [arXiv-TokenRec](https://arxiv.org/pdf/2406.10450).

<img width="806" alt="1743834485885" src="https://github.com/user-attachments/assets/0256fa8e-ca35-41a2-abd6-75fedd4b0a20" />

## 🆕 Enhanced with Deep GNN-LLM Fusion

This enhanced version includes a **deep fusion mechanism** that integrates LLM outputs with **both user and item** GNN (Graph Neural Network) features for improved recommendation quality. The fusion module uses dual multi-head cross-attention and adaptive gating to effectively combine:
- **Semantic understanding** from LLM (learned from tokenized sequences)
- **User structural information** from GNN (user graph embeddings)
- **Item structural information** from GNN (item graph embeddings)

📖 **See [GNN_LLM_FUSION.md](GNN_LLM_FUSION.md) for detailed technical documentation**

## 📚 Technical Documentation

- **[QUANTIZATION_SCHEME.md](QUANTIZATION_SCHEME.md)** - Comprehensive guide to user and item quantization (MQ vs RQ) / 用户和物品量化方案详解
- **[GNN_LLM_FUSION.md](GNN_LLM_FUSION.md)** - Deep GNN-LLM fusion mechanism / 深度 GNN-LLM 融合机制
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation summary and technical details / 实现摘要和技术细节

### Key Improvements
- ✅ Dual multi-head cross-attention for both user and item GNN features
- ✅ Separate adaptive gating for user and item feature balancing
- ✅ Residual connections to preserve original information
- ✅ Backward compatible with existing checkpoints
- ✅ 363K additional trainable parameters for comprehensive feature fusion



## An example of Implementation

Please download the checkpoints at [Google Drive](https://drive.google.com/drive/folders/12OFUuX7a5v7khx_MZiel04N0x5prkdGy?usp=drive_link), and put them in the path of "checkpoints/".

1. **Go to the path of "code"**
```
cd code
```


2. **Whole Pipeline**
```
python main.py --dataset=LastFM --vq --train_vq --vq_model=MQ --n_token=256 --n_book=3
```

3. **Train from checkpoint (LLM)**
```
python main.py --dataset=LastFM --n_token=256 --n_book=3 --train_from_checkpoint
```

4. **Evaluation**
```
python main.py --dataset=LastFM --no_train
```

5. **Test Fusion Module** (Enhanced)
```
python test_fusion.py
```

## Citation
If this project is helpful to your research, please cite our papers:

Qu, Haohao, Wenqi Fan, Zihuai Zhao, and Qing Li. "Tokenrec: learning to tokenize id for llm-based generative recommendation." arXiv preprint arXiv:2406.10450 (2024).
```shell
@article{qu2024tokenrec,
  title={Tokenrec: learning to tokenize id for llm-based generative recommendation},
  author={Qu, Haohao and Fan, Wenqi and Zhao, Zihuai and Li, Qing},
  journal={arXiv preprint arXiv:2406.10450},
  year={2024}
}
```
