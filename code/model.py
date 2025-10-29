import torch
import torch.nn as nn
import utils
import copy
import sys


# K-way + masking
class MQ(nn.Module):
    def __init__(self, input_dim, dim, n_embedding, m_book, mask_ratio=0.2):
        super(MQ, self).__init__()
        self.m_book = m_book
        self.encoders = nn.ModuleList()
        self.codebooks = nn.ModuleList()
        for m in range(m_book):
            codebook = nn.Embedding(n_embedding, dim)
            codebook.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
            encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(256, dim),
                )
            self.codebooks.append(codebook)
            self.encoders.append(encoder)
        
        self.pos = nn.Embedding(1, input_dim)
        self.pos.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
        self.mask_ratio = mask_ratio
            
        self.decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.ReLU(), 
            nn.Linear(128, input_dim),
            )

    def forward(self, x):  # shape = [batch, emb]
        b, e = x.shape
        patch = int(e/self.m_book)
        
        # encode    
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            # mask & position
            mask = x[0, :].bernoulli_(self.mask_ratio).bool()
            mask[:m*patch] = 0  # release the specific masked part
            mask[(m+1)*patch-1:] = 0
            x = torch.masked_fill(x, mask, 0)
            x += self.pos.weight
            
            # quantization
            ze = self.encoders[m](x)
            embedding = self.codebooks[m].weight    
            N, C = ze.shape  # ze: [batch, dim]
            K, _ = embedding.shape  # embedding [n_codewords, dim]
            ze_broadcast = ze.reshape(N, 1, C)
            embedding_broadcast = embedding.reshape(1, K, C)
            distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
            nearest_neighbor = torch.argmin(distance, 1)
            ce = self.codebooks[m](nearest_neighbor)
            ce_list.append(ce)
            res_list.append(ze)
            ce = ze + (ce - ze).detach()  # straight through loss
                        
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        decoder_input = zq

        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, res_list, ce_list
    
    def valid(self, x):  # shape = [batch, emb]
        x += self.pos.weight.data
        
        # encode    
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            ze = self.encoders[m](x)
            embedding = self.codebooks[m].weight.data   
            N, C = ze.shape  # ze: [batch, dim]
            K, _ = embedding.shape  # embedding [n_codewords, dim]
            ze_broadcast = ze.reshape(N, 1, C)
            embedding_broadcast = embedding.reshape(1, K, C)
            distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
            nearest_neighbor = torch.argmin(distance, 1)
            ce = self.codebooks[m](nearest_neighbor)
            ce_list.append(ce)
            res_list.append(ze)
            
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        decoder_input = zq

        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, res_list, ce_list
    
    def encode(self, x):
        x += self.pos.weight.data
        nearest_neighbor_list = []
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            ze = self.encoders[m](x)
            embedding = self.codebooks[m].weight.data  
            N, C = ze.shape  # ze: [batch, dim]
            K, _ = embedding.shape  # embedding [n_codewords, dim]
            ze_broadcast = ze.reshape(N, 1, C)
            embedding_broadcast = embedding.reshape(1, K, C)
            distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
            nearest_neighbor = torch.argmin(distance, 1)
            ce = self.codebooks[m](nearest_neighbor)
            ce_list.append(ce)
            res_list.append(ze)
            nearest_neighbor_list.append(nearest_neighbor)
            
        codeword_idx = torch.stack(nearest_neighbor_list, dim=0).transpose(0, 1)  # shape = [batch_size, n_codebook]
        return codeword_idx
    

class ResidualVQVAE(nn.Module):
    def __init__(self, input_dim, dim, n_embedding, m_book):
        super(ResidualVQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, dim),
            )

        self.m_book = m_book
        self.codebooks = nn.ModuleList()
        for m in range(m_book):
            codebook = nn.Embedding(n_embedding, dim)
            codebook.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
            self.codebooks.append(codebook)
            
        self.decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.ReLU(), 
            nn.Linear(128, input_dim),
            )

    def forward(self, x):
        # encode
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            if m == 0:
                ze = self.encoder(x)
                embedding = self.codebooks[0].weight
                N, C = ze.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                ze_broadcast = ze.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebooks[0](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(ze)
            else:
                res = res_list[m-1] - ce_list[m-1]
                embedding = self.codebooks[m].weight  # It should be learnable!
                N, C = res.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                res_broadcast = res.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - res_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebooks[m](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(res)
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        decoder_input = ze + (zq - ze).detach()

        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, res_list, ce_list

    def valid(self, x):
        # encode
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            if m == 0:
                ze = self.encoder(x)
                embedding = self.codebooks[0].weight.data
                N, C = ze.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                ze_broadcast = ze.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebooks[0](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(ze)
            else:
                res = res_list[m-1] - ce_list[m-1]
                embedding = self.codebooks[m].weight.data  # It should be learnable!
                N, C = res.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                res_broadcast = res.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - res_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebooks[m](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(res)
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        decoder_input = ze + (zq - ze).detach()

        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, res_list, ce_list

    def encode(self, x):
        nearest_neighbor_list = []
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            if m == 0:
                ze = self.encoder(x)
                embedding = self.codebooks[0].weight
                N, C = ze.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                ze_broadcast = ze.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebooks[0](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(ze)
                nearest_neighbor_list.append(nearest_neighbor)
            else:
                res = res_list[m-1] - ce_list[m-1]
                embedding = self.codebooks[m].weight
                N, C = res.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                res_broadcast = res.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - res_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebooks[m](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(res)
                nearest_neighbor_list.append(nearest_neighbor)
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        codeword_idx = torch.stack(nearest_neighbor_list, dim=0).transpose(0, 1)  # shape = [batch_size, n_codebook]
        return codeword_idx


class projection(nn.Module):
    def __init__(self, input_dim, output_dim, target_length, hidden_dim=256):
        super(projection, self).__init__()
        self.l1 = nn.Linear(int(target_length*input_dim), hidden_dim)  # input_dim = 512, output_dim = 64
        self.l2 = nn.Linear(hidden_dim, output_dim)  # input_dim = 512, output_dim = 64
        self.flatten = nn.Flatten()  # default 1 to -1. 0: batch
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.dropout(self.l1(x)))
        x = self.l2(x)
        return x


class GNNLLMFusion(nn.Module):
    """
    Deep fusion module that integrates LLM outputs with user and item GNN features.
    Uses multi-head attention and gating mechanisms for adaptive feature combination.
    """
    def __init__(self, llm_dim, gnn_dim, hidden_dim=128, num_heads=4, dropout=0.2):
        super(GNNLLMFusion, self).__init__()
        self.llm_dim = llm_dim
        self.gnn_dim = gnn_dim
        self.hidden_dim = hidden_dim
        
        # Project LLM and GNN features to common dimension
        self.llm_proj = nn.Linear(llm_dim, hidden_dim)
        self.user_gnn_proj = nn.Linear(gnn_dim, hidden_dim)
        self.item_gnn_proj = nn.Linear(gnn_dim, hidden_dim)
        
        # Multi-head cross-attention: LLM attends to user GNN features
        self.user_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Multi-head cross-attention: LLM attends to item GNN features
        self.item_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gating mechanism for adaptive fusion of user features
        self.gate_user = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Gating mechanism for adaptive fusion of item features
        self.gate_item = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Feature transformation layers (now takes user + item + llm features)
        self.fusion_transform = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, gnn_dim)
        )
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, llm_output, user_gnn_emb, item_gnn_emb=None):
        """
        Args:
            llm_output: [batch_size, llm_dim] - LLM predicted features
            user_gnn_emb: [batch_size, gnn_dim] - User GNN embeddings
            item_gnn_emb: [batch_size, gnn_dim] - Item GNN embeddings (optional for backward compatibility)
        Returns:
            fused_output: [batch_size, gnn_dim] - Fused prediction features
        """
        batch_size = llm_output.shape[0]
        
        # Project to common dimension
        llm_feat = self.llm_proj(llm_output)  # [batch, hidden_dim]
        user_feat = self.user_gnn_proj(user_gnn_emb)  # [batch, hidden_dim]
        
        # Add sequence dimension for attention
        llm_feat_seq = llm_feat.unsqueeze(1)  # [batch, 1, hidden_dim]
        user_feat_seq = user_feat.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Cross-attention: LLM features attend to user GNN features
        user_attn_output, _ = self.user_cross_attention(
            query=llm_feat_seq,
            key=user_feat_seq,
            value=user_feat_seq
        )
        user_attn_output = user_attn_output.squeeze(1)  # [batch, hidden_dim]
        
        # Adaptive gating for user features
        user_combined = torch.cat([llm_feat, user_attn_output], dim=-1)  # [batch, hidden_dim*2]
        user_gate = self.gate_user(user_combined)  # [batch, hidden_dim]
        user_gated_feat = user_gate * llm_feat + (1 - user_gate) * user_attn_output
        
        # If item GNN embeddings are provided, fuse them as well
        if item_gnn_emb is not None:
            item_feat = self.item_gnn_proj(item_gnn_emb)  # [batch, hidden_dim]
            item_feat_seq = item_feat.unsqueeze(1)  # [batch, 1, hidden_dim]
            
            # Cross-attention: LLM features attend to item GNN features
            item_attn_output, _ = self.item_cross_attention(
                query=llm_feat_seq,
                key=item_feat_seq,
                value=item_feat_seq
            )
            item_attn_output = item_attn_output.squeeze(1)  # [batch, hidden_dim]
            
            # Adaptive gating for item features
            item_combined = torch.cat([llm_feat, item_attn_output], dim=-1)  # [batch, hidden_dim*2]
            item_gate = self.gate_item(item_combined)  # [batch, hidden_dim]
            item_gated_feat = item_gate * llm_feat + (1 - item_gate) * item_attn_output
            
            # Combine all three: user gated, item gated, and original llm features
            fusion_input = torch.cat([user_gated_feat, item_gated_feat, llm_feat], dim=-1)  # [batch, hidden_dim*3]
        else:
            # Backward compatibility: only use user features if item features not provided
            fusion_input = torch.cat([user_gated_feat, user_feat, llm_feat], dim=-1)  # [batch, hidden_dim*3]
        
        # Transform to output dimension
        fused_output = self.fusion_transform(fusion_input)  # [batch, gnn_dim]
        
        # Residual connection with original LLM output (projected to gnn_dim if needed)
        if llm_output.shape[-1] == self.gnn_dim:
            fused_output = self.residual_weight * fused_output + (1 - self.residual_weight) * llm_output
        
        return fused_output
