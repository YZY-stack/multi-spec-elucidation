"""
Model definitions for multi-spectral molecular property prediction and SMILES generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from rdkit import Chem
from rdkit.Chem import AllChem


class SMILESLoss(nn.Module):
    """Loss function for SMILES sequence generation."""
    
    def __init__(self, ignore_index):
        super(SMILESLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, output, target):
        """
        Forward pass for SMILES loss calculation.
        
        Args:
            output: Model output [seq_len, batch_size, vocab_size]
            target: Target sequences [seq_len, batch_size]
            
        Returns:
            Computed loss value
        """
        output = output.reshape(-1, output.size(-1))  # [seq_len * batch_size, vocab_size]
        target = target.reshape(-1)  # [seq_len * batch_size]
        loss = self.loss_fn(output, target)
        return loss


class CustomTransformerEncoder(nn.Module):
    """Custom transformer encoder with attention weight output."""
    
    def __init__(self, encoder_layer, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Forward pass through transformer encoder layers.
        
        Args:
            src: Source input tensor
            mask: Attention mask
            src_key_padding_mask: Key padding mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        output = src
        attentions = []  # Store attention weights from each layer

        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attentions.append(attn_weights)

        return output, attentions


class CustomTransformerEncoderLayer(nn.Module):
    """Custom transformer encoder layer with attention weight output."""
    
    def __init__(self, d_model, nhead, **kwargs):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, **kwargs)
        # Feedforward network components
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src shape: [seq_len, batch_size, d_model]
        # Self-attention
        attn_output, attn_weights = self.self_attn(
            src, src, src, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask, need_weights=True
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feedforward network
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src, attn_weights




class AtomPredictionModel(nn.Module):
    """
    Multi-modal spectral data to molecular structure prediction model.
    
    This model takes various spectral inputs (IR, UV, NMR, Mass Spec) and predicts
    molecular structures represented as SMILES strings. It uses a transformer-based
    architecture with custom tokenization for spectral data.
    
    Args:
        vocab_size (int): Size of SMILES vocabulary
        count_tasks_classes (dict): Dictionary of count-based auxiliary tasks and their class counts
        binary_tasks (list): List of binary auxiliary task names
    """
    
    def __init__(self, vocab_size, count_tasks_classes, binary_tasks):
        super(AtomPredictionModel, self).__init__()
        self.d_model = 128  # Embedding dimension
        self.num_spectra = 8  # Number of different spectral modalities

        # Define feature dimensions for each spectral modality
        feature_dims = {
            'ir': 82,              # Infrared spectrum features
            'uv': 10,              # UV-Vis spectrum features  
            'nmr_c': 207,          # Carbon-13 NMR features (28+180)
            'nmr_h': 382,          # Proton NMR features (1D + HSQC + COSY + J2D)
            'f_spectrum': 12,      # Fluorine spectrum features
            'n_spectrum': 14,      # Nitrogen spectrum features
            'o_spectrum': 12,      # Oxygen spectrum features
            'mass_high': 7         # High-resolution mass spectrum features
        }

        # Initialize components
        max_feature_positions = max(feature_dims.values())  # Maximum number of features per spectrum
        self.tokenizer = Tokenizer(feature_dims, self.d_model, max_feature_positions, self.num_spectra)

        # Custom Transformer encoder for spectral feature processing
        encoder_layer = CustomTransformerEncoderLayer(d_model=self.d_model, nhead=1)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers=1)

        # SMILES decoder for molecular structure generation
        self.smiles_decoder = SMILESEncoderDecoder(d_model=self.d_model, vocab_size=vocab_size, num_atom_types=5)




        # # 计数任务的输出头
        # self.count_task_heads = nn.ModuleDict()
        # for task, num_classes in count_tasks_classes.items():
        #     self.count_task_heads[task] = nn.Sequential(
        #         nn.Linear(self.d_model, num_classes),
        #         # nn.ReLU(),
        #         # nn.Linear(64, 128),
        #         # nn.ReLU(),
        #         # nn.Linear(128, num_classes)  # 输出类别数的 logits
        #     )

        # # 二元分类任务的输出头
        # self.binary_task_heads = nn.ModuleDict()
        # for task in binary_tasks:
        #     self.binary_task_heads[task] = nn.Sequential(
        #         nn.Linear(self.d_model, 1),
        #         # nn.ReLU(),
        #         # nn.Linear(64, 128),
        #         # nn.ReLU(),
        #         # nn.Linear(128, 1)  # 输出单个 logit
        #     )


    def forward(self, ir_spectrum, uv_spectrum, c_spectrum, h_spectrum,
                high_res_mass, tgt_seq, tgt_mask=None,
                atom_types=None):
        # Split h_spectrum into h_spectrum_part, f_spectrum, n_spectrum
        # ## All wo J
        # h_spectrum_part = h_spectrum[:, :300]
        # f_spectrum = h_spectrum[:, 300:312]
        # n_spectrum = h_spectrum[:, 312:326]
        # o_spectrum = h_spectrum[:, 326:]

        # # with J
        # h_spectrum_part = h_spectrum[:, :490]
        # f_spectrum = h_spectrum[:, 490:502]
        # n_spectrum = h_spectrum[:, 502:516]
        # o_spectrum = h_spectrum[:, 516:]


        # # 1D NMR only
        # h_spectrum_part = h_spectrum[:, :105]
        # f_spectrum = h_spectrum[:, 105:117]
        # n_spectrum = h_spectrum[:, 117:131]
        # o_spectrum = h_spectrum[:, 131:]

        # # 1D NMR only with HSQC
        # h_spectrum_part = h_spectrum[:, :225]
        # f_spectrum = h_spectrum[:, 225:237]
        # n_spectrum = h_spectrum[:, 237:251]
        # o_spectrum = h_spectrum[:, 251:]

        # # 1D NMR only with HSQC with COSY
        # h_spectrum_part = h_spectrum[:, :405]
        # f_spectrum = h_spectrum[:, 405:417]
        # n_spectrum = h_spectrum[:, 417:431]
        # o_spectrum = h_spectrum[:, 431:]

        # 1D NMR only with HSQC with COSY with J2D
        h_spectrum_part = h_spectrum[:, :382]
        f_spectrum = h_spectrum[:, 382:394]
        n_spectrum = h_spectrum[:, 394:408]
        o_spectrum = h_spectrum[:, 408:]
        

        # Prepare features
        features = {
            'ir': ir_spectrum,
            'uv': uv_spectrum,
            'nmr_c': c_spectrum,
            'nmr_h': h_spectrum_part,
            'f_spectrum': f_spectrum,
            'n_spectrum': n_spectrum,
            'o_spectrum': o_spectrum,
            'mass_high': high_res_mass
        }

        # Tokenize features
        tokens = self.tokenizer(features)  # [batch_size, total_N_features, d_model]

        # Permute for transformer input: [seq_len, batch_size, d_model]
        tokens = tokens.permute(1, 0, 2)

        # Apply transformer encoder and get attention weights
        memory, attentions = self.transformer_encoder(tokens)  # memory: [seq_len, batch_size, d_model]

        # Optionally, pool the memory to get fusion feature (e.g., mean pooling)
        fusion_feat = memory.mean(dim=0)  # [batch_size, d_model]

        # Decode from spectral features
        output = self.smiles_decoder(tgt_seq, memory, tgt_mask, atom_types=atom_types)





        # # 辅助任务预测
        # count_task_outputs = {}
        # for task, head in self.count_task_heads.items():
        #     logits = head(fusion_feat)  # [batch_size, num_classes]
        #     count_task_outputs[task] = logits

        # binary_task_outputs = {}
        # for task, head in self.binary_task_heads.items():
        #     logit = head(fusion_feat).squeeze(1)  # [batch_size]
        #     binary_task_outputs[task] = logit

        # return output, attentions, fusion_feat, count_task_outputs, binary_task_outputs



        return output, attentions, fusion_feat, None, None  # Return attentions along with the output





class Tokenizer(nn.Module):
    def __init__(self, feature_dims, d_model, max_feature_positions, num_spectra):
        """
        Args:
            feature_dims (dict): Mapping spectrum name to number of features.
            d_model (int): Embedding dimension.
            max_feature_positions (int): Maximum number of features across all spectra.
            num_spectra (int): Total number of different spectra.
        """
        super(Tokenizer, self).__init__()
        self.d_model = d_model
        self.feature_embeddings = nn.ModuleDict()
        self.position_embeddings = nn.Embedding(max_feature_positions, d_model)
        self.spectrum_embeddings = nn.Embedding(num_spectra, d_model)

        self.spectrum_names = list(feature_dims.keys())
        for spectrum_name in self.spectrum_names:
            # For scalar features, use Linear(1, d_model)
            self.feature_embeddings[spectrum_name] = nn.Linear(1, d_model)

        self.num_spectra = num_spectra
        self.token_labels = []  # To store labels for each token

    def forward(self, features):
        """
        Args:
            features (dict): Mapping spectrum name to feature tensor of shape [batch_size, N_features].
        
        Returns:
            tokens (Tensor): Token embeddings of shape [batch_size, total_N_features, d_model].
        """
        batch_size = next(iter(features.values())).size(0)
        self.token_labels = []  # Reset labels
        tokens_list = []
        for i, spectrum_name in enumerate(self.spectrum_names):
            spectrum_features = features[spectrum_name]  # Shape: [batch_size, N_features]
            N_features = spectrum_features.size(1)
            # Reshape to [batch_size, N_features, 1]
            spectrum_features = spectrum_features.unsqueeze(-1)  # Shape: [batch_size, N_features, 1]
            # Get feature embeddings
            feature_embeds = self.feature_embeddings[spectrum_name](spectrum_features)  # Shape: [batch_size, N_features, d_model]
            # Get feature positions
            feature_positions = torch.arange(N_features, device=spectrum_features.device).unsqueeze(0).repeat(batch_size, 1)  # Shape: [batch_size, N_features]
            position_embeds = self.position_embeddings(feature_positions)  # Shape: [batch_size, N_features, d_model]
            # Get spectrum embeddings
            spectrum_index = torch.full((batch_size, N_features), i, device=spectrum_features.device, dtype=torch.long)
            spectrum_embeds = self.spectrum_embeddings(spectrum_index)  # Shape: [batch_size, N_features, d_model]
            # Sum embeddings
            embeds = feature_embeds + position_embeds + spectrum_embeds  # Shape: [batch_size, N_features, d_model]
            # Append to tokens
            tokens_list.append(embeds)
            # Create labels for each token
            for pos in range(N_features):
                label = f"{spectrum_name}_{pos}"
                self.token_labels.append(label)
        # Concatenate all tokens along the N_features dimension
        tokens = torch.cat(tokens_list, dim=1)  # Shape: [batch_size, total_N_features, d_model]
        return tokens

    def get_token_labels(self):
        """
        Returns:
            token_labels (list): A list of labels corresponding to each token.
        """
        return self.token_labels



    

    # def forward(self, ir_spectrum, uv_spectrum, c_spectrum, h_spectrum,
    #             low_res_mass, high_res_mass, tgt_seq, tgt_mask=None):
    #     # 编码器部分
    #     spectra_feat = self.spectra_encoder(ir_spectrum, uv_spectrum)
    #     nmr_feat = self.nmr_encoder(c_spectrum, h_spectrum)
    #     # mass_feat = self.mass_spectrum_encoder(low_res_mass, high_res_mass)
    #     # fusion_feat = self.fusion_encoder(spectra_feat, nmr_feat, mass_feat)  # [batch_size, d_model]
    #     fusion_feat = self.fusion_encoder(spectra_feat, nmr_feat)  # [batch_size, d_model]

    #     # 将融合特征作为 memory，并添加序列维度
    #     memory = fusion_feat.unsqueeze(0)  # [1, batch_size, d_model]
    #     # memory = spectra_feat.unsqueeze(0)  # [1, batch_size, d_model]

    #     # 解码器部分
    #     output = self.smiles_decoder(tgt_seq, memory, tgt_mask)

    #     # 辅助任务预测
    #     count_task_outputs = {}
    #     for task, head in self.count_task_heads.items():
    #         logits = head(spectra_feat)  # [batch_size, num_classes]
    #         count_task_outputs[task] = logits

    #     binary_task_outputs = {}
    #     for task, head in self.binary_task_heads.items():
    #         logit = head(spectra_feat).squeeze(1)  # [batch_size]
    #         binary_task_outputs[task] = logit

    #     return output, fusion_feat, count_task_outputs, binary_task_outputs




import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model//2]
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)






class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(CustomTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, **kwargs):
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt





class SMILESEncoderDecoder(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=2, dim_feedforward=512,
                 max_seq_length=100, vocab_size=None, num_atom_types=5):
        super(SMILESEncoderDecoder, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_length)
        decoder_layer = CustomTransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Linear layer for atom types
        self.atom_types_linear = nn.Linear(num_atom_types, d_model)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, tgt, memory, tgt_mask=None, atom_types=None):
        """
        tgt: [seq_len, batch_size]
        memory: [memory_seq_len, batch_size, d_model]
        atom_types: [batch_size, num_atom_types]
        """
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)  # [seq_len, batch_size, d_model]

        # 处理 atom_types
        if atom_types is not None:
            atom_types_feat = self.atom_types_linear(atom_types)  # [batch_size, d_model]
            atom_types_feat = torch.relu(atom_types_feat)
            atom_types_feat = atom_types_feat.unsqueeze(0)  # [1, batch_size, d_model]

            # 将 atom_types_feat 扩展到与 tgt_emb 相同的时间维度
            atom_types_feat = atom_types_feat.expand(tgt_emb.size(0), -1, -1)  # [seq_len, batch_size, d_model]

            # 将 atom_types_feat 与 tgt_emb 结合
            tgt_emb = tgt_emb + atom_types_feat

        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        """生成下三角掩码，用于屏蔽序列中未来的信息"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

