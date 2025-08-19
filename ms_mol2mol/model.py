import os
import torch
import torch.nn as nn
import math


# ============================
# 4. Positional Encoding
# ============================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)






# ============================
# 5. Encoder 增强模块：局部卷积
# ============================
class EncoderWithConv(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super(EncoderWithConv, self).__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        # x: [seq_len, B, d_model]
        x = x.permute(1, 2, 0)  # [B, d_model, seq_len]
        x = self.conv(x)         # [B, d_model, seq_len]
        x = x.permute(2, 0, 1)     # [seq_len, B, d_model]
        return x






# ============================
# 6. 模型（Transformer Encoder-Decoder）
# ============================
class MoleculePretrainingModel(nn.Module):
    def __init__(self, vocab_size, atom_type_dim, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1):
        super(MoleculePretrainingModel, self).__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        # 采用卷积层增强局部特征，然后堆叠 Transformer Encoder
        self.encoder = nn.Sequential(
            EncoderWithConv(d_model),
            nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        )
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_linear = nn.Linear(d_model, vocab_size)
        
        # atom_type 投影
        self.atom_type_proj = nn.Linear(atom_type_dim, d_model)
        # 用于将 atom_type 信息作为全局特征与 decoder 初始状态融合
        self.decoder_init_proj = nn.Linear(d_model, d_model)
    
    def forward(self, src_seq, tgt_seq, atom_types, tgt_mask=None, memory_key_padding_mask=None):
        batch_size, src_len = src_seq.size()
        batch_size, tgt_len = tgt_seq.size()
        
        # Encoder
        src_emb = self.embedding(src_seq) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        src_emb = src_emb.transpose(0, 1)  # [src_len, B, d_model]
        memory = self.encoder(src_emb)       # [src_len, B, d_model]
        
        # Decoder
        tgt_emb = self.embedding(tgt_seq) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)  # [tgt_len, B, d_model]
        
        # 将 atom_type 作为全局特征，经过投影后作为 decoder 初始 token
        atom_emb = self.atom_type_proj(atom_types)   # [B, d_model]
        decoder_init = self.decoder_init_proj(atom_emb).unsqueeze(0)  # [1, B, d_model]
        # 将 decoder 初始状态拼接到 tgt_emb 前面
        tgt_emb = torch.cat([decoder_init, tgt_emb], dim=0)  # [1+tgt_len, B, d_model]
        
        # 生成 tgt_mask 时需考虑序列长度已增加1
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(0)).to(tgt_seq.device)
        
        decoder_output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                                      memory_key_padding_mask=memory_key_padding_mask)
        # 去掉第一个 token（decoder 初始状态），还原为 [tgt_len, B, d_model]
        decoder_output = decoder_output[1:, :, :].transpose(0, 1)  # [B, tgt_len, d_model]
        output_logits = self.output_linear(decoder_output)         # [B, tgt_len, vocab_size]
        return output_logits

