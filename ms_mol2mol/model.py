"""
Molecular Model Architecture

This module contains the core model components for molecular representation learning,
including positional encoding, convolutional enhancement, and transformer-based
encoder-decoder architectures.

Author: ms_mol2mol Team
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    Adds sinusoidal position encodings to input embeddings to provide
    sequence position information to the model.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to input embeddings."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EncoderWithConv(nn.Module):
    """
    Convolutional enhancement module for transformer encoder.
    
    Applies 1D convolution to capture local dependencies before
    transformer processing, improving sequence modeling capabilities.
    """
    
    def __init__(self, d_model: int, kernel_size: int = 3):
        super(EncoderWithConv, self).__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolutional processing to input sequence.
        
        Args:
            x: Input tensor of shape [seq_len, batch_size, d_model]
            
        Returns:
            torch.Tensor: Processed tensor with same shape
        """
        # x: [seq_len, batch_size, d_model]
        x = x.permute(1, 2, 0)  # [batch_size, d_model, seq_len]
        x = self.conv(x)        # [batch_size, d_model, seq_len]
        x = x.permute(2, 0, 1)  # [seq_len, batch_size, d_model]
        return x






class MoleculePretrainingModel(nn.Module):
    """
    Transformer-based model for molecular pretraining.
    
    This model uses an encoder-decoder architecture with atom-type features
    integration for SMILES masked language modeling tasks.
    """
    
    def __init__(self, vocab_size: int, atom_type_dim: int, d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6, 
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super(MoleculePretrainingModel, self).__init__()
        self.d_model = d_model
        
        # Embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Encoder with convolutional enhancement
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.Sequential(
            EncoderWithConv(d_model),
            nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        )
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output projection
        self.output_linear = nn.Linear(d_model, vocab_size)
        
        # Atom-type feature integration
        self.atom_type_proj = nn.Linear(atom_type_dim, d_model)
        self.decoder_init_proj = nn.Linear(d_model, d_model)
    
    def forward(self, src_seq: torch.Tensor, tgt_seq: torch.Tensor, atom_types: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None, 
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for masked language modeling.
        
        Args:
            src_seq: Source sequence (masked SMILES) [batch_size, src_len]
            tgt_seq: Target sequence for teacher forcing [batch_size, tgt_len]  
            atom_types: Molecular features [batch_size, atom_type_dim]
            tgt_mask: Target attention mask
            memory_key_padding_mask: Memory padding mask
            
        Returns:
            torch.Tensor: Output logits [batch_size, tgt_len, vocab_size]
        """
        batch_size, src_len = src_seq.size()
        batch_size, tgt_len = tgt_seq.size()
        
        # Encoder processing
        src_emb = self.embedding(src_seq) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        src_emb = src_emb.transpose(0, 1)  # [src_len, batch_size, d_model]
        memory = self.encoder(src_emb)     # [src_len, batch_size, d_model]
        
        # Decoder processing
        tgt_emb = self.embedding(tgt_seq) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)  # [tgt_len, batch_size, d_model]
        
        # Integrate atom-type features as decoder initialization
        atom_emb = self.atom_type_proj(atom_types)  # [batch_size, d_model]
        decoder_init = self.decoder_init_proj(atom_emb).unsqueeze(0)  # [1, batch_size, d_model]
        
        # Prepend decoder initialization to target embeddings
        tgt_emb = torch.cat([decoder_init, tgt_emb], dim=0)  # [1+tgt_len, batch_size, d_model]
        
        # Generate causal mask for decoder (accounting for prepended feature)
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(0)).to(tgt_seq.device)
        
        # Decoder forward pass
        decoder_output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                                      memory_key_padding_mask=memory_key_padding_mask)
        
        # Remove initialization token and transpose back
        decoder_output = decoder_output[1:, :, :].transpose(0, 1)  # [batch_size, tgt_len, d_model]
        
        # Project to vocabulary
        output_logits = self.output_linear(decoder_output)  # [batch_size, tgt_len, vocab_size]
        
        return output_logits

