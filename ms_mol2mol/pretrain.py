"""
Molecular Pretraining Module

This module implements a transformer-based pretraining approach for molecular representation
learning using SMILES strings. It focuses on commonly used heavy atoms like C, N, O, F and uses masked language
modeling for self-supervised learning.

Author: ms_mol2mol Team
"""

import os
import re
import math
import random
import glob
import lmdb
import logging
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler, IterableDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Descriptors

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')


# ============================
# SMILES Tokenizer and Data Preprocessing (C, N, O, F atoms only)
# ============================
def tokenize_smiles(smiles: str) -> List[str]:
    """
    Tokenize SMILES string focusing on C, N, O, F atoms.
    
    This function uses regex pattern matching to extract chemical tokens,
    including bracketed atoms, ring closures, and chemical bonds.
    
    Args:
        smiles (str): SMILES string representation of a molecule
        
    Returns:
        List[str]: List of chemical tokens
    """
    pattern = r'''
        (\[[CNOF][^\]]*\]) |    # Match bracketed atoms like [C+], [N-] (C,N,O,F only)
        (%\d{2})         |      # Match two-digit ring closures like %12
        ([CNOF])        |       # Match single atom symbols C, N, O, F
        (\d)           |        # Match ring closure digits
        ([=#\-\+\(\)/\\])       # Match chemical bonds, parentheses and slashes
    '''
    tokens = re.findall(pattern, smiles, re.VERBOSE)
    # Extract non-empty parts from each match tuple
    token_list = [next(filter(None, t)).strip() for t in tokens if any(t)]
    return token_list


def smiles_to_indices(smiles: str, char2idx: Dict[str, int], max_length: int) -> List[int]:
    """
    Convert SMILES string to token indices.
    
    Adds <SOS> at the beginning and <EOS> at the end of the sequence.
    Truncates if too long (ensuring <EOS> is preserved) or pads with <PAD> if too short.
    
    Args:
        smiles (str): SMILES string
        char2idx (dict): Character to index mapping
        max_length (int): Maximum sequence length
        
    Returns:
        List[int]: List of token indices
    """
    tokens = tokenize_smiles(smiles)
    indices = [char2idx.get('<SOS>')]
    
    for token in tokens:
        if token in char2idx:
            indices.append(char2idx[token])
        else:
            indices.append(char2idx.get('<UNK>'))
    
    indices.append(char2idx.get('<EOS>'))
    
    if len(indices) < max_length:
        indices += [char2idx.get('<PAD>')] * (max_length - len(indices))
    else:
        indices = indices[:max_length-1] + [char2idx.get('<EOS>')]
    
    return indices


def mask_smiles_indices(indices: List[int], mask_token_idx: int, char2idx: Dict[str, int], 
                       mask_prob: float = 0.15) -> Tuple[List[int], List[bool]]:
    """
    Apply BERT-style masking strategy to token sequence.
    
    30% of selected tokens are replaced with <MASK>, 70% remain unchanged.
    Special tokens (<SOS>, <EOS>, <PAD>) are never masked.
    
    Args:
        indices (List[int]): Original token indices
        mask_token_idx (int): Index of <MASK> token
        char2idx (dict): Character to index mapping
        mask_prob (float): Probability of masking each token
        
    Returns:
        Tuple[List[int], List[bool]]: Masked indices and mask positions
    """
    special_tokens = {
        char2idx.get('<PAD>'), char2idx.get('<SOS>'), 
        char2idx.get('<EOS>'), char2idx.get('<UNK>')
    }
    
    masked_indices = indices.copy()
    mask_positions = [False] * len(indices)

    # Skip first and last tokens (SOS/EOS)
    for i in range(1, len(indices) - 1):
        if indices[i] in special_tokens:
            continue
            
        if random.random() < mask_prob:
            mask_positions[i] = True
            if random.random() < 0.3:  # 30% replace with <MASK>
                masked_indices[i] = mask_token_idx
            # 70% keep original token
    
    return masked_indices, mask_positions





# ============================
# Atomic Features Calculation (using RDKit)
# ============================

# Atomic weights for first five periods
ATOMIC_WEIGHTS = {
    "H": 1.007825, "He": 4.002603, "Li": 7.016004, "Be": 9.012182, "B": 11.009305,
    "C": 12.000000, "N": 14.003074, "O": 15.994915, "F": 18.998403, "Ne": 19.992440,
    "Na": 22.989770, "Mg": 23.985042, "Al": 26.981538, "Si": 27.976926, "P": 30.973762,
    "S": 31.972071, "Cl": 34.968853, "Ar": 39.962383, "K": 38.963707, "Ca": 39.962591,
    "Sc": 44.955911, "Ti": 47.947947, "V": 50.943964, "Cr": 51.940506, "Mn": 54.938045,
    "Fe": 55.934937, "Co": 58.933199, "Ni": 57.935346, "Cu": 62.929601, "Zn": 63.929142,
    "Ga": 68.925581, "Ge": 73.921178, "As": 74.921595, "Se": 79.916521, "Br": 78.918337,
    "Kr": 83.911507, "Rb": 84.911789, "Sr": 87.905612, "Y": 88.905848, "Zr": 89.904704,
    "Nb": 92.906378, "Mo": 97.905408, "Tc": 97.907212, "Ru": 101.904349, "Rh": 102.905504,
    "Pd": 105.903486, "Ag": 106.905097, "Cd": 105.906459, "In": 112.904061, "Sn": 111.904824,
    "Sb": 120.903815, "Te": 127.904461, "I": 126.904468, "Xe": 128.904779
}


def calculate_mol_weight_custom(mol) -> float:
    """
    Calculate molecular weight using custom precise atomic weights.
    
    If an atom is not in ATOMIC_WEIGHTS, it's skipped (only CNOF focus).
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        float: Molecular weight or None if mol is invalid
    """
    if mol is None:
        return None
    
    mol_weight = 0.0
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in ATOMIC_WEIGHTS:
            mol_weight += ATOMIC_WEIGHTS[symbol]
        # Skip atoms not in our target set
    
    return mol_weight


def calculate_dbe(mol) -> int:
    """
    Calculate Degree of Unsaturation (DBE).
    
    Formula: DBE = (2*C + 2 + N - (H + X)) / 2
    Where C, N, H, X are counts of carbon, nitrogen, hydrogen, and halogen atoms.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        int: DBE value
    """
    C = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    N = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
    H = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
    halogens = {'F', 'Cl', 'Br', 'I'}
    X = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in halogens)
    
    dbe = (2 * C + 2 + N - (H + X)) / 2
    return int(dbe)


def compute_atom_types(smiles: str) -> List[float]:
    """
    Compute molecular features including DBE, log-normalized molecular weight,
    and element counts for first five periods.
    
    Args:
        smiles (str): SMILES string
        
    Returns:
        List[float]: Feature vector of length 2 + len(ATOMIC_WEIGHTS)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return zero vector for invalid SMILES
        return [0.0] * (2 + len(ATOMIC_WEIGHTS))
    
    # Calculate DBE
    dbe = calculate_dbe(mol)
    
    # Calculate molecular weight with explicit hydrogens
    mol = Chem.AddHs(mol)
    mol_weight = round(calculate_mol_weight_custom(mol), 6)
    mol_weight = np.log1p(mol_weight) if mol_weight > 0 else 0.0
    
    # Count atoms for each element
    element_counts = []
    for elem in ATOMIC_WEIGHTS.keys():
        count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == elem)
        element_counts.append(count)
    
    return [float(dbe), mol_weight] + element_counts







# ============================
# In-memory Dataset
# ============================

class SMILESPretokenDataset(Dataset):
    """
    Dataset for SMILES pretraining with masked language modeling.
    
    Supports loading from either a single CSV file or a directory containing
    multiple CSV files. Each CSV file should have a 'SMILES' column.
    """
    
    def __init__(self, csv_path: str, char2idx: Dict[str, int], max_seq_length: int, 
                 compute_atom_types_fn, mask_prob: float = 1.0):
        """
        Initialize the dataset.
        
        Args:
            csv_path (str): Path to CSV file or directory containing CSV files
            char2idx (dict): Character to index mapping
            max_seq_length (int): Maximum sequence length (including special tokens)
            compute_atom_types_fn: Function to compute atom-type features from SMILES
            mask_prob (float): Probability of masking tokens
        """
        self.smiles_list = []
        
        # Load CSV files
        if os.path.isdir(csv_path):
            csv_files = sorted(glob.glob(os.path.join(csv_path, "*.csv")))
        else:
            csv_files = [csv_path]
            
        print(f"Found {len(csv_files)} CSV file(s). Loading...")
        
        for csv_file in tqdm(csv_files, desc="Loading CSV files"):
            df = pd.read_csv(csv_file)
            smiles = df["SMILES"].tolist()
            self.smiles_list.extend(smiles)
            
        print(f"Total molecules loaded: {len(self.smiles_list)}")
        
        # Store parameters
        self.mask_prob = mask_prob
        self.char2idx = char2idx
        self.max_seq_length = max_seq_length
        self.compute_atom_types = compute_atom_types_fn
        self.mask_token_idx = char2idx.get('<MASK>')
        
        # Assume all SMILES are valid (preprocessing should handle invalid ones)
        self.valid_indices = list(range(len(self.smiles_list)))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple containing:
                - src_indices: Masked token indices
                - true_indices: Original token indices  
                - atom_types: Molecular features
                - mask_positions: Boolean mask indicating masked positions
        """
        real_idx = self.valid_indices[idx]
        smiles = self.smiles_list[real_idx]
        
        # Convert SMILES to indices
        true_indices = smiles_to_indices(smiles, self.char2idx, self.max_seq_length)
        
        # Apply masking
        src_indices, mask_positions = mask_smiles_indices(
            true_indices, self.mask_token_idx, self.char2idx, mask_prob=self.mask_prob
        )
        
        # Convert to tensors
        src_indices = torch.tensor(src_indices, dtype=torch.long)
        true_indices = torch.tensor(true_indices, dtype=torch.long)
        mask_positions = torch.tensor(mask_positions, dtype=torch.bool)
        
        # Compute atom-type features
        atom_types = self.compute_atom_types(smiles)
        atom_types = torch.tensor(atom_types, dtype=torch.float32)
        
        return src_indices, true_indices, atom_types, mask_positions


    




# ============================
# Positional Encoding
# ============================

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    Adds sinusoidal position encodings to input embeddings to provide
    sequence position information.
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
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ============================
# Encoder Enhancement Module: Local Convolution
# ============================

class EncoderWithConv(nn.Module):
    """
    Convolutional layer for local feature enhancement in transformer encoder.
    
    Applies 1D convolution to capture local dependencies before transformer processing.
    """
    
    def __init__(self, d_model: int, kernel_size: int = 3):
        super(EncoderWithConv, self).__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch_size, d_model]
        x = x.permute(1, 2, 0)  # [batch_size, d_model, seq_len]
        x = self.conv(x)        # [batch_size, d_model, seq_len]
        x = x.permute(2, 0, 1)  # [seq_len, batch_size, d_model]
        return x






# ============================
# Model (Transformer Encoder-Decoder)
# ============================

class CustomTransformerDecoderLayer(nn.Module):
    """
    Custom transformer decoder layer with explicit attention mechanisms.
    
    Implements self-attention, cross-attention, and feedforward components
    with residual connections and layer normalization.
    """
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 512, dropout: float = 0.1):
        super(CustomTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, **kwargs) -> torch.Tensor:
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


class MoleculePretrainingModel(nn.Module):
    """
    Transformer-based model for molecular pretraining.
    
    Uses encoder-decoder architecture with atom-type features integration
    for SMILES masked language modeling.
    """
    
    def __init__(self, vocab_size: int, atom_type_dim: int, d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6, 
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super(MoleculePretrainingModel, self).__init__()
        self.d_model = d_model
        
        # Embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Encoder with local convolution enhancement
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
                tgt_mask=None, memory_key_padding_mask=None) -> torch.Tensor:
        """
        Forward pass for masked language modeling.
        
        Args:
            src_seq: Source sequence (masked SMILES)
            tgt_seq: Target sequence (for teacher forcing)
            atom_types: Molecular features
            tgt_mask: Target attention mask
            memory_key_padding_mask: Memory padding mask
            
        Returns:
            torch.Tensor: Output logits for vocabulary prediction
        """
        batch_size, src_len = src_seq.size()
        batch_size, tgt_len = tgt_seq.size()
        
        # Encoder
        src_emb = self.embedding(src_seq) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        src_emb = src_emb.transpose(0, 1)  # [src_len, batch_size, d_model]
        memory = self.encoder(src_emb)     # [src_len, batch_size, d_model]
        
        # Decoder
        tgt_emb = self.embedding(tgt_seq) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)  # [tgt_len, batch_size, d_model]
        
        # Integrate atom-type features as decoder initialization
        atom_emb = self.atom_type_proj(atom_types)  # [batch_size, d_model]
        decoder_init = self.decoder_init_proj(atom_emb).unsqueeze(0)  # [1, batch_size, d_model]
        
        # Concatenate decoder init with target embeddings
        tgt_emb = torch.cat([decoder_init, tgt_emb], dim=0)  # [1+tgt_len, batch_size, d_model]
        
        # Generate causal mask for decoder
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





# ============================
# SMILES Vocabulary and Mappings (C, N, O, F atoms only)
# ============================

SMILES_VOCAB = [
    # Special tokens
    '<PAD>', '<SOS>', '<EOS>', '<UNK>', '<MASK>',
    
    # Basic atom symbols
    'C', 'N', 'O', 'F',
    
    # Charged atom forms
    '[C]', '[CH]', '[CH2]', '[CH3]', 
    '[N+]', '[N-]', '[NH+]', '[NH2+]', '[NH3+]',
    '[O-]', '[OH+]',
    
    # Chemical symbols
    '(', ')', '[', ']', '=', '#', '-', '+', '/', '\\',
    
    # Ring closure markers (two digits)
    *[f'%{i}' for i in range(10, 100)],
    
    # Single digits (0-9)
    *[str(i) for i in range(10)],

    # Common isotope markers
    '[13C]', '[14C]', '[15N]'
]

vocab_size = len(SMILES_VOCAB)
char2idx = {token: idx for idx, token in enumerate(SMILES_VOCAB)}
idx2char = {idx: token for idx, token in enumerate(SMILES_VOCAB)}


def decode_indices(indices: List[int], idx2char: Dict[int, str]) -> str:
    """
    Decode token indices back to SMILES string.
    
    Stops at <EOS> token and skips <SOS> and <PAD> tokens.
    
    Args:
        indices (List[int]): Token indices
        idx2char (dict): Index to character mapping
        
    Returns:
        str: Decoded SMILES string
    """
    tokens = []
    for idx in indices:
        token = idx2char.get(idx, '')
        if token == '<EOS>':
            break
        if token in ['<SOS>', '<PAD>']:
            continue
        tokens.append(token)
    return ''.join(tokens)







# ============================
# Training Functions (DDP + Learning Rate Warmup + Dynamic Gradient Clipping)
# ============================

def setup_logger(log_file: str) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(fh)
    
    return logger


def get_mask_prob(epoch: int, total_epochs: int, start_ratio: float = 0.0, 
                  end_ratio: float = 1.0, power: float = 0.5) -> float:
    """
    Calculate mask probability for current epoch using non-linear progression.

    Uses a power function to control the rate of masking increase over epochs.
    Lower power values result in faster initial increase.

    Args:
        epoch (int): Current epoch (0-indexed)
        total_epochs (int): Total number of epochs
        start_ratio (float): Initial mask probability
        end_ratio (float): Final mask probability  
        power (float): Power parameter controlling progression rate

    Returns:
        float: Mask probability for current epoch
        
    Example:
        For total_epochs=10:
        - epoch=0: mask_ratio = 0
        - epoch=2: mask_ratio ≈ 0.47 (with power=0.5)
        - epoch=9: mask_ratio = 1.0
    """
    if total_epochs <= 1:
        return end_ratio
    
    # Calculate progression ratio
    ratio = (epoch / (total_epochs - 1)) ** power
    return start_ratio + (end_ratio - start_ratio) * ratio


def train(rank: int, world_size: int, csv_folder: str, num_epochs: int = 10, 
          batch_size: int = 128, max_seq_length: int = 300):
    """
    Main training function with distributed data parallel support.
    
    Args:
        rank (int): Process rank for distributed training
        world_size (int): Total number of processes
        csv_folder (str): Directory containing CSV files (one per epoch)
        num_epochs (int): Number of training epochs
        batch_size (int): Training batch size
        max_seq_length (int): Maximum sequence length
    """
    # Setup distributed training environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10314'
    
    # Initialize distributed process group
    dist.init_process_group(backend='gloo', init_method='env://', 
                           world_size=world_size, rank=rank)
    torch.manual_seed(42)
    device = torch.device('cuda', rank)
    
    # Initialize logging and monitoring (rank 0 only)
    if rank == 0:
        writer = SummaryWriter(log_dir="runs/exp")
        logger = setup_logger("training.log")
        logger.info("Training started.")
    
    # Get CSV files (expect one file per epoch)
    csv_files = sorted(glob.glob(os.path.join(csv_folder, "*.csv")))
    assert len(csv_files) == num_epochs, f"Expected {num_epochs} CSV files, found {len(csv_files)}"
    
    # Calculate total training steps for learning rate scheduling
    total_steps = 0
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        num_samples = len(df)
        num_batches = math.ceil(num_samples / batch_size)
        total_steps += num_batches

    global_step = 0

    # Initialize model
    atom_type_dim = len(compute_atom_types("C"))
    print(f"Atom type dimension: {atom_type_dim}")
    
    model = MoleculePretrainingModel(
        vocab_size=vocab_size, 
        atom_type_dim=atom_type_dim, 
        d_model=512, 
        nhead=8,
        num_encoder_layers=6, 
        num_decoder_layers=6,
        dim_feedforward=2048, 
        dropout=0.1
    )
    
    # Load pretrained weights if available
    checkpoint_path = "./512_molecule_pretraining_model_epoch4.pth"
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Remove 'module.' prefix from DDP state dict
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Rank {rank}: Loaded pretrained weights from {checkpoint_path}")
    else:
        print(f"Rank {rank}: No checkpoint found at {checkpoint_path}")

    # Setup model for distributed training
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    
    # Initialize optimizer and scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx.get('<PAD>'), reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    def lr_lambda(step):
        """Learning rate schedule with warmup and cosine decay."""
        warmup_steps = 10000
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / 
                                      max(1, (total_steps - warmup_steps))))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Training loop - each epoch uses a different CSV file
    for epoch in range(4, num_epochs):  # Resume from epoch 4
        current_csv_file = csv_files[epoch]
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs} using CSV file: {current_csv_file}")
        
        # Calculate dynamic mask probability
        current_mask_prob = get_mask_prob(epoch, num_epochs, start_ratio=0.0, 
                                         end_ratio=1.0, power=0.7)

        # Initialize dataset and dataloader for current epoch
        dataset = SMILESPretokenDataset(
            current_csv_file, char2idx, max_seq_length, 
            compute_atom_types, mask_prob=current_mask_prob
        )
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                               num_workers=4, pin_memory=True)
        
        # Set epoch for sampler to ensure proper shuffling
        sampler.set_epoch(epoch)
        
        # Training loop for current epoch
        total_loss = 0.0
        if rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)
        else:
            pbar = dataloader
        
        for i, (src_seq, tgt_seq, atom_types, mask_positions) in enumerate(pbar):
            # Move data to device
            src_seq = src_seq.to(device)
            tgt_seq = tgt_seq.to(device)
            atom_types = atom_types.to(device)
            mask_positions = mask_positions.to(device)
            
            # Prepare teacher forcing sequences
            tgt_input = tgt_seq[:, :-1]  # Remove last token for input
            tgt_output = tgt_seq[:, 1:]  # Remove first token for output
            mask_positions_target = mask_positions[:, 1:]  # Align with target
            
            # Generate causal mask (accounting for prepended atom feature)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_input.size(1) + 1).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output_logits = model(src_seq, tgt_input, atom_types, tgt_mask=tgt_mask)
            
            # Calculate loss
            loss_tensor = criterion(output_logits.reshape(-1, vocab_size), 
                                   tgt_output.reshape(-1))
            loss = torch.mean(loss_tensor)
            
            # Backward pass with dynamic gradient clipping
            loss.backward()
            current_max_norm = 1.0 - 0.9 * (global_step / total_steps)
            current_max_norm = max(0.1, current_max_norm)  # Minimum clip value
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=current_max_norm)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            global_step += 1
            
            # Logging and monitoring (rank 0 only)
            if rank == 0:
                writer.add_scalar("Loss/iteration", loss.item(), global_step)
                writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], global_step)
                pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                
                # Log sample predictions every 100 steps
                if (i + 1) % 100 == 0:
                    pred_tokens = torch.argmax(output_logits[0], dim=-1).cpu().tolist()
                    true_tokens = tgt_output[0].cpu().tolist()
                    mask_tokens = src_seq[0].cpu().tolist()
                    mask_positions_list = mask_positions[0].cpu().tolist()

                    # Decode tokens to SMILES strings
                    pred_smiles = decode_indices(pred_tokens, idx2char)
                    true_smiles = decode_indices(true_tokens, idx2char)
                    mask_smiles = decode_indices(mask_tokens, idx2char)

                    # Show which tokens were masked
                    masked_true_tokens = [idx2char[t] for t, m in zip(true_tokens, mask_positions_list) if m]

                    log_msg = (
                        f"Step {global_step}: Loss {loss.item():.4f}\n"
                        f"True SMILES   : {true_smiles}\n"
                        f"Masked SMILES : {mask_smiles}\n"
                        f"Masked Tokens : {masked_true_tokens}\n"
                        f"Pred SMILES   : {pred_smiles}"
                    )
                    print("\n" + log_msg)
                    logger.info(log_msg)
                    writer.add_text("Predictions", log_msg, global_step)

        # End of epoch logging and model saving
        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            epoch_msg = f"Epoch [{epoch+1}/{num_epochs}] finished, Avg Loss: {avg_loss:.4f}"
            print(epoch_msg)
            logger.info(epoch_msg)
            writer.add_scalar("Loss/epoch", avg_loss, epoch)
            
            # Save model checkpoint
            model_path = f"512_molecule_pretraining_model_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
            writer.add_text("Model Save", f"Model saved to {model_path}", epoch)
            writer.flush()
    
    # Cleanup
    if rank == 0:
        writer.close()
    dist.destroy_process_group()


# ============================
# Distributed Training Entry Point
# ============================

def main():
    """Main entry point for distributed training."""
    csv_folder = "./zinc20_data"  # Directory containing CSV files (10 files)
    world_size = 8  # Number of GPUs
    
    mp.spawn(train, args=(world_size, csv_folder, 10, 128, 300), 
             nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

