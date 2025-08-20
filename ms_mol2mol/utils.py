"""
Molecular Utilities Module

This module provides utility functions for SMILES tokenization, data processing,
and molecular feature extraction for the ms_mol2mol framework.

Author: ms_mol2mol Team
"""

import os
import math
import random
import re
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import numpy as np


# ============================
# SMILES Processing Utilities
# ============================

def tokenize_smiles(smiles: str) -> List[str]:
    """
    Tokenize SMILES string focusing on C, N, O, F atoms.
    
    Uses regex pattern matching to extract chemical tokens including
    bracketed atoms, ring closures, and chemical bonds.
    
    Args:
        smiles (str): SMILES string representation
        
    Returns:
        List[str]: List of chemical tokens
    """
    pattern = r'''
        (\[[CNOF][^\]]*\]) |    # Match bracketed atoms like [C+], [N-] (C,N,O,F only)
        (%\d{2})         |      # Match two-digit ring closures like %12
        ([CNOF])        |       # Match single atom symbols C, N, O, F
        (\d+)           |       # Match ring closure numbers
        ([=#\-\+\(\)/\\])       # Match chemical bonds, parentheses and slashes
    '''
    tokens = re.findall(pattern, smiles, re.VERBOSE)
    # Extract non-empty parts from each match tuple
    token_list = [next(filter(None, t)).strip() for t in tokens if any(t)]
    return token_list


def smiles_to_indices(smiles: str, char2idx: Dict[str, int], max_length: int) -> List[int]:
    """
    Convert SMILES string to token indices.
    
    Adds <SOS> at the beginning and <EOS> at the end.
    Truncates if too long (preserving <EOS>) or pads with <PAD> if too short.
    
    Args:
        smiles (str): SMILES string
        char2idx (dict): Character to index mapping
        max_length (int): Maximum sequence length
        
    Returns:
        List[int]: Token indices
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






def mask_smiles_indices(indices: List[int], mask_token_idx: int, char2idx: Dict[str, int], mask_prob: float = 0.15) -> Tuple[List[int], List[bool]]:
    """
    Apply BERT-style masking strategy to token sequences.
    
    Masking strategy:
      - 30% replaced with <MASK> token
      - 70% kept unchanged
    Returns mask_positions (bool type, True indicates position is masked).
    Special tokens (<SOS>, <EOS>, <PAD>) and beginning/end positions are not masked.
    
    Args:
        indices (List[int]): Original token indices
        mask_token_idx (int): Index of the <MASK> token
        char2idx (Dict[str, int]): Character to index mapping
        mask_prob (float): Probability of masking each token
        
    Returns:
        Tuple[List[int], List[bool]]: Masked indices and mask positions
    """
    pad_token = char2idx.get('<PAD>')
    sos_token = char2idx.get('<SOS>')
    eos_token = char2idx.get('<EOS>')
    
    masked_indices = indices.copy()
    mask_positions = [False] * len(indices)
    
    for i in range(1, len(indices)-1):
        # Skip special tokens
        if indices[i] in [pad_token, sos_token, eos_token]:
            continue
        if random.random() < mask_prob:
            mask_positions[i] = True
            if random.random() < 0.3:
                masked_indices[i] = mask_token_idx
            # 70% keep unchanged
    return masked_indices, mask_positions









def decode_indices(indices: List[int], idx2char: Dict[int, str]) -> str:
    """
    Decode index sequence to SMILES string, stopping at <EOS>.
    Skips <SOS> and <PAD> tokens.
    
    Args:
        indices (List[int]): Token indices
        idx2char (Dict[int, str]): Index to character mapping
        
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
# Molecular Feature Calculation (using rdkit)
# ============================
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:
    print("Warning: RDKit not found. Some functions may not work properly.")

# Atomic weights for first five periods of periodic table
atomic_weights = {
    "H": 1.007825,
    "He": 4.002603,
    "Li": 7.016004,
    "Be": 9.012182,
    "B": 11.009305,
    "C": 12.000000,
    "N": 14.003074,
    "O": 15.994915,
    "F": 18.998403,
    "Ne": 19.992440,
    "Na": 22.989770,
    "Mg": 23.985042,
    "Al": 26.981538,
    "Si": 27.976926,
    "P": 30.973762,
    "S": 31.972071,
    "Cl": 34.968853,
    "Ar": 39.962383,
    "K": 38.963707,
    "Ca": 39.962591,
    "Sc": 44.955911,
    "Ti": 47.947947,
    "V": 50.943964,
    "Cr": 51.940506,
    "Mn": 54.938045,
    "Fe": 55.934937,
    "Co": 58.933199,
    "Ni": 57.935346,
    "Cu": 62.929601,
    "Zn": 63.929142,
    "Ga": 68.925581,
    "Ge": 73.921178,
    "As": 74.921595,
    "Se": 79.916521,
    "Br": 78.918337,
    "Kr": 83.911507,
    "Rb": 84.911789,
    "Sr": 87.905612,
    "Y": 88.905848,
    "Zr": 89.904704,
    "Nb": 92.906378,
    "Mo": 97.905408,
    "Tc": 97.907212,
    "Ru": 101.904349,
    "Rh": 102.905504,
    "Pd": 105.903486,
    "Ag": 106.905097,
    "Cd": 105.906459,
    "In": 112.904061,
    "Sn": 111.904824,
    "Sb": 120.903815,
    "Te": 127.904461,
    "I": 126.904468,
    "Xe": 128.904779
}

def calculate_mol_weight_custom(mol):
    """
    Calculate molecular weight using custom precise atomic weights.
    If an atom is not in atomic_weights, use RDKit's built-in atomic weight as fallback.
    
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
        if symbol in atomic_weights:
            mol_weight += atomic_weights[symbol]
        else:
            try:
                mol_weight += Descriptors.AtomicWeight(symbol)
            except:
                # Fallback for unknown atoms
                mol_weight += 1.0
    return mol_weight

def calculate_dbe(mol):
    """
    Calculate the DBE (Degree of Unsaturation) of a molecule:
      DBE = (2*C + 2 + N - (H + X)) / 2
    Where C, N, H, X are the counts of carbon, nitrogen, hydrogen, and halogen atoms respectively.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        int: Degree of unsaturation
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
    Calculate molecular DBE, precise molecular weight (log-normalized), and element counts
    for the first five periods using RDKit.
    
    Returns vector dimension: 2 + len(atomic_weights).
    
    Args:
        smiles (str): SMILES string representation
        
    Returns:
        List[float]: Feature vector [dbe, log_mol_weight, atom_counts...]
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None
        
    if mol is None:
        # This case should be filtered during data preprocessing
        return [0.0] * (2 + len(atomic_weights))
        
    dbe = calculate_dbe(mol)
    mol = Chem.AddHs(mol)
    mol_weight = round(calculate_mol_weight_custom(mol), 6)
    mol_weight = np.log1p(mol_weight) if mol_weight > 0 else 0.0
    
    counts = []
    for elem in atomic_weights.keys():
        cnt = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == elem)
        counts.append(cnt)
        
    return [float(dbe), mol_weight] + counts

