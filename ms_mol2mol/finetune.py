"""
Molecular Fine-tuning Module

This module implements fine-tuning capabilities for the transformer-based molecular 
representation model. It includes distributed training support, dynamic gradient clipping,
and advanced learning rate scheduling for optimal performance.

Author: ms_mol2mol Team
"""

import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
import re
import math
import random
import lmdb
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection for debugging
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler, IterableDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from rdkit.Chem import AllChem, DataStructs

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from rdkit import Chem


MASK_TOKEN = "<MASK>"  # Ensure this token is added to vocabulary


# ============================
# SMILES Tokenizer and Data Preprocessing (Enhanced for C, N, O, F plus additional elements)
# ============================
# def tokenize_smiles(smiles):
#     pattern = r'''
#         (\[[CNOF][^\]]*\]) |    # Match atoms within brackets, such as [C+], [N-] (requiring first letter to be C, N, O, F)
#         (%\d{2})         |      # Match two-digit ring closure markers, such as %12
#         ([CNOF])        |       # Match single atom symbols C, N, O, F
#         (\d+)           |       # Match ring closure numbers (one or more digits)
#         ([=#\-\+\(\)/\\])       # Match chemical bonds, parentheses and slashes
#     '''
#     tokens = re.findall(pattern, smiles, re.VERBOSE)
#     # Each match returns a tuple, take the non-empty part
#     token_list = [next(filter(None, t)).strip() for t in tokens if any(t)]
#     return token_list

def tokenize_smiles(smiles: str) -> List[str]:
    """
    Enhanced SMILES tokenizer supporting expanded element set.
    
    Matches multi-character elements (Br/Cl) first, then single characters.
    Supports bracketed atoms, ring closures, and chemical bonds.
    
    Args:
        smiles (str): SMILES string representation
        
    Returns:
        List[str]: List of chemical tokens
    """
    # Enhanced pattern: matches special tokens, bracketed atoms, multi-char elements first
    pattern = r'''(<[A-Z]+>)|(\[[^\]]+])|(%\d{2})|(Br|Cl)|([BCNOFHPSI])|(\d+)|([=#\-\+\(\)/\\])'''
    tokens = re.findall(pattern, smiles, re.VERBOSE)
    return [next(filter(None, t)).strip() for t in tokens if any(t)]

def smiles_to_indices(smiles: str, char2idx: Dict[str, int], max_length: int) -> Optional[List[int]]:
    """
    Convert SMILES string to token indices with robust error handling.
    
    Returns None for empty strings or empty token lists to trigger dataset resampling.
    Unknown tokens are mapped to <UNK> instead of causing failures.
    
    Args:
        smiles (str): SMILES string
        char2idx (Dict[str, int]): Character to index mapping
        max_length (int): Maximum sequence length
        
    Returns:
        Optional[List[int]]: Token indices or None if invalid
    """
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        return None

    tokens = tokenize_smiles(smiles)
    if len(tokens) == 0:
        return None  # Only empty tokens count as failure

    indices = [char2idx['<SOS>']]
    for t in tokens:
        indices.append(char2idx.get(t, char2idx['<UNK>']))  # Map unknown tokens
    indices.append(char2idx['<EOS>'])

    if len(indices) < max_length:
        indices += [char2idx['<PAD>']] * (max_length - len(indices))
    else:
        indices = indices[:max_length-1] + [char2idx['<EOS>']]
    return indices





# ============================
# Atom-type Feature Calculation (Using RDKit)
# ============================
from rdkit import Chem
from rdkit.Chem import Descriptors


# Atomic weights for first five periods
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

# Common elements subset for focused analysis
atomic_weights = {
    "H": 1.007825,
    "B": 11.009305,
    "C": 12.000000,
    "N": 14.003074,
    "O": 15.994915,
    "P": 30.973762,
    "S": 31.972071,
    "F": 18.998403,
    "Cl": 34.968853,
    "Br": 78.918337,
    "I": 126.904468,
}


# def calculate_mol_weight_custom(mol):
#     """
#     使用自定义精确原子量计算分子量，
#     若某原子不在 atomic_weights 中，则使用 RDKit 内置原子量作为后备。
#     """
#     if mol is None:
#         return None
#     mol_weight = 0.0
#     for atom in mol.GetAtoms():
#         symbol = atom.GetSymbol()
#         if symbol in atomic_weights:
#             mol_weight += atomic_weights[symbol]
#         else:
#             mol_weight += Descriptors.AtomicWeight(symbol)
#     return mol_weight


ptable = Chem.GetPeriodicTable()  # RDKit internal periodic table

def calculate_mol_weight_custom(mol: Chem.Mol) -> float:
    """
    Calculate precise molecular weight using RDKit periodic table.
    Returns 0.0 for any exception to avoid NaN values.
    
    Args:
        mol (Chem.Mol): RDKit molecule object
        
    Returns:
        float: Molecular weight or 0.0 if calculation fails
    """
    if mol is None:
        return 0.0

    weight = 0.0
    for atom in mol.GetAtoms():
        try:
            weight += ptable.GetAtomicWeight(atom.GetAtomicNum())
        except Exception:  # Handle unusual elements
            weight += 0.0
    return weight


def calculate_dbe(mol: Chem.Mol) -> int:
    """
    Calculate Degree of Unsaturation (DBE) for a molecule.
    DBE = (2*C + 2 + N - (H + X)) / 2
    where C, N, H, X are counts of carbon, nitrogen, hydrogen, and halogen atoms.
    
    Args:
        mol (Chem.Mol): RDKit molecule object
        
    Returns:
        int: Degree of unsaturation
    """
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    nitrogen_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
    hydrogen_count = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
    halogens = {'F', 'Cl', 'Br', 'I'}
    halogen_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in halogens)
    dbe = (2 * carbon_count + 2 + nitrogen_count - (hydrogen_count + halogen_count)) / 2
    return int(dbe)

# def compute_atom_types(smiles):
#     """
#     利用 rdkit 计算分子的 DBE、精确分子量（对数归一化）以及前五周期各元素的计数，
#     返回向量维度为：2 + len(atomic_weights)。
#     """
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         # 理论上此情况在数据预处理阶段已被过滤
#         return [0.0] * (2 + len(atomic_weights))
#     dbe = calculate_dbe(mol)
#     mol = Chem.AddHs(mol)
#     mol_weight = round(calculate_mol_weight_custom(mol), 6)
#     mol_weight = np.log1p(mol_weight) if mol_weight > 0 else 0.0
#     counts = []
#     for elem in atomic_weights.keys():
#         cnt = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == elem)
#         counts.append(cnt)
#     return [float(dbe), mol_weight] + counts


def compute_atom_types(smiles: str) -> List[float]:
    """
    Compute atomic features for a molecule including DBE, molecular weight, and element counts.
    
    Returns a feature vector of length = 2 + len(atomic_weights):
    [DBE, log(mol_weight + 1)] + element counts (log-normalized)
    All components are guaranteed to be finite values.
    
    Args:
        smiles (str): SMILES string representation
        
    Returns:
        List[float]: Feature vector with DBE, log molecular weight, and element counts
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0.0] * (2 + len(atomic_weights))

    # Calculate DBE (Degree of Unsaturation)
    dbe = calculate_dbe(mol)

    # Calculate precise molecular weight (log-normalized)
    mol_weight = calculate_mol_weight_custom(Chem.AddHs(mol))
    log_mol_weight = math.log1p(mol_weight) if mol_weight > 0 else 0.0  # Avoid log(0)

    # Calculate element counts (log-normalized for better distribution)
    element_counts = []
    for element in atomic_weights.keys():
        count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == element)
        element_counts.append(math.log1p(count))  # Log normalization
        
    return [float(dbe), log_mol_weight] + element_counts


def _has_bad_values(tensor: torch.Tensor) -> bool:
    """Check if tensor contains NaN or Inf values."""
    return tensor is not None and (torch.isnan(tensor).any() or torch.isinf(tensor).any())


def catch_nan(model: torch.nn.Module) -> bool:
    """
    Check model parameters and gradients for NaN/Inf values.
    
    Args:
        model (torch.nn.Module): Model to check
        
    Returns:
        bool: True if NaN/Inf values found, False otherwise
    """
    for name, param in model.named_parameters():
        if _has_bad_values(param):
            print(f'★ Parameter NaN/Inf detected in: {name}')
            return True
        if _has_bad_values(param.grad):
            print(f'★ Gradient NaN/Inf detected in: {name}')
            return True
    return False


# ============================
# Data Augmentation Functions
# ============================
import random
import functools
import glob
import re
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem import AllChem, BRICS


def sanitize_or_fallback(mol: Chem.Mol, fallback: str) -> str:
    """
    Attempt to generate Kekule SMILES; fallback to original string on failure.
    
    Args:
        mol (Chem.Mol): RDKit molecule object
        fallback (str): Fallback SMILES string
        
    Returns:
        str: Kekule SMILES or fallback string
    """
    try:
        Chem.SanitizeMol(mol)
        result = Chem.MolToSmiles(mol, canonical=False, kekuleSmiles=True)
        return result if result else fallback
    except Exception:
        return fallback


# ============================
# Low-level Perturbation Functions
# ============================

def atom_miscount(smiles: str) -> str:
    """
    Atom miscount perturbation with two strategies:
    - 'del': Directly remove a non-carbon atom
    - 'add': Atom swapping (consistent with original implementation)
    
    Args:
        smiles (str): Input SMILES string
        
    Returns:
        str: Perturbed SMILES string
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles

    choice = random.choice(["del", "add"])
    if choice == "del":
        del_candidates = [a for a in mol.GetAtoms()
                          if a.GetSymbol() in ["O", "F", "Cl", "Br", "N", "S", "P", "I"]]
        if not del_candidates:
            return smiles
        tgt = random.choice(del_candidates)
        em = Chem.EditableMol(mol)
        em.RemoveAtom(tgt.GetIdx())          # ✅ 真·删除
        mol = em.GetMol()
    else:  # add / swap
        atom = random.choice(mol.GetAtoms())
        mapping = {
            "C": "N",  "N": "C",
            "O": "S",  "S": "O",
            "F": "Cl", "Cl": "F",
            "Br": "I", "I": "Br",
        }
        atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(
            mapping.get(atom.GetSymbol(), "C")
        ))

    return sanitize_or_fallback(mol, smiles)

# --- ② Ring 相关 ---
def ring_shift(sm: str) -> str:
    def repl(m):
        tok = m.group(0)
        if tok.startswith("%"):
            return f"%{(int(tok[1:])+1)%90+10:02d}"
        return str((int(tok) + 1) % 10)
    return re.sub(r"%\d{2}|\d", repl, sm, count=1)

def ring_break(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return smiles
    ring_bonds = [b.GetIdx() for b in mol.GetBonds() if b.IsInRing()]
    if not ring_bonds: return smiles
    b = mol.GetBondWithIdx(random.choice(ring_bonds))
    em = Chem.EditableMol(mol)
    em.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    return sanitize_or_fallback(em.GetMol(), smiles)

# --- ③ FG drift ---
def fg_drift(smiles: str) -> str:
    patt = Chem.MolFromSmarts("[CX3]=O")
    mol  = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.HasSubstructMatch(patt):
        return smiles
    idx = mol.GetSubstructMatch(patt)[0]        # 羰碳
    orig_nbr = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()][0]
    cand_atoms = [a.GetIdx() for a in mol.GetAtoms()
                  if a.GetIdx() not in (idx, orig_nbr) and
                     not mol.GetBondBetweenAtoms(idx, a.GetIdx())]
    if not cand_atoms: return smiles
    new_nbr = random.choice(cand_atoms)
    em = Chem.EditableMol(mol)
    em.RemoveBond(idx, orig_nbr)
    em.AddBond(idx, new_nbr, Chem.rdchem.BondType.DOUBLE)
    return sanitize_or_fallback(em.GetMol(), smiles)

# --- ④ Bond swap ---
def bond_swap(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return smiles
    b = random.choice(mol.GetBonds())
    em = Chem.EditableMol(mol)
    mapping = {
        Chem.rdchem.BondType.SINGLE:  Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.DOUBLE:  Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.TRIPLE:  Chem.rdchem.BondType.DOUBLE,
    }
    new_type = mapping.get(b.GetBondType(), b.GetBondType())
    em.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    em.AddBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), new_type)
    return sanitize_or_fallback(em.GetMol(), smiles)

# --- ⑤ Fragment permute ---
def frag_permute(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return smiles
    frags = list(BRICS.BRICSDecompose(mol, returnMols=True))
    if len(frags) < 2: return smiles
    random.shuffle(frags)
    new = Chem.CombineMols(frags[0], frags[1])
    return sanitize_or_fallback(new, smiles)

# --- ⑥ FG mutate ---
FG_SMARTS = {
    "carbonyl"      : "[CX3]=[OX1]",
    "aldehyde"      : "[CX3H1](=O)[#6]",
    "ketone"        : "[#6][CX3](=O)[#6]",
    "ester"         : "[CX3](=O)[OX2][#6]",
    "amide"         : "[NX3][CX3](=O)[#6]",
    "carboxylic"    : "C(=O)[OX2H1]",
    "alcohol"       : "[OX2H][CX4]",
    "amine_primary" : "[NX3;H2][CX4]",
    "amine_sec"     : "[NX3;H1]([#6])[#6]",
    "amine_tert"    : "[NX3]([#6])([#6])[#6]",
    "nitro"         : "[NX3](=O)(=O)[O-]?",
}
FG_REPLACE_SMILES = {
    "carbonyl": "C=O",        "aldehyde": "CO",
    "ketone": "COC",          "ester": "CC",
    "amide": "CNC",           "carboxylic": "CO",
    "alcohol": "F",           "amine_primary": "O",
    "amine_sec": "OC",        "amine_tert": "C",
    "nitro": "CN",
}

def fg_mutate(smiles: str) -> str:
    """
    Functional group mutation with deletion or replacement strategies.
    
    Args:
        smiles (str): Input SMILES string
        
    Returns:
        str: SMILES with mutated functional groups
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return smiles
        
    matches = [(name, Chem.MolFromSmarts(pattern)) 
               for name, pattern in FG_SMARTS.items()
               if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))]
    if not matches: 
        return smiles
        
    try:
        name, pattern = random.choice(matches)
        if random.random() < 0.75:  # Delete functional group
            mol2 = Chem.DeleteSubstructs(mol, pattern)
        else:  # Replace functional group
            replacement = Chem.MolFromSmiles(FG_REPLACE_SMILES[name])
            mol2 = AllChem.ReplaceSubstructs(mol, pattern, replacement, replaceAll=False)[0]
        return sanitize_or_fallback(mol2, smiles)
    except Exception:
        return smiles


def substruct_shuffle(smiles: str) -> str:
    """
    Shuffle molecular substructures using BRICS decomposition.
    
    Args:
        smiles (str): Input SMILES string
        
    Returns:
        str: SMILES with shuffled substructures
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return smiles
        
    fragments = list(BRICS.BRICSDecompose(mol, returnMols=False))
    if len(fragments) < 2: 
        return smiles
        
    selected = random.sample(fragments, k=min(3, len(fragments)))
    random.shuffle(selected)
    return ".".join(selected)


# ============================
# Comprehensive SMILES Corruptor
# ============================

class RobustSmilesCorruptor:
    """
    Unified SMILES corruption interface with safe operations.
    Removes BRICS-related operations to avoid [*] placeholders.
    Applies 1-2 perturbations sequentially for stability.
    """
    def __init__(self):
        self.token_config = dict(mask_p=0.15, delete_p=0.15, replace_p=0.10)
        self.operation_counter = {}

        # Safe operation list (excluding BRICS operations)
        self._operations = [
            ("token_level", self._token_level_corruption),
            ("fg_drift", fg_drift),
            ("atom_miscount", atom_miscount),
            ("ring_shift", ring_shift),
            ("bond_swap", bond_swap),
        ]

    def _token_level_corruption(self, smiles: str) -> str:
        """
        Apply token-level corruptions: masking, deletion, and replacement.
        
        Args:
            smiles (str): Input SMILES string
            
        Returns:
            str: Corrupted SMILES string
        """
        output_tokens = []
        for token in tokenize_smiles(smiles):
            rand_value = random.random()
            if rand_value < self.token_config["mask_p"]:
                output_tokens.append(MASK_TOKEN)
            elif rand_value < self.token_config["mask_p"] + self.token_config["delete_p"]:
                continue  # Skip token (deletion)
            elif rand_value < sum(self.token_config.values()):
                # Random replacement with supported atoms
                output_tokens.append(random.choice(
                    ['B', 'C', 'N', 'O', 'F', 'H', 'P', 'S', 'I', 'Br', 'Cl']
                ))
            else:
                output_tokens.append(token)
        return "".join(output_tokens)

    def __call__(self, smiles: str) -> str:
        """
        Apply corruption operations to SMILES string.
        
        Args:
            smiles (str): Input SMILES string
            
        Returns:
            str: Corrupted SMILES string
        """
        num_operations = random.choices([1, 2], weights=[0.7, 0.3])[0]  # Max 2 operations
        chosen_operations = random.sample(self._operations, k=num_operations)

        result = smiles
        for operation_name, operation_func in chosen_operations:
            try:
                result = operation_func(result)
                self.operation_counter[operation_name] = self.operation_counter.get(operation_name, 0) + 1
            except Exception:
                self.operation_counter["fallback"] = self.operation_counter.get("fallback", 0) + 1
                return smiles
        return result


# ============================
# In-memory Dataset Implementation
# ============================
import glob
import random

class SMILESPretokenDataset(Dataset):
    """
    High-performance in-memory SMILES dataset with corruption and atomic features.
    Supports both single files and directory patterns for flexible data loading.
    """
    def __init__(self, csv_path: str, char2idx: Dict[str, int], max_seq_length: int, 
                 compute_atom_types_fn, corruption_level: float = 0.2):
                  如果是文件夹，则加载该文件夹下所有 CSV 文件（每个 CSV 文件第一行为标题，列名为 "SMILES"）；
                  如果是 CSV 文件，则只加载该文件的数据。
        char2idx: SMILES 到索引的映射字典
        max_seq_length: 最大序列长度（含 <SOS>, <EOS>, padding）
        compute_atom_types_fn: 根据 SMILES 返回 atom-type 特征向量的函数
        mask_prob: mask 的概率
        """

        self.smiles_list = []
        # # 如果传入的是文件夹，则加载所有 csv 文件；如果是单个 csv 文件，则只加载该文件
        # if os.path.isdir(csv_path):
        #     csv_files = sorted(glob.glob(os.path.join(csv_path, "*.csv")))
        # else:
        #     csv_files = [csv_path]
        # print(f"Found {len(csv_files)} CSV file(s). Loading...")
        # for csv_file in tqdm(csv_files):
        df = pd.read_csv(csv_path)
        smiles = df["SMILES"].tolist()
        self.smiles_list.extend(smiles)
        print(f"Total molecules loaded: {len(self.smiles_list)}")
        self.vocab_self_check()

        # self.mask_prob = mask_prob
        self.char2idx = char2idx
        self.max_seq_length = max_seq_length
        self.compute_atom_types = compute_atom_types_fn
        self.mask_token_idx = char2idx.get('<MASK>')
        
        # 假设所有 SMILES 均有效
        self.valid_indices = list(range(len(self.smiles_list)))

        self.corruptor = RobustSmilesCorruptor()

    def vocab_self_check(self):
        from collections import Counter
        bad = Counter()
        for s in tqdm(self.smiles_list):
            for t in tokenize_smiles(s):
                if t not in char2idx:
                    bad[t] += 1
        print(bad.most_common(50))

    # ---------- __len__ --------------------------------------
    def __len__(self):
        return len(self.valid_indices)

    # ---------- __getitem__ ----------------------------------
    def __getitem__(self, idx):
        trial = 0
        while True:
            smiles = self.smiles_list[self.valid_indices[idx]]
            corrupted = self.corruptor(smiles)

            src_indices = smiles_to_indices(corrupted, self.char2idx, self.max_seq_length)
            tgt_indices = smiles_to_indices(smiles,    self.char2idx, self.max_seq_length)

            if src_indices is not None and tgt_indices is not None:
                break  # 👍 取得合法样本

            trial += 1
            if trial >= 10:
                raise RuntimeError(f"Too many invalid SMILES around idx={idx}")
            # 随机换一个索引再试
            idx = random.randint(0, len(self.valid_indices) - 1)


        if random.random() < 0.001:
            print(f"ori: {smiles} | corr: {corrupted}")


        src_tensor = torch.tensor(src_indices, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long)

        # mask_positions 这里全设 False
        mask_positions = torch.zeros_like(src_tensor, dtype=torch.bool)  # 占位


        # 原子类型特征不变
        # atom_types = torch.tensor(self.compute_atom_types(smiles), dtype=torch.float32)
        atom_types = torch.tensor(
            self.compute_atom_types(smiles), dtype=torch.float32
        )
        # ★ 把任何 nan/inf 均替换为有限值
        atom_types = torch.nan_to_num(atom_types, nan=0.0, posinf=1e4, neginf=-1e4)

        return src_tensor, tgt_tensor, atom_types, mask_positions

    



# ============================
#  Dataset for SMILES Refinement
# ============================
class SMILESPretokenDataset(Dataset):
    """
    用于“预测 SMILES → 修正为 ground-truth SMILES”的场景。

    CSV 格式（无表头或忽略表头）:
        gt_smiles , pred_smiles [, 其它列...]

    - src  : pred_smiles  （模型要修正的输入）
    - tgt  : gt_smiles    （监督目标）
    - 可选: 设置 augment_src=True 时，再对 pred_smiles 做一次轻量扰动以增强鲁棒性
    """
    def __init__(self,
                 csv_path: str,
                 char2idx: dict,
                 max_seq_length: int,
                 compute_atom_types_fn,
                 augment_src: bool = True):
        super().__init__()

        # -------- 1. 读 CSV --------
        df = pd.read_csv(csv_path, header=None)        # 不信任表头，直接按列位
        if df.shape[1] < 2:
            raise ValueError("CSV 至少需要两列：gt_smiles,pred_smiles")
        self.tgt_list  = df.iloc[:, 0].astype(str).tolist()   # ground-truth
        self.src_list  = df.iloc[:, 1].astype(str).tolist()   # prediction

        assert len(self.tgt_list) == len(self.src_list)
        print(f"Loaded {len(self.src_list):,} (src,tgt) pairs from {csv_path}")

        # -------- 2. 基本属性 --------
        self.char2idx        = char2idx
        self.max_seq_length  = max_seq_length
        self.compute_atom_types = compute_atom_types_fn
        self.augment_src     = augment_src
        if augment_src:
            self.corruptor = RobustSmilesCorruptor()   # 轻量增广

        # -------- 3. 词表自检（可注释掉以加速）--------
        self._vocab_self_check()

    # ---------- 词表检查 ----------
    def _vocab_self_check(self, topk: int = 20):
        from collections import Counter
        bad = Counter()
        for sm in tqdm(self.tgt_list + self.src_list,
                       desc="vocab-self-check", leave=False):
            for tok in tokenize_smiles(sm):
                if tok not in self.char2idx:
                    bad[tok] += 1
        if bad:
            print(f"[Vocab] unknown tokens (top {topk}): {bad.most_common(topk)}")
        else:
            print("[Vocab] ✓ no unknown tokens")

    # ---------- __len__ ----------
    def __len__(self):
        return len(self.src_list)

    # ---------- __getitem__ ----------
    def __getitem__(self, idx):
        """若 src/tgt 任一 tokenization 失败，随机换一个 idx"""
        for trial in range(10):
            src_smiles = self.src_list[idx]
            tgt_smiles = self.tgt_list[idx]

            if self.augment_src:
                src_smiles = self.corruptor(src_smiles)

            src_idx = smiles_to_indices(src_smiles, self.char2idx, self.max_seq_length)
            tgt_idx = smiles_to_indices(tgt_smiles, self.char2idx, self.max_seq_length)

            if src_idx is not None and tgt_idx is not None:
                break    # 👍 合法
            idx = random.randint(0, len(self.src_list) - 1)
        else:
            # 连续 10 次失败才抛错，避免死循环
            raise RuntimeError(f"Too many invalid SMILES (last idx={idx})")

        # --- 后续保持不变 ---
        src_tensor = torch.tensor(src_idx, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_idx, dtype=torch.long)

        # ---- 3. Atom-type 特征：用 ground-truth 计算 ----
        atom_types = torch.tensor(
            self.compute_atom_types(tgt_smiles), dtype=torch.float32
        )
        atom_types = torch.nan_to_num(atom_types, nan=0.0, posinf=1e4, neginf=-1e4)

        # ---- 4. 占位 mask_positions（暂时全 False）----
        mask_positions = torch.zeros_like(src_tensor, dtype=torch.bool)

        # ---- 5. Debug 随机打印 ----
        if random.random() < 0.001:
            print(f"SRC {src_smiles}  |  TGT {tgt_smiles}")

        return src_tensor, tgt_tensor, atom_types, mask_positions





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




# def gen_sub_mask(sz, device):
#     return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

def gen_sub_mask(sz, device):
    mask = torch.triu(torch.full((sz, sz), float('-inf'), device=device), 1)
    mask.masked_fill_(mask == float('-inf'), -1e4)   # ✅ Patch-2：不给全 -inf
    return mask

class MoleculePretrainingModel(nn.Module):
    def __init__(self, vocab_size, atom_type_dim,
                 d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.Sequential(EncoderWithConv(d_model),
                                     nn.TransformerEncoder(enc, num_layers=num_encoder_layers))

        dec = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = nn.TransformerDecoder(dec, num_layers=num_decoder_layers)

        self.atom_proj   = nn.Linear(atom_type_dim, d_model)
        self.atom_norm   = nn.LayerNorm(atom_type_dim)      ### NEW ###
        self.init_proj   = nn.Linear(d_model, d_model)
        self.out_linear  = nn.Linear(d_model, vocab_size)

    def forward(self, src_seq, tgt_seq, atom_types, memory_key_padding_mask=None):
        # ---------- Encoder ----------
        src = self.pos_encoder(self.embedding(src_seq) * math.sqrt(self.d_model))   # [B,L,E]
        memory = self.encoder(src.transpose(0,1))                                   # [L,B,E]

        # ---------- Decoder ----------
        tgt = self.pos_encoder(self.embedding(tgt_seq) * math.sqrt(self.d_model))

        atom_feats = self.atom_norm(atom_types)                 ### NEW ###
        atom_emb   = self.init_proj(self.atom_proj(atom_feats)).unsqueeze(1)  ### MOD ###
        # atom_emb = self.init_proj(self.atom_proj(atom_types)).unsqueeze(1)          # [B,1,E]
        tgt = torch.cat([atom_emb, tgt], dim=1)                                     # prepend

        tgt_mask = gen_sub_mask(tgt.size(1), tgt.device)     # ★FIX④

        dec_out = self.decoder(tgt.transpose(0,1), memory,
                               tgt_mask=tgt_mask,
                               memory_key_padding_mask=memory_key_padding_mask)     # ★FIX⑤
        logits = self.out_linear(dec_out.transpose(0,1)[:,1:])  # 去掉 prepend
        return logits





# ============================
# 3. SMILES 词表与映射（仅包含 C、N、O、F 及相关符号）
# ============================
SMILES_VOCAB = [
    # 特殊 Token
    '<PAD>', '<SOS>', '<EOS>', '<UNK>', '<MASK>',
    
    # 原子符号（基础）
    # 'C', 'N', 'O', 'F',
    'B', 'C', 'N', 'O', 'F', 'H', 'P', 'S', 'I',
    'Br', 'Cl',
    
    # 带电原子形式
    '[C]', '[CH]', '[CH2]', '[CH3]', 
    '[N+]', '[N-]', '[NH+]', '[NH2+]', '[NH3+]',
    '[O-]', '[OH+]',
    # 新的带电原子
    '[Si]', '[S-]', '[S+]', '[O]', '[P+]', '[B-]', '[PH]', '[O+]', '[C-]',
    
    # 化学符号
    '(', ')', '[', ']', '=', '#', '-', '+', '/', '\\',
    
    # 环闭标记（两位数）
    *[f'%{i}' for i in range(10, 100)],
    
    # 数字（0-9 和 10-99）
    *[str(i) for i in range(100)],
    
    # 补充常见同位素标记
    '[13C]', '[14C]', '[15N]'
]


vocab_size = len(SMILES_VOCAB)
char2idx = {token: idx for idx, token in enumerate(SMILES_VOCAB)}
idx2char = {idx: token for idx, token in enumerate(SMILES_VOCAB)}

assert '<UNK>' in char2idx, "词表里缺少 <UNK>！"

def decode_indices(indices, idx2char):
    """
    根据索引序列解码为 SMILES 字符串，遇到 <EOS> 停止。
    跳过 <SOS> 与 <PAD>。
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
# 8. 训练函数（DDP + tqdm + 学习率 Warmup + 动态梯度裁剪 + 有效性检查）
# ============================
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import logging


############################################
# 设置日志记录器
############################################
def setup_logger(log_file):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
    return logger



############################################
def train(rank, world_size, csv_folder, num_epochs=10, batch_size=128, max_seq_length=300):
    # 设置多卡训练的环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10412'
    
    # 初始化分布式进程组
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.manual_seed(42)
    device = torch.device('cuda', rank)
    
    # 仅在 rank==0 初始化 TensorBoard writer 和日志记录器
    if rank == 0:
        writer = SummaryWriter(log_dir="runs/exp")
        logger = setup_logger("training.log")
        logger.info("Training started.")

    global_step = 0
    #total_steps = 1000000  # 一百万
    # total_steps = 4000
    # total_steps = 300000000



    # # 获取 CSV 文件列表（假设 csv_folder 下正好有 10 个 CSV 文件）
    # csv_files = sorted(glob.glob(os.path.join(csv_folder, "*.csv")))
    # assert len(csv_files) == num_epochs, f"Expected {num_epochs} CSV files, but found {len(csv_files)}"

    # 预先计算所有 epoch 的总步数（仅供 lr scheduler 使用）
    # total_steps = 30000000
    total_steps = 2000000
    # for csv_file in csv_files:
    #     df = pd.read_csv(csv_file)
    #     num_samples = len(df)
    #     num_batches = math.ceil(num_samples / batch_size)
    #     total_steps += num_batches


    # 初始化模型、损失、优化器、调度器（模型保持不变，所有 epoch 共用）
    # atom_type_dim = len(compute_atom_types("C"))
    atom_type_dim = 13
    # print(atom_type_dim)
    model = MoleculePretrainingModel(vocab_size, atom_type_dim, d_model=512, nhead=8,
                                     num_encoder_layers=6, num_decoder_layers=6,
                                     dim_feedforward=2048, dropout=0.1).to(device)
    # checkpoint_path = "init_pretrain_molecule_pretraining_model_epoch.pth"
    # checkpoint_path = "zinc20_pretrain.pth"
    checkpoint_path = "zinc20_s2m.pth"
    if os.path.exists(checkpoint_path):
        # state_dict = torch.load(checkpoint_path, map_location=device)
        # new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # model.load_state_dict(new_state_dict, strict=False)

        raw_sd = torch.load(checkpoint_path, map_location=device)

        # 先统一去掉 "module." 前缀
        raw_sd = {k.replace("module.", ""): v for k, v in raw_sd.items()}

        # 当前模型的参数形状
        model_sd = model.state_dict()

        # 只保留 “名称存在且形状一致” 的权重
        filtered_sd, skipped = {}, []
        for k, v in raw_sd.items():
            if k in model_sd and v.shape == model_sd[k].shape:
                filtered_sd[k] = v
            else:
                skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape) if k in model_sd else None))

        print(f"Loaded {len(filtered_sd)} tensors; skipped {len(skipped)}:")
        for name, old_shape, new_shape in skipped:
            print(f"  - {name}: ckpt {old_shape} -> model {new_shape}")

        missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
        print(f"State-dict ok.  missing={missing},  unexpected={unexpected}")


        print(f"Rank {rank} loaded pretrained weights.")
    else:
        print(f"Rank {rank} checkpoint not found.")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    pad_idx = char2idx['<PAD>']
    unk_idx = char2idx['<UNK>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    def lr_lambda(step):
        warmup_steps = 10000
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, (total_steps - warmup_steps))))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    for epoch in range(num_epochs):
        # ===== dataset / sampler / dataloader =====
        dataset  = SMILESPretokenDataset(
            csv_folder, char2idx, max_seq_length, compute_atom_types)
        sampler  = DistributedSampler(dataset, world_size, rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                sampler=sampler, num_workers=4,
                                pin_memory=True)
        sampler.set_epoch(epoch)

        #save_interval = max(1, len(dataloader) // 10)  # ★ 每 epoch 10 份
        save_interval = max(1, len(dataloader) // 2)
        if rank == 0:
            pbar = tqdm(dataloader,
                        desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)
        else:
            pbar = dataloader

        total_loss = 0.0
        for i, (src_seq, tgt_seq, atom_types, mask_pos) in enumerate(pbar):
            src_seq, tgt_seq = src_seq.to(device), tgt_seq.to(device)
            atom_types = atom_types.to(device)

            tgt_in, tgt_out = tgt_seq[:, :-1], tgt_seq[:, 1:]
            src_pad = (src_seq == char2idx["<PAD>"])

            optimizer.zero_grad()
            logits = model(src_seq, tgt_in, atom_types,
                           memory_key_padding_mask=src_pad)
            
            if _has_bad(logits):
                print('★ forward-pass NaN/Inf in logits')
                break

            valid = (tgt_out != pad_idx) & (tgt_out != unk_idx)   # 同时过滤 PAD & UNK
            if valid.sum() == 0:          # ✅ Patch-1
                continue                  # 跳过该 mini-batch
            loss  = criterion(logits[valid], tgt_out[valid]).mean()

            if _has_bad(loss):
                print('★ loss NaN/Inf BEFORE backward')
                break



            loss.backward()
            if catch_nan(model):                    # <-- 一旦触发立刻保存后跳出
                torch.save(model.state_dict(), 'nan_dump.pth')
                break


            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            global_step += 1

            # ----- 进度条 / TB -----
            if rank == 0:
                writer.add_scalar("Loss/iter", loss.item(), global_step)
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=optimizer.param_groups[0]["lr"])

                # ★★★ 按步保存 ★★★
                if ((i + 1) % save_interval == 0) or (
                        i + 1 == len(dataloader)):
                    part = (i + 1) // save_interval + (
                        0 if (i + 1) % save_interval else 0)
                    ckpt_name = f"semantic_ckpt_e{epoch+1:03d}_p{part:02d}.pth"
                    torch.save(model.state_dict(), ckpt_name)
                    logger.info(f"Saved checkpoint -> {ckpt_name}")

        # ===== epoch 统计 =====
        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{num_epochs} | "
                        f"AvgLoss {avg_loss:.4f}")
            writer.add_scalar("Loss/epoch", avg_loss, epoch)
            writer.flush()

    if rank == 0:
        writer.close()
    dist.destroy_process_group()


# ============================
# 9. DDP 多卡训练入口
# ============================
def main():
    csv_folder = "./gp/csv/corr_draw/top10_preds_long_2.csv"  # CSV 文件所在目录（10个 CSV 文件）
    world_size = 8  # 使用 4 张 GPU
    mp.spawn(train, args=(world_size, csv_folder, 10, 128, 400), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()


