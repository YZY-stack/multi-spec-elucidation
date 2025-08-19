import os
import math
import random
import torch
import torch.nn as nn
import re
import numpy as np

# -------------------------------
# 1. 定义辅助函数
# -------------------------------
# ============================
def tokenize_smiles(smiles):
    pattern = r'''
        (\[[CNOF][^\]]*\]) |    # 匹配方括号内的原子，如 [C+], [N-] 等（要求第一个字母为 C、N、O、F）
        (%\d{2})         |      # 匹配两位数的环闭标记，如 %12
        ([CNOF])        |       # 匹配单个原子符号 C, N, O, F
        (\d+)           |       # 匹配环闭数字（一个或多个数字）
        ([=#\-\+\(\)/\\])       # 匹配化学键、括号和斜杠等符号
    '''
    tokens = re.findall(pattern, smiles, re.VERBOSE)
    # 每个匹配返回的是一个元组，取其中非空的部分
    token_list = [next(filter(None, t)).strip() for t in tokens if any(t)]
    return token_list

def smiles_to_indices(smiles, char2idx, max_length):
    """
    将 SMILES 转换为 token 索引序列：
      - 序列头添加 <SOS>，尾添加 <EOS>
      - 超长时截断保证最后一位为 <EOS>
      - 不足时用 <PAD> 填充
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






def mask_smiles_indices(indices, mask_token_idx, char2idx, mask_prob=0.15):
    """
    对 token 序列采用 BERT 式 mask 策略：
      - 30% 替换为 <MASK>
      - 70% 保持原样
    同时返回 mask_positions（bool 类型，True 表示该位置被 mask）。
    首尾以及特殊 token (<SOS>, <EOS>, <PAD>) 均不 mask。
    """
    pad_token = char2idx.get('<PAD>')
    sos_token = char2idx.get('<SOS>')
    eos_token = char2idx.get('<EOS>')
    
    masked_indices = indices.copy()
    mask_positions = [False] * len(indices)
    
    for i in range(1, len(indices)-1):
        # 跳过特殊 token
        if indices[i] in [pad_token, sos_token, eos_token]:
            continue
        if random.random() < mask_prob:
            mask_positions[i] = True
            if random.random() < 0.3:
                masked_indices[i] = mask_token_idx
            # 70% 保持原样
    return masked_indices, mask_positions









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
# 2. Atom-type 特征计算（使用 rdkit）
# ============================
from rdkit import Chem
from rdkit.Chem import Descriptors

# 定义前五周期原子量表
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
    使用自定义精确原子量计算分子量，
    若某原子不在 atomic_weights 中，则使用 RDKit 内置原子量作为后备。
    """
    if mol is None:
        return None
    mol_weight = 0.0
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in atomic_weights:
            mol_weight += atomic_weights[symbol]
        else:
            mol_weight += Descriptors.AtomicWeight(symbol)
    return mol_weight

def calculate_dbe(mol):
    """
    计算分子的 DBE（Degree of Unsaturation）：
      DBE = (2*C + 2 + N - (H + X)) / 2
    其中 C、N、H、X 分别为碳、氮、氢和卤素原子数
    """
    C = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    N = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
    H = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
    halogens = {'F', 'Cl', 'Br', 'I'}
    X = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in halogens)
    dbe = (2 * C + 2 + N - (H + X)) / 2
    return int(dbe)

def compute_atom_types(smiles):
    """
    利用 rdkit 计算分子的 DBE、精确分子量（对数归一化）以及前五周期各元素的计数，
    返回向量维度为：2 + len(atomic_weights)。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # 理论上此情况在数据预处理阶段已被过滤
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

