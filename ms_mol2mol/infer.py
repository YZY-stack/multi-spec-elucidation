"""
Molecular Inference Module

This module provides inference capabilities for the trained molecular transformer model.
It includes SMILES generation, beam search, and molecular property prediction functionality.

Author: ms_mol2mol Team
"""

import os
import math
import random
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger, DataStructs
RDLogger.DisableLog('rdApp.*')

from model import *
from utils import *

# ============================
# Model and Vocabulary Definitions
# ============================

# Define SMILES vocabulary (consistent with training)
SMILES_VOCAB = [
    # Special tokens
    '<PAD>', '<SOS>', '<EOS>', '<UNK>', '<MASK>',
    
    # Basic atomic symbols
    'C', 'N', 'O', 'F',
    
    # Charged atomic forms (example extensions)
    '[C]', '[CH]', '[CH2]', '[CH3]', 
    '[N+]', '[N-]', '[NH+]', '[NH2+]', '[NH3+]',
    '[O-]', '[OH+]',
    
    # Chemical symbols
    '(', ')', '[', ']', '=', '#', '-', '+', '/', '\\',
    
    # Ring closure markers (two digits)
    *[f'%{i}' for i in range(10, 100)],
    
    # Numbers (0-9 and 10-99)
    *[str(i) for i in range(100)],
    
    # Common isotope markers
    '[13C]', '[14C]', '[15N]'
]

vocab_size = len(SMILES_VOCAB)
char2idx = {token: idx for idx, token in enumerate(SMILES_VOCAB)}
idx2char = {idx: token for idx, token in enumerate(SMILES_VOCAB)}

# Atom type dimension based on compute_atom_types("C") length
atom_type_dim = len(compute_atom_types("C"))




# ============================
# Autoregressive Generation Helper Functions
# ============================

def generate_smiles(model: nn.Module, src_seq: torch.Tensor, atom_types: torch.Tensor, 
                   max_length: int, char2idx: Dict[str, int], additional_info: List[int], 
                   device: torch.device) -> List[int]:
    """
    Generate complete SMILES sequence using greedy decoding strategy.
    
    Args:
        model (nn.Module): Trained transformer model
        src_seq (torch.Tensor): [1, seq_len] masked SMILES index sequence
        atom_types (torch.Tensor): [1, atom_type_dim] atomic features for original SMILES
        max_length (int): Maximum generation length (excluding <SOS>)
        char2idx (Dict[str, int]): Character to index mapping
        additional_info (List[int]): [max_C, max_N, max_O, max_F] maximum atom counts
        device (torch.device): Computation device
        
    Returns:
        List[int]: Generated token indices (excluding initial <SOS>)
    """
    model.eval()
    with torch.no_grad():
        # 1. Encoder processing
        src_emb = model.embedding(src_seq) * math.sqrt(model.d_model)
        src_emb = model.pos_encoder(src_emb)
        # Transformer Encoder requires input shape [seq_len, batch, d_model]
        src_emb = src_emb.transpose(0, 1)  # [seq_len, 1, d_model]
        memory = model.encoder(src_emb)      # [seq_len, 1, d_model]

        # 2. Decoder initialization: start with <SOS>
        input_tgt = torch.tensor([[char2idx['<SOS>']]], device=device)  # [1, 1]

        for _ in range(max_length):
            # Current decoder input embedding with positional encoding
            tgt_emb = model.embedding(input_tgt) * math.sqrt(model.d_model)
            tgt_emb = model.pos_encoder(tgt_emb)
            tgt_emb = tgt_emb.transpose(0, 1)  # [tgt_len, 1, d_model]

            # Compute decoder initial token using atom type features
            atom_emb = model.atom_type_proj(atom_types)  # [1, d_model]
            decoder_init = model.decoder_init_proj(atom_emb).unsqueeze(0)  # [1, 1, d_model]

            # Concatenate decoder initial token with current tgt_emb
            decoder_input = torch.cat([decoder_init, tgt_emb], dim=0)  # [1+tgt_len, 1, d_model]

            # Generate autoregressive mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.size(0)).to(device)

            # Decoder computation
            decoder_output = model.decoder(decoder_input, memory, tgt_mask=tgt_mask)
            # Remove decoder initial token
            decoder_output = decoder_output[1:, 0, :]  # [tgt_len, d_model]

            # Get logits distribution from last position
            logits = model.output_linear(decoder_output[-1, :])  # [vocab_size]

            # ============================
            # Apply atom count constraints based on additional_info
            # ============================
            # Current generation sequence (excluding <SOS>)
            current_tokens = input_tgt.squeeze(0).tolist()[1:]
            # Count each atom type (assuming 'C','N','O','F' correspond to unique tokens in vocabulary)
            count_C = current_tokens.count(char2idx['C'])
            count_N = current_tokens.count(char2idx['N'])
            count_O = current_tokens.count(char2idx['O'])
            count_F = current_tokens.count(char2idx['F'])

            # Set logits to -∞ for atoms that have reached their limits
            if count_C >= additional_info[0]:
                logits[char2idx['C']] = -float('inf')
            if count_N >= additional_info[1]:
                logits[char2idx['N']] = -float('inf')
            if count_O >= additional_info[2]:
                logits[char2idx['O']] = -float('inf')
            if count_F >= additional_info[3]:
                logits[char2idx['F']] = -float('inf')

            # Select next token (using temperature sampling for randomness)
            temperature = 1.0
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).unsqueeze(0)  # [1, 1]

            # Append to input sequence
            input_tgt = torch.cat([input_tgt, next_token], dim=1)

            # Early termination if <EOS> is generated
            if next_token.item() == char2idx['<EOS>']:
                break

        # Return generated token sequence (excluding <SOS>)
        generated_tokens = input_tgt[0, 1:].tolist()
        return generated_tokens



def obtain_HRMS_feature(smiles):
    """
    检索对应当前pred分子对应gt分子的 DBE、精确分子量（对数归一化）以及前五周期各元素的计数，
    返回向量维度为：2 + len(atomic_weights)。
    """
    gt_smiles = smiles
    mol = Chem.MolFromSmiles(gt_smiles)
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


import random
def padding_or_deleting_atoms(input_smiles, input_atom_count, gt_atom_count):
    """
    根据 gt_atom_count 调整 input_smiles 的原子数量（C, N, O, F），
    直接字符级插入或删除，不保证生成的 SMILES 合法性。
    """
    smiles_chars = list(input_smiles)
    atom_symbols = ['C', 'N', 'O', 'F']

    for i, atom_symbol in enumerate(atom_symbols):
        diff = gt_atom_count[i] - input_atom_count[i]

        if diff > 0:  # 缺原子 -> 插入
            for _ in range(diff):
                pos = random.randint(0, len(smiles_chars))
                smiles_chars.insert(pos, atom_symbol)

        elif diff < 0:  # 多原子 -> 删除
                        for _ in range(-diff):
                positions = [idx for idx, ch in enumerate(smiles_chars) if ch == atom_symbol]
                if positions:
                    smiles_chars.pop(random.choice(positions))

    return "".join(smiles_chars)


def inference(input_smiles: str, atom_types: List[float], additional_info: List[int], 
             model: nn.Module, device: torch.device, max_seq_length: int, 
             char2idx: Dict[str, int], idx2char: Dict[int, str], mask_prob: float = 0.15) -> Tuple[str, List[int], List[int]]:
    """
    Inference function that performs SMILES generation with masking.
    
    Args:
        input_smiles (str): Input SMILES string
        atom_types (List[float]): Atomic type features
        additional_info (List[int]): Target atom counts [C, N, O, F]
        model (nn.Module): Trained model
        device (torch.device): Computation device
        max_seq_length (int): Maximum sequence length
        char2idx (Dict[str, int]): Character to index mapping
        idx2char (Dict[int, str]): Index to character mapping
        mask_prob (float): Probability of masking tokens
        
    Returns:
        Tuple containing:
        - generated_smiles: Model-generated SMILES string
        - src_indices: Masked token index sequence
        - true_indices: Original complete token index sequence
    """
    input_atom_count = compute_additional_info(input_smiles)
    if input_atom_count != additional_info:
        input_smiles = padding_or_deleting_atoms(input_smiles, input_atom_count, additional_info)
    
    true_indices = smiles_to_indices(input_smiles, char2idx, max_seq_length)
    src_indices, _ = mask_smiles_indices(true_indices, char2idx['<MASK>'], char2idx, mask_prob=mask_prob)
    
    src_tensor = torch.tensor([src_indices], dtype=torch.long, device=device)
    atom_types_tensor = torch.tensor([atom_types], dtype=torch.float32, device=device)
    
    generated_tokens = generate_smiles(model, src_tensor, atom_types_tensor, max_length=max_seq_length, 
                                      char2idx=char2idx, additional_info=additional_info, device=device)
    generated_smiles = decode_indices(generated_tokens, idx2char)
    
    return generated_smiles, src_indices, true_indices


def compute_additional_info(smiles: str) -> List[int]:
    """
    Compute counts of C, N, O, F atoms in a given SMILES molecule.
    
    Args:
        smiles (str): SMILES string representation
        
    Returns:
        List[int]: Atom counts in order [C, N, O, F]. Returns [0, 0, 0, 0] if SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0, 0, 0, 0]
    
    count_C = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    count_N = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
    count_O = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
    count_F = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
    
    return [count_C, count_N, count_O, count_F]


def compute_atom_count(smiles: str) -> Dict[str, int]:
    """
    Calculate counts of C, N, O, F atoms in SMILES string.
    
    Args:
        smiles (str): SMILES string representation
        
    Returns:
        Dict[str, int]: Dictionary with atom counts for C, N, O, F
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # If RDKit parsing fails, use string-based counting
        count_C = smiles.count('C')
        count_N = smiles.count('N')
        count_O = smiles.count('O')
        count_F = smiles.count('F')
        return {"C": count_C, "N": count_N, "O": count_O, "F": count_F}

    atom_count = {"C": 0, "N": 0, "O": 0, "F": 0}
    
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in atom_count:
            atom_count[symbol] += 1

    return atom_count


# ============================
# Main Inference Function
# ============================
if __name__ == '__main__':
    
    infer_one_sample = False  # Whether to perform single sample inference or batch inference
    
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Hyperparameter settings (consistent with training)
    max_seq_length = 400
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1

    # Instantiate model and load pretrained weights
    model = MoleculePretrainingModel(
        vocab_size, atom_type_dim, d_model=d_model, nhead=nhead,
        num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward, dropout=dropout
    )
    model = model.to(device)

    # 加载预训练模型权重
    #checkpoint_path = "pre_large_weights/molecule_pretraining_model_epoch7.pth"
    # checkpoint_path = "finetuned_molecule_pretraining_model_epoch10.pth"
    checkpoint_path = "./repair_old_finetuned_molecule_pretraining_model_epoch10.pth"
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        # 如果权重文件是在 DDP 下保存的，可能带有 "module." 前缀，此处需要去除
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, True)
        print(f"Loaded model weights from {checkpoint_path}")
    else:
        print(f"Checkpoint {checkpoint_path} not found. Using randomly initialized weights.")

    model.eval()
    

    if infer_one_sample:
        # 输入一个 SMILES 字符串
        # input_smiles = "CCOC(=O)C1=CC=C1"
        input_smiles = "COC(=O)CCC(C)=N"
        true_atom_counts = {"C": 9, "N": 2, "O": 2, "F": 0}
        print("Input SMILES:", input_smiles)

        
        # 进行推理
        generated_smiles, src_indices, true_indices = inference(
            input_smiles, model, device, max_seq_length, char2idx, idx2char, mask_prob=0.5,
        )

        # 解码 mask 后的 SMILES 以便查看（注意：mask 位置会显示为 <MASK>）
        masked_smiles = decode_indices(src_indices, idx2char)
        true_smiles = decode_indices(true_indices, idx2char)

        print("Masked SMILES input:", masked_smiles)
        print("Original SMILES    :", true_smiles)
        print("Generated SMILES   :", generated_smiles)

        # ---------------------------
        # 1. 计算原子数量 (C, N, O, F)
        pred_atom_counts = compute_atom_count(generated_smiles)

        print("\n=== Atom Counts ===")
        print("True molecule atom counts:")
        for atom in ['C', 'N', 'O', 'F']:
            print(f"  {atom}: {true_atom_counts.get(atom, 0)}")
            
        print("Predicted molecule atom counts:")
        for atom in ['C', 'N', 'O', 'F']:
            print(f"  {atom}: {pred_atom_counts.get(atom, 0)}")

        # 检查真实与预测的原子数量是否匹配
        atom_match = all(true_atom_counts.get(atom, 0) == pred_atom_counts.get(atom, 0) for atom in ['C', 'N', 'O', 'F'])

        # ---------------------------
        # 2. 计算合法性评估
        # 利用 RDKit 判断生成的 SMILES 是否为有效分子
        from rdkit import Chem

        def is_valid_smiles(smiles):
            """判断 SMILES 是否有效，返回 True 或 False"""
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None

        validity = is_valid_smiles(generated_smiles)
        print("\nGenerated SMILES 合法性评估:", "有效" if validity else "无效")

        # ---------------------------
        # 3. 计算 BLEU 得分
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoothie = SmoothingFunction().method4

        # 这里将 SMILES 字符串视为字符序列，当然你也可以采用自定义的 tokenization 策略
        reference = [list(true_smiles)]
        candidate = list(generated_smiles)
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        print("\nBLEU score:", bleu_score)

    else:
        # 批量推理，采样N个不同结果，去重、筛选、排序并输出csv
        input_file = "./fangyang/gp/csv/corr_draw/temperature_sampling_results.csv"
        output_file = "./fangyang/gp/csv/corr_draw/inference_topN_candidates.csv"
        N = 100  # 每个smiles采样N个不同结果

        from rdkit.Chem import AllChem
        from rdkit import DataStructs

        def is_valid_smiles(smiles):
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None

        def get_fingerprint(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

        df = pd.read_csv(input_file)
        smiles_list = df['best_predicted_smiles'].tolist()
        all_results = []
        for smiles in tqdm(smiles_list):
            try:
                true_smiles = df[df['best_predicted_smiles']==smiles]['true_smiles'].values[0]
                atom_types = obtain_HRMS_feature(true_smiles)
                additional_info = compute_additional_info(true_smiles)
                candidates = set()
                candidate_info = []
                pred_fp = get_fingerprint(smiles)
                true_atom_counts = compute_atom_count(true_smiles)
                # 多次采样，使用不同的mask_prob增加多样性
                for i in range(N*2):  # 采样多一些，后续去重
                    # mask_prob从0到0.5变化，增加多样性
                    mask_prob = i / (N*2) * 0.5  # 从0.0到0.5线性变化
                    generated_smiles, src_indices, true_indices = inference(
                        smiles, atom_types, additional_info, model, device, max_seq_length, char2idx, idx2char, mask_prob=mask_prob,
                    )
                    # 跳过包含<UNK>的候选
                    if '<UNK>' in generated_smiles:
                        continue
                    # 跳过不合法的候选
                    valid = is_valid_smiles(generated_smiles)
                    if not valid:
                        continue
                    if generated_smiles in candidates:
                        continue
                    candidates.add(generated_smiles)
                    candidates.add(generated_smiles)
                    # 原子数
                    pred_atom_counts = compute_atom_count(generated_smiles)
                    atom_match = all(true_atom_counts.get(atom, 0) == pred_atom_counts.get(atom, 0) for atom in ['C', 'N', 'O', 'F'])
                    # tanimoto
                    cand_fp = get_fingerprint(generated_smiles)
                    if pred_fp is not None and cand_fp is not None:
                        tanimoto = DataStructs.TanimotoSimilarity(pred_fp, cand_fp)
                    else:
                        tanimoto = 0.0
                    candidate_info.append({
                        'input_smiles': smiles,
                        'true_smiles': true_smiles,
                        'candidate': generated_smiles,
                        'valid': True,  # 所有候选都是合法的
                        'atom_match': atom_match,
                        'tanimoto': tanimoto
                    })
                    if len(candidates) >= N:
                        break
                # 排序：valid>atom_match>tanimoto（不做筛选，只做排序）
                candidate_info = sorted(candidate_info, key=lambda x: (-x['valid'], -x['atom_match'], -x['tanimoto']))
                
                # 打印当前predicted_smiles的所有候选结果
                print(f"\n=== Predicted SMILES: {smiles} ===")
                print(f"Ground Truth SMILES: {true_smiles}")
                print(f"Total candidates generated: {len(candidate_info)}")
                print("Top candidates (ranked by valid > atom_match > tanimoto):")
                for i, c in enumerate(candidate_info[:N], 1):
                    print(f"  Rank {i}: {c['candidate']}")
                    print(f"    Valid: {c['valid']}, Atom Match: {c['atom_match']}, Tanimoto: {c['tanimoto']:.4f}")
                
                # 计算BLEU分数：检查是否有与ground-truth完全一致的候选
                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                smoothie = SmoothingFunction().method4
                
                has_exact_match = any(c['candidate'] == true_smiles for c in candidate_info)
                if has_exact_match:
                    bleu_score = 1.0
                    print(f"Found exact match with ground truth! BLEU = 1.0")
                else:
                    # 用top-1候选计算BLEU
                    if candidate_info:
                        best_candidate = candidate_info[0]['candidate']
                        reference = [list(true_smiles)]
                        candidate = list(best_candidate)
                        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
                        print(f"No exact match. BLEU with top-1 candidate: {bleu_score:.4f}")
                    else:
                        bleu_score = 0.0
                        print(f"No candidates generated. BLEU = 0.0")
                
                # 只保留top-N
                for c in candidate_info[:N]:
                    c['bleu_score'] = bleu_score  # 添加BLEU分数到结果中
                    all_results.append(c)
            except Exception as e:
                print(f"Error processing SMILES '{smiles}': {e}")
        # 输出到csv
        out_df = pd.DataFrame(all_results)
        out_df.to_csv(output_file, index=False)
        print(f"已输出所有候选到 {output_file}")
        
        # 计算overall BLEU score
        print("\n=== Overall BLEU Score Calculation ===")
        unique_smiles_results = {}
        
        # 按input_smiles分组，每个predicted_smiles只保留一个结果
        for result in all_results:
            input_smiles = result['input_smiles']
            if input_smiles not in unique_smiles_results:
                unique_smiles_results[input_smiles] = result
        
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoothie = SmoothingFunction().method4
        
        def are_same_molecule(smiles1, smiles2):
            """判断两个SMILES是否代表同一个分子（使用canonical SMILES）"""
            try:
                mol1 = Chem.MolFromSmiles(smiles1)
                mol2 = Chem.MolFromSmiles(smiles2)
                if mol1 is None or mol2 is None:
                    return smiles1 == smiles2  # 如果无法解析，直接字符串比较
                canonical1 = Chem.MolToSmiles(mol1, canonical=True)
                canonical2 = Chem.MolToSmiles(mol2, canonical=True)
                return canonical1 == canonical2
            except:
                return smiles1 == smiles2
        
        total_bleu = 0.0
        exact_matches = 0
        total_samples = len(unique_smiles_results)
        
        for input_smiles, result in unique_smiles_results.items():
            true_smiles = result['true_smiles']
            best_candidate = result['candidate']
            
            # 检查是否是同一个分子
            if are_same_molecule(best_candidate, true_smiles):
                bleu_score = 1.0
                exact_matches += 1
            else:
                # 计算BLEU分数
                reference = [list(true_smiles)]
                candidate = list(best_candidate)
                bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
            
            total_bleu += bleu_score
        
        overall_bleu = total_bleu / total_samples if total_samples > 0 else 0.0
        exact_match_rate = exact_matches / total_samples if total_samples > 0 else 0.0
        
        print(f"Total samples: {total_samples}")
        print(f"Exact matches (same molecule): {exact_matches}")
        print(f"Exact match rate: {exact_match_rate:.4f}")
        print(f"Overall BLEU score: {overall_bleu:.4f}")


