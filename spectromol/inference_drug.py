"""
SpectroMol Drug Inference Module

This module provides specialized inference functionality for drug molecular structure 
elucidation from multi-modal spectral data.

Author: SpectroMol Team
"""

import os
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import re
from collections import Counter
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

# RDKit imports for molecular processing
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
RDLogger.DisableLog('rdApp.*')

# NLP and distance metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from Levenshtein import distance as lev
from sklearn.preprocessing import StandardScaler

# Local imports
from model import *
from dataset import *
from metrics import *


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_split_mode = 'scaffold'


# Predefined SMILES character vocabulary for drug molecules
SMILES_VOCAB = [
    '<PAD>', '<SOS>', '<EOS>', '<UNK>',
    'C', 'N', 'O', 'F',
    '1', '2', '3', '4', '5',
    '#', '=', '(', ')',
]

vocab_size = len(SMILES_VOCAB)

# Create character to index mapping and index to character mapping
char2idx = {token: idx for idx, token in enumerate(SMILES_VOCAB)}
idx2char = {idx: token for idx, token in enumerate(SMILES_VOCAB)}







def inference_with_analysis(model, dataloader, char2idx, idx2char, max_seq_length=100, save_dir='corr_draw'):
    model.eval()
    smoothie = SmoothingFunction().method4

    total_bleu_score = 0.0
    total_cos_score = 0.0
    total_correct = 0
    total_smiles = 0
    bad_sm_count = 0

    maccs_sim_list = []
    rdk_sim_list = []
    morgan_sim_list = []
    levs = []
    mw_diffs = []
    formula_accs = []
    formula_diffs = []
    top1_accs = []
    mcs_ratios = []
    mcs_tanimotos = []
    mcs_coefficients = []
    sensitivity_list = []
    specificity_list = []

    bleu_scores = []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            ir_spectrum, uv_spectrum, c_spectrum, h_spectrum, high_res_mass, \
            smiles_indices, auxiliary_targets, atom_types = batch


            ir_spectrum = ir_spectrum.to(device)
            uv_spectrum = uv_spectrum.to(device)
            c_spectrum = c_spectrum.to(device)
            h_spectrum = h_spectrum.to(device)
            high_res_mass = high_res_mass.to(device)
            smiles_indices = smiles_indices.to(device)
            atom_types = atom_types.to(device) if atom_types is not None else None

            batch_size = ir_spectrum.size(0)

            true_smiles_list = []
            for i in range(batch_size):
                true_indices = smiles_indices[i]
                true_smiles_tokens = []
                for idx in true_indices:
                    idx = idx.item()
                    if idx == char2idx['<EOS>']:
                        break
                    elif idx not in [char2idx['<PAD>'], char2idx['<SOS>']]:
                        true_smiles_tokens.append(idx2char.get(idx, '<UNK>'))
                true_smiles_str = ''.join(true_smiles_tokens)
                true_smiles_list.append(true_smiles_str)



            # atom_types order is [C, N, O, F]
            atom_counts_array = atom_types[:, 1:].cpu().numpy()  # [batch_size, 4]
            required_atom_counts = []
            for counts in atom_counts_array:
                # 转为字典
                req_dict = dict(zip(['C', 'N', 'O', 'F'], counts))
                required_atom_counts.append(req_dict)

            # 推断
            predicted_smiles_list = inference_beam(
                model,
                ir_spectrum,
                uv_spectrum,
                c_spectrum,
                h_spectrum,
                high_res_mass,
                char2idx,
                idx2char,
                max_seq_length=100,
                atom_types=atom_types,
                required_atom_counts=required_atom_counts,
                beam_size=5,  # 自定义beam_size
            )

            for i in range(batch_size):
                true_smiles_str = true_smiles_list[i]
                predicted_smiles_str = predicted_smiles_list[i]

                total_smiles += 1

                try:
                    tmp = Chem.CanonSmiles(predicted_smiles_str)
                except:
                    bad_sm_count += 1

                try:
                    tmp_1 = Chem.CanonSmiles(predicted_smiles_str)
                    tmp_2 = Chem.CanonSmiles(true_smiles_str)
                    if tmp_1 == tmp_2:
                        total_correct += 1
                except:
                    if predicted_smiles_str == true_smiles_str:
                        total_correct += 1

                reference = [list(true_smiles_str)]
                candidate = list(predicted_smiles_str)
                bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
                total_bleu_score += bleu_score

                cos_sim = cosine_similarity(predicted_smiles_str, true_smiles_str)
                total_cos_score += cos_sim

                l_dis = levenshtein_distance(predicted_smiles_str, true_smiles_str)
                levs.append(l_dis)

                try:
                    mol_output = Chem.MolFromSmiles(predicted_smiles_str)
                    mol_gt = Chem.MolFromSmiles(true_smiles_str)
                    if mol_output and mol_gt:
                        maccs_sim = maccs_similarity(predicted_smiles_str, true_smiles_str)
                        if maccs_sim is not None:
                            maccs_sim_list.append(maccs_sim)
                        rdk_sim = rdkit_similarity(predicted_smiles_str, true_smiles_str)
                        if rdk_sim is not None:
                            rdk_sim_list.append(rdk_sim)
                        morgan_sim = morgan_similarity(predicted_smiles_str, true_smiles_str)
                        if morgan_sim is not None:
                            morgan_sim_list.append(morgan_sim)
                        mcs_ratio, mcs_tanimoto, mcs_coefficient = get_max_mcs(true_smiles_str, predicted_smiles_str)
                        mcs_ratios.append(mcs_ratio)
                        mcs_tanimotos.append(mcs_tanimoto)
                        mcs_coefficients.append(mcs_coefficient)
                    else:
                        maccs_sim_list.append(0)
                        rdk_sim_list.append(0)
                        morgan_sim_list.append(0)
                        mcs_ratios.append(0)
                        mcs_tanimotos.append(0)
                        mcs_coefficients.append(0)
                except:
                    pass

                mw_diff = molecular_weight_difference(predicted_smiles_str, true_smiles_str)
                if mw_diff is not None:
                    mw_diffs.append(mw_diff)

                formula_acc = molecular_formula_accuracy(predicted_smiles_str, true_smiles_str)
                formula_accs.append(formula_acc)

                formula_diff = get_formulas_avg_distance(predicted_smiles_str, true_smiles_str, hydrogens=False)
                if formula_diff is not None:
                    formula_diffs.append(formula_diff)

                top1_acc = top1_accuracy(predicted_smiles_str, true_smiles_str)
                top1_accs.append(top1_acc)

                sensitivity, specificity = calculate_sensitivity_specificity(predicted_smiles_str, true_smiles_str)
                sensitivity_list.append(sensitivity)
                specificity_list.append(specificity)

                bleu_scores.append({
                    'true_smiles': true_smiles_str,
                    'predicted_smiles': predicted_smiles_str,
                    'bleu_score': bleu_score
                })

    avg_bleu_score = total_bleu_score / total_smiles if total_smiles > 0 else 0
    avg_accuracy = total_correct / total_smiles if total_smiles > 0 else 0
    validity = (total_smiles - bad_sm_count) / total_smiles if total_smiles > 0 else 0
    avg_cos_sim = total_cos_score / total_smiles if total_smiles > 0 else 0
    avg_levenshtein = np.mean(levs) if levs else 0
    avg_maccs_sim = np.mean(maccs_sim_list) if maccs_sim_list else 0
    avg_rdk_sim = np.mean(rdk_sim_list) if rdk_sim_list else 0
    avg_morgan_sim = np.mean(morgan_sim_list) if morgan_sim_list else 0
    avg_mw_diff = np.mean(mw_diffs) if mw_diffs else 0
    avg_formula_acc = np.mean(formula_accs) if formula_accs else 0
    avg_formula_diff = np.mean(formula_diffs) if formula_diffs else 0
    avg_top1_acc = np.mean(top1_accs) if top1_accs else 0
    avg_mcs_ratio = np.mean(mcs_ratios) if mcs_ratios else 0
    avg_mcs_tanimoto = np.mean(mcs_tanimotos) if mcs_tanimotos else 0
    avg_mcs_coefficient = np.mean(mcs_coefficients) if mcs_coefficients else 0

    all_elements = set()
    for sens in sensitivity_list:
        all_elements.update(sens.keys())
    avg_sensitivity = {element: 0 for element in all_elements}
    avg_specificity = {element: 0 for element in all_elements}

    for element in all_elements:
        sens_values = [sens[element] for sens in sensitivity_list if element in sens]
        spec_values = [spec[element] for spec in specificity_list if element in spec]
        avg_sensitivity[element] = np.mean(sens_values) if sens_values else 0
        avg_specificity[element] = np.mean(spec_values) if spec_values else 0

    print(f'BLEU: {avg_bleu_score}')
    print(f'Top-1 Accuracy: {avg_accuracy}')
    print(f'Validity: {validity}')
    print(f'Cosine Similarity: {avg_cos_sim}')
    print(f'Levenshtein Distance: {avg_levenshtein}')
    print(f'MACCS Similarity: {avg_maccs_sim}')
    print(f'RDKit Similarity: {avg_rdk_sim}')
    print(f'Morgan Similarity: {avg_morgan_sim}')
    print(f'Molecular Weight Difference: {avg_mw_diff}')
    print(f'Molecular Formula Accuracy: {avg_formula_acc}')
    print(f'Molecular Formula Difference: {avg_formula_diff}')
    print(f'MCS Ratio: {avg_mcs_ratio}')
    print(f'MCS Tanimoto: {avg_mcs_tanimoto}')
    print(f'MCS Coefficient: {avg_mcs_coefficient}')
    print('Average Sensitivity per element:')
    for element, value in avg_sensitivity.items():
        print(f'{element}: {value}')
    print('Average Specificity per element:')
    for element, value in avg_specificity.items():
        print(f'{element}: {value}')

    import csv
    with open(os.path.join(save_dir, 'bleu_scores_drug_beam_temp_at.csv'), 'w', newline='') as csvfile:
        fieldnames = ['true_smiles', 'predicted_smiles', 'bleu_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in bleu_scores:
            writer.writerow(row)

    return bleu_scores








# 定义单个样本的推理函数
def inference(model, ir_spectrum, uv_spectrum, c_spectrum, h_spectrum,
              high_res_mass, char2idx, idx2char, max_seq_length=100, atom_types=None):
    model.eval()
    with torch.no_grad():
        # Split h_spectrum into h_spectrum_part, f_spectrum, n_spectrum
        h_spectrum_part = h_spectrum[:, :382]
        f_spectrum = h_spectrum[:, 382:394]
        n_spectrum = h_spectrum[:, 394:408]
        o_spectrum = h_spectrum[:, 408:]

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
        tokens = model.tokenizer(features)  # Shape: [batch_size, total_N_features, d_model]

        # Permute for transformer input: [seq_len, batch_size, d_model]
        tokens = tokens.permute(1, 0, 2)

        # Apply transformer encoder
        memory, attention = model.transformer_encoder(tokens)  # Shape: [seq_len, batch_size, d_model]


        batch_size = ir_spectrum.size(0)
        device = ir_spectrum.device

        # 用<SOS>标记初始化输入序列
        tgt_indices = torch.full((1, batch_size), char2idx['<SOS>'], dtype=torch.long, device=device)

        generated_tokens = []

        for _ in range(max_seq_length):
            tgt_mask = model.smiles_decoder.generate_square_subsequent_mask(tgt_indices.size(0)).to(device)

            output = model.smiles_decoder(
                tgt_indices,
                memory,
                tgt_mask=tgt_mask,
                atom_types=atom_types
            )

            output_logits = output[-1, :, :]  # [batch_size, vocab_size]
            next_token = output_logits.argmax(dim=-1)  # [batch_size]
            generated_tokens.append(next_token.unsqueeze(0))
            tgt_indices = torch.cat([tgt_indices, next_token.unsqueeze(0)], dim=0)

            if (next_token == char2idx['<EOS>']).all():
                break

        generated_tokens = torch.cat(generated_tokens, dim=0)  # [seq_len, batch_size]
        generated_smiles = []
        for i in range(batch_size):
            token_indices = generated_tokens[:, i].cpu().numpy()
            tokens = []
            for idx in token_indices:
                if idx == char2idx['<EOS>']:
                    break
                elif idx not in [char2idx['<PAD>'], char2idx['<SOS>']]:
                    tokens.append(idx2char.get(idx, '<UNK>'))
            smiles_str = ''.join(tokens)
            generated_smiles.append(smiles_str)

        return generated_smiles







def apply_constraints_for_beam(candidate_seq, char2idx, idx2char, required_atom_counts, open_rings):
    """
    对单条候选序列进行约束检查，并返回一个valid_tokens布尔向量表示可用的token。
    candidate_seq: list of int (token indices)
    required_atom_counts: dict {'C':int, 'N':int, 'O':int, 'F':int}
    open_rings: dict for ring status {'1':0,'2':0,'3':0,'4':0,'5':0}
    """
    # 将candidate_seq中去掉<PAD>,<SOS>,<EOS>
    seq_chars = []
    for t in candidate_seq:
        if t == char2idx['<EOS>']:
            break
        if t not in [char2idx['<PAD>'], char2idx['<SOS>']]:
            seq_chars.append(idx2char[t])
    
    # 初始化valid_tokens
    vocab_size = len(idx2char)
    valid_tokens = [True] * vocab_size

    current_atom_counts = {'C':0,'N':0,'O':0,'F':0}
    for tok in seq_chars:
        if tok in current_atom_counts:
            current_atom_counts[tok] += 1

    # 原子计数约束
    for atom, req_count in required_atom_counts.items():
        if current_atom_counts[atom] >= req_count:
            a_idx = char2idx[atom]
            valid_tokens[a_idx] = False

    # 括号匹配
    open_p = seq_chars.count('(')
    close_p = seq_chars.count(')')
    if close_p >= open_p:
        # 不能再生成')'
        rp_idx = char2idx.get(')', None)
        if rp_idx is not None:
            valid_tokens[rp_idx] = False

    # 避免连续两个键('#','=')
    if len(seq_chars) > 0 and seq_chars[-1] in ['#','=']:
        # 下个token不能是'#','='
        hash_idx = char2idx['#']
        eq_idx = char2idx['=']
        valid_tokens[hash_idx] = False
        valid_tokens[eq_idx] = False

    # # 环标记规则 (同之前的简化逻辑)
    # # 若某数字环已出现两次，不允许再次出现
    # digits = ['1','2','3','4','5']
    # count_digits = {d:0 for d in digits}
    # for tok in seq_chars:
    #     if tok in digits:
    #         count_digits[tok] += 1

    # for d in digits:
    #     if count_digits[d] >= 2:
    #         # 禁止再次出现该数字
    #         d_idx = char2idx[d]
    #         valid_tokens[d_idx] = False

    # 将所有invalid的token置为False
    return valid_tokens


def update_ring_status_for_beam(seq_chars, open_rings):
    digits = ['1','2','3','4','5']
    count_digits = {d:0 for d in digits}
    for tok in seq_chars:
        if tok in digits:
            count_digits[tok] += 1
    for d in digits:
        c = count_digits[d]
        if c == 1:
            open_rings[d] = 1
        elif c == 2:
            open_rings[d] = 2
    return open_rings


def inference_beam(
    model, ir_spectrum, uv_spectrum, c_spectrum, h_spectrum,
    high_res_mass, char2idx, idx2char,
    max_seq_length=100, atom_types=None,
    required_atom_counts=None, beam_size=5, device='cuda'
):
    model.eval()
    with torch.no_grad():
        # h_spectrum_part = h_spectrum[:, :300]
        # f_spectrum = h_spectrum[:, 300:312]
        # n_spectrum = h_spectrum[:, 312:326]
        # o_spectrum = h_spectrum[:, 326:]
        # h_spectrum_part = h_spectrum[:, :595]
        # f_spectrum = h_spectrum[:, 595:607]
        # n_spectrum = h_spectrum[:, 607:621]
        # o_spectrum = h_spectrum[:, 621:]

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

        batch_size = ir_spectrum.size(0)

        # Tokenize features
        tokens = model.tokenizer(features)  # Shape: [batch_size, total_N_features, d_model]

        # Permute for transformer input: [seq_len, batch_size, d_model]
        tokens = tokens.permute(1, 0, 2)

        # Apply transformer encoder
        memory, attention = model.transformer_encoder(tokens)  # Shape: [seq_len, batch_size, d_model]

        if required_atom_counts is None:
            required_atom_counts = [{'C':999,'N':999,'O':999,'F':999} for _ in range(batch_size)]

        final_sequences = []
        for b in range(batch_size):
            # 取出第b个样本的memory和atom_types
            mem_single = memory[:, b:b+1, :] # [src_len, 1, d_model]
            if atom_types is not None:
                atom_single = atom_types[b:b+1, :] # [1, num_atom_types]
            else:
                atom_single = torch.zeros((1, 6), device=device)

            # 扩展到beam_size
            mem = mem_single.expand(-1, beam_size, -1) # [src_len, beam_size, d_model]
            atom_beam = atom_single.expand(beam_size, -1) # [beam_size, num_atom_types]

            # 初始化beam
            beams = [{
                'seq': [char2idx['<SOS>']],
                'log_prob': 0.0,
                'open_rings': {'1':0,'2':0,'3':0,'4':0,'5':0},
                'ended': False
            }]

            for step in range(max_seq_length):
                # 在这里对所有beam的序列进行长度对齐
                max_len = max(len(b['seq']) for b in beams)
                for b_ in beams:
                    if len(b_['seq']) < max_len:
                        # 用<PAD>补齐
                        b_['seq'].extend([char2idx['<PAD>']] * (max_len - len(b_['seq'])))

                current_length = max_len
                tgt_indices = torch.tensor([beam['seq'] for beam in beams], dtype=torch.long, device=device).T
                tgt_mask = model.smiles_decoder.generate_square_subsequent_mask(current_length).to(device)

                output = model.smiles_decoder(tgt_indices, mem, tgt_mask=tgt_mask, atom_types=atom_beam)
                # output: [seq_len, beam_size, vocab_size]
                output_logits = output[-1, :, :]  # [beam_size, vocab_size]
                

                all_expanded = []
                for i, beam in enumerate(beams):
                    if beam['ended']:
                        # 已结束的不扩展
                        all_expanded.append(beam)
                        continue

                    valid_tokens = apply_constraints_for_beam(
                        beam['seq'], char2idx, idx2char, required_atom_counts[b], beam['open_rings']
                    )

                    beam_logits = output_logits[i].clone()

                    # 约束过滤在topk前进行
                    for t_idx, valid in enumerate(valid_tokens):
                        if not valid:
                            beam_logits[t_idx] = float('-inf')

                    # 若全部无效
                    if torch.all(torch.isinf(beam_logits)):
                        continue

                    # topk选取
                    topk_values, topk_indices = torch.topk(beam_logits, beam_size, dim=-1)

                    # 展开beam
                    log_probs = F.log_softmax(beam_logits.unsqueeze(0), dim=-1)[0]
                    for val, idx_tok in zip(topk_values, topk_indices):
                        new_seq = beam['seq'] + [idx_tok.item()]
                        new_log_prob = beam['log_prob'] + log_probs[idx_tok].item()
                        seq_chars = [idx2char[t] for t in new_seq if t not in [char2idx['<PAD>'], char2idx['<SOS>'], char2idx['<EOS>']]]
                        new_open_rings = update_ring_status_for_beam(seq_chars, beam['open_rings'].copy())
                        ended = (idx_tok.item() == char2idx['<EOS>'])

                        all_expanded.append({
                            'seq': new_seq,
                            'log_prob': new_log_prob,
                            'open_rings': new_open_rings,
                            'ended': ended
                        })

                if len(all_expanded) == 0:
                    # 无法扩展
                    break

                # 从all_expanded中选出log_prob最高的beam_size条
                all_expanded = sorted(all_expanded, key=lambda x: x['log_prob'], reverse=True)
                beams = all_expanded[:beam_size]

                if all(b['ended'] for b in beams):
                    # 全部结束
                    break

            ended_beams = [beam for beam in beams if beam['ended']]
            if len(ended_beams) == 0:
                best_beam = max(beams, key=lambda x: x['log_prob'])
            else:
                best_beam = max(ended_beams, key=lambda x: x['log_prob'])

            final_seq = best_beam['seq']
            tokens_list = []
            for idx_t in final_seq:
                if idx_t == char2idx['<EOS>']:
                    break
                elif idx_t not in [char2idx['<PAD>'], char2idx['<SOS>']]:
                    tokens_list.append(idx2char[idx_t])
            smiles_str = ''.join(tokens_list)
            final_sequences.append(smiles_str)

        return final_sequences








def load_model(model_path, vocab_size, char2idx):
    # 初始化模型
    model = AtomPredictionModel(vocab_size=vocab_size, count_tasks_classes=None, binary_tasks=None)
    model.to(device)

    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device), True)
    model.eval()

    return model





if __name__ == "__main__":
    # 定义模型文件路径
    # model_path = './fangyang/gp/csv/weights_scaffold_at/0806_ft.pth'
    model_path = './fangyang/gp/csv/weights_scaffold_semantic_simple/best_semantic_supervised.pth'

    # 加载模型
    model = load_model(model_path, vocab_size, char2idx)

    
    


    # 假设我们有一个预先定义的 SMILES 字符集
    SMILES_VOCAB = ['<PAD>', '<SOS>', '<EOS>', '<UNK>',
                    'C', 'N', 'O', 'F',
                    '1', '2', '3', '4', '5',
                    '#', '=', '(', ')',
                    ]
    vocab_size = len(SMILES_VOCAB)

    # 创建字符到索引的映射和索引到字符的映射
    char2idx = {token: idx for idx, token in enumerate(SMILES_VOCAB)}
    idx2char = {idx: token for idx, token in enumerate(SMILES_VOCAB)}




    # uv
    print('load uv file...')
    uv_max_value = 15.0
    uv_spe_filtered = pd.read_csv('./gp/qm9_all_raw_spe/uv.csv')
    peak_columns = [col for col in uv_spe_filtered.columns if 'peak' in col]
    uv_spe_filtered[peak_columns] = uv_spe_filtered[peak_columns] / uv_max_value
    uv_spe_filtered = uv_spe_filtered.to_numpy()
    print('uv_spe_filtered:', uv_spe_filtered.shape)


    # ir
    print('load ir file...')
    ir_max_value = 4000.0
    ir_spe_filtered = pd.read_csv('./gp/qm9_all_raw_spe/ir_82.csv')
    peak_columns = [col for col in ir_spe_filtered.columns if 'peak' in col]
    ir_spe_filtered[peak_columns] = ir_spe_filtered[peak_columns] / ir_max_value
    ir_spe_filtered = ir_spe_filtered.to_numpy()
    print('ir_spe_filtered:', ir_spe_filtered.shape)


    # c-nmr
    print('load 1dc-nmr with dept file...')
    cnmr_max_value = 220.0
    cnmr_min_value = -10.0
    nmrc_spe_filtered = pd.read_csv('./gp/t50_drug_database/C0_to_C9_C_DEPT_NMR.csv')
    peak_columns = [col for col in nmrc_spe_filtered.columns if 'peak' in col]
    nmrc_spe_filtered[peak_columns] = (nmrc_spe_filtered[peak_columns] - cnmr_min_value) / (cnmr_max_value - cnmr_min_value)
    nmrc_spe_filtered = nmrc_spe_filtered.to_numpy()

    print('load 2dc-nmr (c-c, c-x) file...')
    cnmr_2d_max_value = 450.0
    cnmr_2d_min_value = -400.0
    twoD_nmr = pd.read_csv('./gp/t50_drug_database/2d_cnmr.csv')
    peak_columns = [col for col in twoD_nmr.columns if 'peak' in col]
    twoD_nmr[peak_columns] = (twoD_nmr[peak_columns] - cnmr_2d_min_value) / (cnmr_2d_max_value - cnmr_2d_min_value)
    twoD_nmr = twoD_nmr.to_numpy()
    nmrc_spe_filtered = np.concatenate((nmrc_spe_filtered, twoD_nmr), axis=1)
    print('nmrc_spe_filtered:', nmrc_spe_filtered.shape)



    # h-nmr
    print('load 1d h-nmr file...')
    nmrh_max_value = 12.0
    nmrh_min_value = -2.0
    nmrh_spe_filtered = pd.read_csv('./gp/t50_drug_database/C0_to_C9_spin_H_NMR.csv')
    peak_columns = [col for col in nmrh_spe_filtered.columns if 'peak' in col]

    # 过滤H-NMR异常值 - 先识别异常样本，但不对其归一化
    print('Filtering H-NMR samples with abnormal values...')
    threshold = 500.0
    nmrh_max_values = nmrh_spe_filtered[peak_columns].max(axis=1)
    h_nmr_abnormal_mask = nmrh_max_values > threshold
    h_nmr_abnormal_indices = set(np.where(h_nmr_abnormal_mask)[0])

    # 使用你设定的min和max值进行归一化（对所有样本，包括异常样本，但异常样本稍后会被过滤掉）
    nmrh_spe_filtered[peak_columns] = (nmrh_spe_filtered[peak_columns] - nmrh_min_value) / (nmrh_max_value - nmrh_min_value)
    nmrh_spe_filtered = nmrh_spe_filtered.to_numpy()


    # HSQC
    hsqc_max_value = 400.0
    hsqc_min_value = -350.0
    hsqc = pd.read_csv('./gp/t50_drug_database/2D_H-X_HSQC.csv')
    peak_columns = [col for col in hsqc.columns if 'peak' in col]
    hsqc[peak_columns] = (hsqc[peak_columns] - hsqc_min_value) / (hsqc_max_value - hsqc_min_value)
    hsqc = hsqc.to_numpy()


    # COSY
    cosy_max_value = 14.0
    cosy_min_value = -2.0
    nmr_cosy = pd.read_csv('./gp/t50_drug_database/C0_to_C9_2D_COSY.csv')
    hxyh_columns = [col for col in nmr_cosy.columns if 'H_X_Y_H' in col]
    nmr_cosy = nmr_cosy[hxyh_columns]
    peak_columns = [col for col in nmr_cosy.columns if 'peak' in col]
    nmr_cosy[peak_columns] = (nmr_cosy[peak_columns] - cosy_min_value) / (cosy_max_value - cosy_min_value)
    nmr_cosy = nmr_cosy.to_numpy()

    # # J2D
    # j2d_max_value = 30.0
    # j2d_min_value = -30.0
    # j2d = pd.read_csv('./gp/qm9_all_raw_spe/2d_h_j2d.csv')
    # j2d_columns = [col for col in j2d.columns if 'coupling' in col]
    # j2d = j2d[j2d_columns]

    # # 过滤J2D异常值 - 正确检测异常值
    # print('Filtering J2D samples with abnormal values...')
    # j2d_max_values = j2d[j2d_columns].abs().max(axis=1)
    # j2d_abnormal_mask = j2d_max_values > threshold
    # j2d_abnormal_indices = set(np.where(j2d_abnormal_mask)[0])

    # # 使用设定的min和max值进行归一化
    # j2d[j2d_columns] = (j2d[j2d_columns] - j2d_min_value) / (j2d_max_value - j2d_min_value)
    # j2d = j2d.to_numpy()

    # 合并所有异常样本索引
    # all_abnormal_indices = h_nmr_abnormal_indices.union(j2d_abnormal_indices)
    all_abnormal_indices = h_nmr_abnormal_indices
    all_abnormal_indices = sorted(list(all_abnormal_indices))
    print(f"Found {len(all_abnormal_indices)} samples with abnormal values (> {threshold})")

    if len(all_abnormal_indices) > 0:
        print(f"First 10 abnormal sample indices: {all_abnormal_indices[:10]}")


    # x-nmr
    print('load x-nmr file...')
    # F-NMR
    fnmr_max_value = 0.0001
    fnmr_min_value = -400.0
    nmrf_spe_filtered = pd.read_csv('./gp/t50_drug_database/C0_to_C9_F_NMR.csv')
    peak_columns = [col for col in nmrf_spe_filtered.columns if 'peak' in col]
    nmrf_spe_filtered[peak_columns] = (nmrf_spe_filtered[peak_columns] - fnmr_min_value) / (fnmr_max_value - fnmr_min_value)
    nmrf_spe_filtered = nmrf_spe_filtered.to_numpy()

    # N-NMR  
    nnmr_max_value = 400.0
    nnmr_min_value = -260.0
    nmrn_spe_filtered = pd.read_csv('./gp/t50_drug_database/C0_to_C9_N_NMR.csv')
    peak_columns = [col for col in nmrn_spe_filtered.columns if 'peak' in col]
    nmrn_spe_filtered[peak_columns] = (nmrn_spe_filtered[peak_columns] - nnmr_min_value) / (nnmr_max_value - nnmr_min_value)
    nmrn_spe_filtered = nmrn_spe_filtered.to_numpy()

    # O-NMR
    onmr_max_value = 460.0
    onmr_min_value = -385.0
    nmro_spe_filtered = pd.read_csv('./gp/t50_drug_database/C0_to_C9_O_NMR.csv')
    peak_columns = [col for col in nmro_spe_filtered.columns if 'peak' in col]
    nmro_spe_filtered[peak_columns] = (nmro_spe_filtered[peak_columns] - onmr_min_value) / (onmr_max_value - onmr_min_value)
    nmro_spe_filtered = nmro_spe_filtered.to_numpy()

    # combine all h-nmr and x-nmr features together
    nmrh_spe_filtered = np.concatenate((nmrh_spe_filtered, hsqc, nmr_cosy, nmrf_spe_filtered, nmrn_spe_filtered, nmro_spe_filtered), axis=1)
    # nmrh_spe_filtered = np.concatenate((nmrh_spe_filtered, hsqc, nmr_cosy, j2d, nmrf_spe_filtered, nmrn_spe_filtered, nmro_spe_filtered), axis=1)

    print('nmrh_spe_filtered:', nmrh_spe_filtered.shape)


    # zhipu
    print('load high-mass file...')
    mass = pd.read_csv('./gp/t50_drug_database/MS.csv')
    high_mass_spe = mass.to_numpy()
    print('high-mass_spe:', high_mass_spe.shape)


    # atom type
    atom_type = high_mass_spe[:, 1:-1]
    print(f"Atom type shape: {atom_type.shape}")


    # smiles
    smiles_list = pd.read_csv('./gp/t50_drug_database/smiles.csv').values.tolist() ### [[smiles1], [smiles2], ...]
    smiles_lengths = [len(smiles[0]) for smiles in smiles_list]
    max_smiles_length = max(smiles_lengths)
    max_seq_length = max_smiles_length + 2
    print(f"SMILES 序列的最大长度为：{max_smiles_length}")
    print(f"模型中应使用的 max_seq_length 为：{max_seq_length}")


    # 获取所有辅助任务
    # # Get the list of columns
    # # auxiliary_data = pd.read_csv('./fangyang/gp/csv/smiles-transformer-master/aligned_smiles_id_aux_task_canonical.csv')
    # auxiliary_data = pd.read_csv('./fangyang/gp/csv/smiles-transformer-master/aligned_smiles_id_aux_task.csv')
    # columns = auxiliary_data.columns.tolist()
    # # Exclude 'smiles' and 'id' columns to get auxiliary tasks
    # auxiliary_tasks = [col for col in columns if col not in ['smiles', 'id']]
    # print(f"Auxiliary tasks: {auxiliary_tasks}")



    # file_prefixes = {
    #     "c_nmr": './fangyang/gp/csv/smiles-transformer-master/Auxiliary_Task/C_NMR_TA.csv',
    #     "h_nmr": './fangyang/gp/csv/smiles-transformer-master/Auxiliary_Task/H_NMR_TA.csv',
    #     # "ir": './fangyang/gp/csv/smiles-transformer-master/Auxiliary_Task/IR_TA.csv',
    #     "ms": './fangyang/gp/csv/smiles-transformer-master/Auxiliary_Task/MS_TA.csv',
    # }
    # auxiliary_data = pd.DataFrame()
    # for prefix, filepath in file_prefixes.items():
    #     df = pd.read_csv(filepath).iloc[:, 3:]
    #     df.columns = [f"{prefix}_{col}" for col in df.columns]
    #     auxiliary_data = pd.concat([auxiliary_data, df], axis=1)


    auxiliary_data = pd.read_csv('./gp/aligned_smiles_id_aux_task.csv').iloc[:, 2:]


    columns = auxiliary_data.columns.tolist()
    auxiliary_tasks = [col for col in columns]
    # auxiliary_tasks = ['ring_count']

    # 从 auxiliary_data 中筛选包含 "ring" 的列
    # ring_columns = [col for col in auxiliary_data.columns if "ring" in col.lower()]
    # ring_columns = [
    #     "c_nmr_Ring_size1", "c_nmr_Ring_size2", "c_nmr_Ring_size3", "c_nmr_Ring_size4", "c_nmr_Ring_size5", "c_nmr_Ring_size6",
    #     "h_nmr_H_connected_ring_size1", "h_nmr_H_connected_ring_size2", "h_nmr_H_connected_ring_size3", "h_nmr_H_connected_ring_size4", "h_nmr_H_connected_ring_size5", "h_nmr_H_connected_ring_size6", "h_nmr_H_connected_ring_size7", "h_nmr_H_connected_ring_size8",
    # ]
    # # 只保留带有 "ring" 的特征
    # auxiliary_data = auxiliary_data[ring_columns]
    # # 更新 auxiliary_tasks 列表
    # auxiliary_tasks = ring_columns
    print(f"Auxiliary tasks: {auxiliary_tasks}")
    print(f"Number of ATs: {len(auxiliary_tasks)}")




    def get_indices(smiles_series, smiles_to_index):
        indices = []
        missing_smiles = []
        for smiles in smiles_series:
            idx = smiles_to_index.get(smiles)
            if idx is not None:
                indices.append(idx)
            else:
                missing_smiles.append(smiles)
        return indices, missing_smiles


    # 创建 SMILES 到索引的映射
    smiles_to_index = {smiles[0]: idx for idx, smiles in enumerate(smiles_list)}

    # 划分验证集数据
    val_ir_spe_filtered = ir_spe_filtered[:len(nmrh_spe_filtered)]
    val_uv_spe_filtered = uv_spe_filtered[:len(nmrh_spe_filtered)]
    val_nmrh_spe_filtered = nmrh_spe_filtered
    val_nmrc_spe_filtered = nmrc_spe_filtered
    val_high_mass_spe = high_mass_spe
    val_smiles_list = smiles_list
    # val_aux_data = auxiliary_data.iloc.reset_index(drop=True)
    atom_types_list_val = atom_type

    # 划分测试集数据
    test_ir_spe_filtered = ir_spe_filtered[:len(nmrh_spe_filtered)]
    test_uv_spe_filtered = uv_spe_filtered[:len(nmrh_spe_filtered)]
    test_nmrh_spe_filtered = nmrh_spe_filtered
    test_nmrc_spe_filtered = nmrc_spe_filtered
    test_high_mass_spe = high_mass_spe
    test_smiles_list = smiles_list
    # test_aux_data = auxiliary_data.iloc.reset_index(drop=True)
    atom_types_list_test = atom_type





    # 定义 count_tasks 和 binary_tasks
    count_tasks = [at for at in auxiliary_tasks if 'Has' not in at and 'Is' not in at]
    binary_tasks = [at for at in auxiliary_tasks if 'Has' in at or 'Is' in at]


    # 创建验证集数据集
    val_dataset = SpectraDataset(
        ir_spectra=val_ir_spe_filtered,
        uv_spectra=val_uv_spe_filtered,
        c_spectra=val_nmrc_spe_filtered,
        h_spectra=val_nmrh_spe_filtered,
        high_mass_spectra=val_high_mass_spe,
        smiles_list=val_smiles_list,
        auxiliary_data=None,
        char2idx=char2idx,
        max_seq_length=max_seq_length,
        count_tasks=count_tasks,
        binary_tasks=binary_tasks,
        atom_types_list=atom_types_list_val, 
    )

    # 创建测试集数据集
    test_dataset = SpectraDataset(
        ir_spectra=test_ir_spe_filtered,
        uv_spectra=test_uv_spe_filtered,
        c_spectra=test_nmrc_spe_filtered,
        h_spectra=test_nmrh_spe_filtered,
        high_mass_spectra=test_high_mass_spe,
        smiles_list=test_smiles_list,
        auxiliary_data=None,
        char2idx=char2idx,
        max_seq_length=max_seq_length,
        count_tasks=count_tasks,
        binary_tasks=binary_tasks,
        atom_types_list=atom_types_list_val, 
    )


    from torch.utils.data import DataLoader

    # 创建验证集数据加载器
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=128, 
        shuffle=False,
        num_workers=4,
        drop_last=True
    )

    # 创建测试集数据加载器（如果需要）
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )








    # 进行推理并计算BLEU得分，保存注意力分析结果
    avg_bleu_score = inference_with_analysis(
        model,
        test_dataloader,
        char2idx,
        idx2char,
        max_seq_length=100,
        save_dir='./fangyang/gp/csv/corr_draw'  # 保存注意力图的目录
    )

    # print(f"Average BLEU score on test set: {avg_bleu_score}")
