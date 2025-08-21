"""
SpectroMol Robustness Analysis Module

This module performs robustness analysis of the SpectroMol model by evaluating
performance under various perturbations and noise conditions in spectral data.

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


# Predefined SMILES character vocabulary
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






# Define inference function including BLEU score calculation and attention analysis
# Define inference function including BLEU score calculation and attention analysis for each molecule
def inference_with_analysis(model, dataloader, char2idx, idx2char, max_seq_length=100, save_dir='corr_draw'):
    model.eval()
    smoothie = SmoothingFunction().method4


    total_bleu_score = 0.0
    total_cos_score = 0.0
    total_correct = 0
    total_smiles = 0
    bad_sm_count = 0
    n_exact = 0
    maccs_sim, rdk_sim, morgan_sim, levs = [], [], [], []

    bleu_scores = []  # Store BLEU scores for each molecule

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Unpack batch data
            ir_spectrum, raman_spectrum, c_spectrum, h_spectrum, low_res_mass, high_res_mass, \
            smiles_indices, auxiliary_targets, atom_types, coordinates = batch

            # Move data to device
            ir_spectrum = ir_spectrum.to(device)
            raman_spectrum = raman_spectrum.to(device)
            c_spectrum = c_spectrum.to(device)
            h_spectrum = h_spectrum.to(device)
            low_res_mass = low_res_mass.to(device)
            high_res_mass = high_res_mass.to(device)
            smiles_indices = smiles_indices.to(device)
            atom_types = atom_types.to(device) if atom_types is not None else None

            batch_size = ir_spectrum.size(0)

            # Get true SMILES strings
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

            # Use model for prediction, get predicted SMILES strings
            predicted_smiles_list = inference(
                model,
                ir_spectrum,
                raman_spectrum,
                c_spectrum,
                h_spectrum,
                low_res_mass,
                high_res_mass,
                char2idx,
                idx2char,
                max_seq_length=max_seq_length,
                atom_types=atom_types
            )

            # For each sample in the batch
            for i in range(batch_size):
                true_smiles_str = true_smiles_list[i]
                predicted_smiles_str = predicted_smiles_list[i]

                


                # For validity
                try:
                    tmp = Chem.CanonSmiles(predicted_smiles_str)
                except:
                    bad_sm_count += 1

                try:
                    mol_output = Chem.MolFromSmiles(predicted_smiles_str)
                    mol_gt = Chem.MolFromSmiles(true_smiles_str)
                    if Chem.MolToInchi(mol_output) == Chem.MolToInchi(mol_gt):
                        n_exact += 1
                    maccs_sim.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(mol_output), MACCSkeys.GenMACCSKeys(mol_gt), metric=DataStructs.TanimotoSimilarity))
                    rdk_sim.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(mol_output), Chem.RDKFingerprint(mol_gt), metric=DataStructs.TanimotoSimilarity))
                    morgan_sim.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(mol_output, 2), AllChem.GetMorganFingerprint(mol_gt, 2)))
                except:
                    pass


                # Compute SMILES accuracy
                try:
                    tmp_1 = Chem.CanonSmiles(predicted_smiles_str)
                    tmp_2 = Chem.CanonSmiles(true_smiles_str)
                    if tmp_1 == tmp_2:
                        total_correct += 1
                except:
                    if predicted_smiles_str == true_smiles_str:
                        total_correct += 1
                total_smiles += 1


                # Compute BLEU score
                if tmp_1 == tmp_2:
                    bleu_score = 1.0
                else:
                    reference = [list(true_smiles_str)]
                    candidate = list(predicted_smiles_str)
                    bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
                total_bleu_score += bleu_score


                # Compute Cos-sim
                cos_sim = cosine_similarity(predicted_smiles_str, true_smiles_str)
                total_cos_score += cos_sim

                # Compute L-Distance
                l_dis = lev(predicted_smiles_str, true_smiles_str)
                levs.append(l_dis)






                # Save BLEU score for each molecule
                bleu_scores.append({
                    'true_smiles': true_smiles_str,
                    'predicted_smiles': predicted_smiles_str,
                    'bleu_score': bleu_score
                })

                # # Call analyze_feature_correlations function, save analysis results
                # if true_smiles_str == 'O=C1CCC2(NC12)C#N':
                #     analyze_feature_correlations(
                #         model,
                #         ir_spectrum[i].unsqueeze(0),
                #         raman_spectrum[i].unsqueeze(0),
                #         c_spectrum[i].unsqueeze(0),
                #         h_spectrum[i].unsqueeze(0),
                #         low_res_mass[i].unsqueeze(0),
                #         high_res_mass[i].unsqueeze(0),
                #         atom_types=atom_types[i].unsqueeze(0) if atom_types is not None else None,
                #         smiles=true_smiles_str,
                #         save_dir=save_dir
                #     )
    
    # Save metrics
    avg_bleu_score = total_bleu_score / total_smiles
    accuracy = total_correct / total_smiles
    # Compute validity
    validity  = (total_smiles - bad_sm_count) / total_smiles
    cos_sim_all = total_cos_score / total_smiles
    exact = n_exact * 1.0 / total_smiles

    print(f'BLEU: {avg_bleu_score}, Top-1 Acc: {accuracy}, Validity: {validity}, Cos-similarity: {cos_sim_all}, Exact: {exact}, Levenshtein: {np.mean(levs)}, MACCS FTS: {np.mean(maccs_sim)}, RDKit FTS: {np.mean(rdk_sim)}, Morgan FTS: {np.mean(morgan_sim)}.')

    # # Save BLEU scores as CSV file
    # import csv
    # with open(os.path.join(save_dir, 'bleu_scores_cano.csv'), 'w', newline='') as csvfile:
    #     fieldnames = ['true_smiles', 'predicted_smiles', 'bleu_score']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for row in bleu_scores:
    #         writer.writerow(row)

    return bleu_scores  # Return list of BLEU scores for each molecule







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
            ir_spectrum, raman_spectrum, c_spectrum, h_spectrum, low_res_mass, high_res_mass, \
            smiles_indices, auxiliary_targets, atom_types, coordinates = batch

            ir_spectrum = ir_spectrum.to(device)
            raman_spectrum = raman_spectrum.to(device)
            c_spectrum = c_spectrum.to(device)
            h_spectrum = h_spectrum.to(device)
            low_res_mass = low_res_mass.to(device)
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



            # Extract atom_types[:, 2:] as required atom counts for each sample
            # Assume atom_types[:, 2:] order is [C, N, O, F]
            atom_counts_array = atom_types[:, 2:].cpu().numpy()  # [batch_size, 4]
            required_atom_counts = []
            for counts in atom_counts_array:
                # Convert to dictionary
                req_dict = dict(zip(['C', 'N', 'O', 'F'], counts))
                required_atom_counts.append(req_dict)

            # Inference
            predicted_smiles_list = inference(
                model,
                ir_spectrum,
                raman_spectrum,
                c_spectrum,
                h_spectrum,
                low_res_mass,
                high_res_mass,
                char2idx,
                idx2char,
                max_seq_length=100,
                atom_types=atom_types,
                required_atom_counts=required_atom_counts,
                beam_size=5,  # Custom beam_size
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
    with open(os.path.join(save_dir, 'bleu_scores_new.csv'), 'w', newline='') as csvfile:
        fieldnames = ['true_smiles', 'predicted_smiles', 'bleu_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in bleu_scores:
            writer.writerow(row)

    return bleu_scores








# Define single sample inference function
def inference_(model, ir_spectrum, raman_spectrum, c_spectrum, h_spectrum,
              low_res_mass, high_res_mass, char2idx, idx2char, max_seq_length=100, atom_types=None):
    model.eval()
    with torch.no_grad():
        # Split h_spectrum into h_spectrum_part, f_spectrum, n_spectrum
        h_spectrum_part = h_spectrum[:, :595]
        f_spectrum = h_spectrum[:, 595:607]
        n_spectrum = h_spectrum[:, 607:621]
        o_spectrum = h_spectrum[:, 621:]

        features = {
            'ir': ir_spectrum,
            'raman': raman_spectrum,
            'nmr_c': c_spectrum,
            'nmr_h': h_spectrum_part,
            'f_spectrum': f_spectrum,
            'n_spectrum': n_spectrum,
            'o_spectrum': o_spectrum,
            # 'mass_low': low_res_mass,
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

        # Initialize input sequence with <SOS> token
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
    Perform constraint checking on a single candidate sequence, and return a valid_tokens boolean vector indicating available tokens.
    candidate_seq: list of int (token indices)
    required_atom_counts: dict {'C':int, 'N':int, 'O':int, 'F':int}
    open_rings: dict for ring status {'1':0,'2':0,'3':0,'4':0,'5':0}
    """
    # Remove <PAD>, <SOS>, <EOS> from candidate_seq
    seq_chars = []
    for t in candidate_seq:
        if t == char2idx['<EOS>']:
            break
        if t not in [char2idx['<PAD>'], char2idx['<SOS>']]:
            seq_chars.append(idx2char[t])
    
    # Initialize valid_tokens
    vocab_size = len(idx2char)
    valid_tokens = [True] * vocab_size

    current_atom_counts = {'C':0,'N':0,'O':0,'F':0}
    for tok in seq_chars:
        if tok in current_atom_counts:
            current_atom_counts[tok] += 1

    # Atom count constraints
    for atom, req_count in required_atom_counts.items():
        if current_atom_counts[atom] >= req_count:
            a_idx = char2idx[atom]
            valid_tokens[a_idx] = False

    # Parentheses matching
    open_p = seq_chars.count('(')
    close_p = seq_chars.count(')')
    if close_p >= open_p:
        # Cannot generate ')' anymore
        rp_idx = char2idx.get(')', None)
        if rp_idx is not None:
            valid_tokens[rp_idx] = False

    # Avoid consecutive bonds ('#','=')
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


def inference(
    model, ir_spectrum, raman_spectrum, c_spectrum, h_spectrum,
    low_res_mass, high_res_mass, char2idx, idx2char,
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
        h_spectrum_part = h_spectrum[:, :595]
        f_spectrum = h_spectrum[:, 595:607]
        n_spectrum = h_spectrum[:, 607:621]
        o_spectrum = h_spectrum[:, 621:]

        # Prepare features
        features = {
            'ir': ir_spectrum,
            'raman': raman_spectrum,
            'nmr_c': c_spectrum,
            'nmr_h': h_spectrum_part,
            'f_spectrum': f_spectrum,
            'n_spectrum': n_spectrum,
            'o_spectrum': o_spectrum,
            # 'mass_low': low_mass,
            'mass_high': high_res_mass
        }

        batch_size = ir_spectrum.size(0)





        # # 对除 mass_high 外的所有谱数据随机 mask 掉 5% 的元素
        # for key in features:
        #     if key == 'mass_high':
        #         continue
        #     mask = (torch.rand_like(features[key]) < 0.05).float()
        #     features[key] = features[key] * (1 - mask)

        tokens = model.tokenizer(features)  # [batch_size, total_features, d_model]
        tokens = tokens.permute(1, 0, 2)       # [seq_len, batch_size, d_model]
        memory, _ = model.transformer_encoder(tokens)




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















# 定义analyze_feature_correlations分析函数
def analyze_feature_correlations(model, ir_column, nmr_c_column, ir_spectrum, raman_spectrum, c_spectrum, h_spectrum,
                                 low_res_mass, high_res_mass, atom_types=None,
                                 smiles='', save_dir='corr_draw'):
    model.eval()
    with torch.no_grad():
        # Split h_spectrum into h_spectrum_part, f_spectrum, n_spectrum
        h_spectrum_part = h_spectrum[:, :595]
        f_spectrum = h_spectrum[:, 595:607]
        n_spectrum = h_spectrum[:, 607:621]
        o_spectrum = h_spectrum[:, 621:]

        # 选择需要的 features（去掉 high-mass）
        features = {
            'ir': ir_spectrum,
            # 'raman': raman_spectrum,
            'nmr_c': c_spectrum,
            # 'nmr_h': h_spectrum_part,
            # 'f_spectrum': f_spectrum,
            # 'n_spectrum': n_spectrum,
            # 'o_spectrum': o_spectrum,
            # 'mass_low': low_res_mass,
            'mass_high': high_res_mass
        }

        # Tokenize features
        tokens = model.tokenizer(features)  # Shape: [batch_size, total_N_features, d_model]

        # Permute for transformer input: [seq_len, batch_size, d_model]
        tokens = tokens.permute(1, 0, 2)

        # Apply transformer encoder
        memory, attentions = model.transformer_encoder(tokens)  # Shape: [seq_len, batch_size, d_model]

        # 提取注意力权重
        attn_weights = attentions[0]  # Shape: [batch_size, seq_len, seq_len]
        attn_weights_example = attn_weights[0].cpu()  # Shape: [seq_len, seq_len]

        # 获取 token 的标签
        token_labels = model.tokenizer.get_token_labels()  # A list of labels for each token



        
        # **只保留peak部分的ir和cnmr的部分，去掉high-mass**
        # **1. 创建符合条件的 token 集合**
        valid_nmr_c = {f"nmr_c_{i}" for i in range(0, 27, 3)}  # 取 0, 3, 6, ..., 24
        valid_ir = {f"ir_{i}" for i in range(41)}  # 取 0 到 40

        # **2. 过滤 token_labels，保持原始顺序**
        filtered_token_labels = [
            label for label in token_labels 
            if (label in valid_nmr_c or label in valid_ir or not label.startswith("nmr_c_") and not label.startswith("ir_"))
            and "mass_high" not in label
        ]

        # **4. 过滤 attention 矩阵，确保行列匹配新的 token**
        filtered_indices = [token_labels.index(label) for label in filtered_token_labels]
        attn_weights_example_filtered = attn_weights_example[np.ix_(filtered_indices, filtered_indices)]

        # **3. 重新编号 ir 和 nmr_c，让它们从 0 开始**
        renamed_token_labels = []
        ir_counter = 0  # ir_编号从 0 开始
        nmr_c_counter = 0  # nmr_c_编号从 0 开始

        for label in filtered_token_labels:
            if label.startswith("ir_"):
                renamed_token_labels.append(f"ir_{ir_counter}")
                ir_counter += 1
            elif label.startswith("nmr_c_"):
                renamed_token_labels.append(f"nmr_c_{nmr_c_counter}")
                nmr_c_counter += 1
            else:
                renamed_token_labels.append(label)  # 其他 token 保持不变

        # **5. 创建 DataFrame 并匹配新的 token labels**
        attn_df = pd.DataFrame(attn_weights_example_filtered, index=renamed_token_labels, columns=renamed_token_labels)






        # # **1. 将 ir_column 和 nmr_c_column 转换成 attention-matrix 里的 token label**
        # selected_ir_tokens = [f"ir_{int(peak.replace('peak', ''))}" for peak in ir_column]
        # selected_nmr_c_tokens = [f"nmr_c_{int(col.replace('Is_carbonyl_C', ''))}" for col in nmr_c_column]

        # # **2. 过滤 attn_df，只保留 ir_column 和 nmr_c_column 相关行列**
        # selected_tokens = selected_ir_tokens + selected_nmr_c_tokens
        # attn_df_selected = attn_df.loc[selected_tokens, selected_tokens]




        # **绘制热图**
        plt.figure(figsize=(15, 15))
        sns.heatmap(
            attn_df,
            cmap="Reds",  # 颜色映射改为红色
            annot=False,  # 不标出数值
            cbar_kws={"shrink": 0.8}  # 颜色条大小调整
        )


        # 设置标题
        plt.title(f'Attention Weights Between Features\nSMILES: {smiles}', fontsize=16, fontweight='bold')

        # 设置 x 和 y 轴标签大小
        plt.xticks(fontsize=14, rotation=90)  # 旋转标签避免重叠
        plt.yticks(fontsize=14)

        # 确保所有刻度显示
        plt.xticks(ticks=np.arange(len(filtered_token_labels)), labels=filtered_token_labels, fontsize=14, rotation=90)
        plt.yticks(ticks=np.arange(len(filtered_token_labels)), labels=filtered_token_labels, fontsize=14)

        # 让布局更紧凑
        plt.tight_layout()

        # 处理 SMILES 字符串用于文件名
        smiles_filename = re.sub(r'[^\w\-_\. ]', '_', smiles)
        save_path = os.path.join(save_dir, f'{smiles_filename}_corr.png')

        # 保存高清图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  
        plt.close()





def load_model(model_path, vocab_size, char2idx):
    # 初始化模型
    model = AtomPredictionModel(vocab_size=vocab_size, count_tasks_classes=None, binary_tasks=None)
    model.to(device)

    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model





if __name__ == "__main__":
    # 定义模型文件路径
    model_path = '/root/workspace/smiles-transformer-master/csv/weights_scaffold_at/ir_uv_mass_nmr_scaffold.pth'

    # 加载模型
    model = load_model(model_path, vocab_size, char2idx)



    scaler = StandardScaler()

    # ir and raman
    print('load raman file...')
    raman_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/raw_spec/process_uv.csv').iloc[:, 1:]
    peak_columns = [col for col in raman_spe_filtered.columns if 'peak' in col]
    raman_spe_filtered[peak_columns] = scaler.fit_transform(raman_spe_filtered[peak_columns])
    raman_spe_filtered = raman_spe_filtered.to_numpy()
    # raman_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/raman_spe_filtered_values.csv', header=None).to_numpy()
    print('load ir file...')
    # ir_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/ir_spe_filtered_values.csv', header=None).to_numpy()
    ir_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/sparse_ir_wsmiles.csv').iloc[:, 1:]
    ir_raw = pd.read_csv('/root/workspace/smiles-transformer-master/csv/chengchunliu.csv')
    peak_columns_ir = [col for col in ir_spe_filtered.columns if 'peak' in col]
    ir_spe_filtered[peak_columns_ir] = scaler.fit_transform(ir_spe_filtered[peak_columns_ir])
    ir_spe_filtered = ir_spe_filtered.to_numpy()
    print('raman_spe_filtered:', raman_spe_filtered.shape)
    print('ir_spe_filtered:', ir_spe_filtered.shape)

    # nmr
    # nmrh_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/nmrh_spe_filtered_values.csv', header=None).to_numpy()
    # nmrc_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/nmrc_spe_filtered_values.csv', header=None).to_numpy()
    nmrc_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/sparse_cnmr_new_new.csv').to_numpy()
    twoD_nmr = pd.read_csv('/root/workspace/smiles-transformer-master/csv/2d_c_nmr.csv')
    peak_columns = [col for col in twoD_nmr.columns if 'peak' in col]
    twoD_nmr[peak_columns] = scaler.fit_transform(twoD_nmr[peak_columns])
    twoD_nmr = twoD_nmr.to_numpy()
    twod_twod = pd.read_csv('/root/workspace/smiles-transformer-master/csv/2D_NMR/2D_2D/13C_13C_INADEQUATE_DEPT/13C_13C_INADEQUATE_DEPT.csv').iloc[:, 8:]
    peak_columns = [col for col in twod_twod.columns if 'peak' in col]
    twod_twod[peak_columns] = scaler.fit_transform(twod_twod[peak_columns])
    twod_twod = twod_twod.to_numpy()
    # nmrc_spe_filtered = np.concatenate((nmrc_spe_filtered, twoD_nmr, twod_twod), axis=1)
    nmrc_spe_filtered = np.concatenate((nmrc_spe_filtered, twoD_nmr), axis=1)

    # nmrh_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/sparse_hnmr_new.csv').to_numpy()
    nmrh_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/hnmr_spin.csv').to_numpy()
    nmrf_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/sparse_fnmr.csv').to_numpy()
    nmrn_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/sparse_nnmr.csv').to_numpy()
    # nmro_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/raw_O_processed.csv').iloc[:, 2:].to_numpy()
    nmro_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/raw_O_processed.csv').iloc[:, 2:]
    peak_columns = [col for col in nmro_spe_filtered.columns if 'peak' in col]
    nmro_spe_filtered[peak_columns] = scaler.fit_transform(nmro_spe_filtered[peak_columns])
    nmro_spe_filtered = nmro_spe_filtered.to_numpy()

    # HSQC
    hsqc = pd.read_csv('/root/workspace/smiles-transformer-master/csv/HSQC/HSQC.csv').iloc[:, 8:]
    peak_columns = [col for col in hsqc.columns if 'peak' in col]
    hsqc[peak_columns] = scaler.fit_transform(hsqc[peak_columns])
    hsqc = hsqc.to_numpy()

    # COSY
    nmr_cosy = pd.read_csv('/root/workspace/smiles-transformer-master/csv/COSY/1H_1H/3J/1H_1H_3J.csv')
    hxyh_columns = [col for col in nmr_cosy.columns if 'H_X_Y_H' in col]
    nmr_cosy = nmr_cosy[hxyh_columns]
    peak_columns = [col for col in nmr_cosy.columns if 'peak' in col]
    nmr_cosy[peak_columns] = scaler.fit_transform(nmr_cosy[peak_columns])
    nmr_cosy = nmr_cosy.to_numpy()

    print('nmrh_spe_filtered:', nmrh_spe_filtered.shape)
    print('nmrc_spe_filtered:', nmrc_spe_filtered.shape)

    j2d = pd.read_csv('/root/workspace/smiles-transformer-master/csv/j2d_hnmr.csv')
    j2d_columns = [col for col in j2d.columns if 'coupling' in col]
    j2d = j2d[j2d_columns]
    # coupling_columns = [col for col in j2d.columns if 'coupling_constants' in col]
    # j2d[coupling_columns] = scaler.fit_transform(j2d[coupling_columns])
    j2d = j2d.to_numpy()

    # import pdb;pdb.set_trace()
    # nmrh_spe_filtered = np.concatenate((nmrh_spe_filtered, hsqc, nmrf_spe_filtered, nmrn_spe_filtered, nmro_spe_filtered), axis=1)
    nmrh_spe_filtered = np.concatenate((nmrh_spe_filtered, hsqc, nmr_cosy, j2d, nmrf_spe_filtered, nmrn_spe_filtered, nmro_spe_filtered), axis=1)
    # nmrh_spe_filtered = np.concatenate((nmrh_spe_filtered, hsqc, nmr_cosy, nmrf_spe_filtered, nmrn_spe_filtered), axis=1)
    # nmrc_spe_filtered = np.load('/root/workspace/smiles-transformer-master/csv/spin_nmrc_values.npy')
    # nmrh_spe_filtered = np.load('/root/workspace/smiles-transformer-master/csv/spin_nmrh_values.npy')


    # zhipu
    mass = pd.read_csv('/root/workspace/smiles-transformer-master/csv/sparse_ms_spectra_with_normalized_inten.csv')
    high_mass_spe = mass.iloc[:, 1:8].to_numpy()
    print('load high-mass file...')
    # high_mass_spe = pd.read_csv('/root/workspace/smiles-transformer-master/csv/sparse_highmass_new.csv').to_numpy()  # 128140 rows × 2 columns
    print('load low-mass file...')
    low_mass_spe = mass.iloc[:, 8:]
    # low_mass_spe = pd.read_csv('/root/workspace/smiles-transformer-master/csv/lowprecision_zhipu_values.csv', header=None).to_numpy()  # 128140 rows × 766 columns
    # low_mass_spe = pd.read_csv('/root/workspace/smiles-transformer-master/csv/sparse_lowmass.csv').to_numpy()
    # low_mass_spe = pd.read_csv('/root/workspace/smiles-transformer-master/csv/sparse_lowmass.csv')
    peak_columns = [col for col in low_mass_spe.columns if 'peak' in col]
    low_mass_spe[peak_columns] = scaler.fit_transform(low_mass_spe[peak_columns])
    low_mass_spe = low_mass_spe.to_numpy()
    print('low-mass_spe:', low_mass_spe.shape)
    print('high-mass_spe:', high_mass_spe.shape)

    # smiles
    smiles_list = pd.read_csv('/root/workspace/smiles-transformer-master/csv/aligned_smiles.csv').values.tolist()
    # smiles_list = pd.read_csv('/root/workspace/smiles-transformer-master/aligned_smiles_with_canonical_index.csv')['canonical_smiles'].tolist()

    # smiles_lengths = [len(smiles) for smiles in smiles_list]
    smiles_lengths = [len(smiles[0]) for smiles in smiles_list]
    # smiles_lengths = [3, 12, 7, 15, 11]
    max_smiles_length = max(smiles_lengths)
    # max_smiles_length = 15
    max_seq_length = max_smiles_length + 2
    # max_seq_length = 17
    print(f"SMILES 序列的最大长度为：{max_smiles_length}")
    print(f"模型中应使用的 max_seq_length 为：{max_seq_length}")



    # 获取所有辅助任务
    # Get the list of columns
    # auxiliary_data = pd.read_csv('/root/workspace/smiles-transformer-master/aligned_smiles_id_aux_task_canonical.csv')
    auxiliary_data = pd.read_csv('/root/workspace/smiles-transformer-master/aligned_smiles_id_aux_task.csv')
    columns = auxiliary_data.columns.tolist()
    # Exclude 'smiles' and 'id' columns to get auxiliary tasks
    auxiliary_tasks = [col for col in columns if col not in ['smiles', 'id']]
    print(f"Auxiliary tasks: {auxiliary_tasks}")






    atom_type = high_mass_spe[:, :-1]
    print(f"Atom type shape: {atom_type.shape}")






    # 创建 SMILES 到索引的映射
    # smiles_to_index = {smiles: idx for idx, smiles in enumerate(smiles_list)}
    smiles_to_index = {smiles[0]: idx for idx, smiles in enumerate(smiles_list)}


    # 加载训练集、验证集和测试集
    val_df = pd.read_csv(f'/root/workspace/smiles-transformer-master/csv/dataset/{data_split_mode}/val.csv')
    test_df = pd.read_csv(f'/root/workspace/smiles-transformer-master/csv/dataset/{data_split_mode}/test.csv')






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

    # 获取验证集索引
    val_indices, val_missing_smiles = get_indices(val_df['smiles'], smiles_to_index)
    # 获取测试集索引
    test_indices, test_missing_smiles = get_indices(test_df['smiles'], smiles_to_index)

    # 打印缺失的 SMILES（如果有）
    if val_missing_smiles:
        print(f"Missing smiles in val set: {val_missing_smiles}")
    if test_missing_smiles:
        print(f"Missing smiles in test set: {test_missing_smiles}")


    # 划分验证集数据
    val_ir_spe_filtered = ir_spe_filtered[val_indices]
    val_raman_spe_filtered = raman_spe_filtered[val_indices]
    val_nmrh_spe_filtered = nmrh_spe_filtered[val_indices]
    val_nmrc_spe_filtered = nmrc_spe_filtered[val_indices]
    val_low_mass_spe = low_mass_spe[val_indices]
    val_high_mass_spe = high_mass_spe[val_indices]
    val_smiles_list = [smiles_list[idx] for idx in val_indices]
    val_aux_data = auxiliary_data.iloc[val_indices].reset_index(drop=True)
    atom_types_list_val = atom_type[val_indices]

    # 划分测试集数据
    test_ir_spe_filtered = ir_spe_filtered[test_indices]
    test_raman_spe_filtered = raman_spe_filtered[test_indices]
    test_nmrh_spe_filtered = nmrh_spe_filtered[test_indices]
    test_nmrc_spe_filtered = nmrc_spe_filtered[test_indices]
    test_low_mass_spe = low_mass_spe[test_indices]
    test_high_mass_spe = high_mass_spe[test_indices]
    test_smiles_list = [smiles_list[idx] for idx in test_indices]
    test_aux_data = auxiliary_data.iloc[test_indices].reset_index(drop=True)
    atom_types_list_test = atom_type[test_indices]




    # 定义 count_tasks 和 binary_tasks
    count_tasks = [
        'H_count', 'C_count', 'N_count', 'O_count', 'F_count',
        'Alkyl_H', 'Alkenyl_H', 'Aromatic_H', 'Hetero_H',
        'Primary_C', 'Secondary_C', 'Tertiary_C', 'Alkene_C',
        'Alkyne_C', 'Cyano_C', 'Carbonyl_C', 'C_N_C',
        'alcohol_count', 'aldehyde_count', 'ketone_count',
        'amine_primary_count', 'amine_secondary_count', 'amine_tertiary_count',
        'amide_count', 'ester_count', 'ether_count', 'nitrile_count'
    ]
    binary_tasks = ['Has_Benzene', 'Has_Heterocycle', 'Has_Alkene']

    # 创建验证集数据集
    val_dataset = SpectraDataset(
        ir_spectra=val_ir_spe_filtered,
        raman_spectra=val_raman_spe_filtered,
        c_spectra=val_nmrc_spe_filtered,
        h_spectra=val_nmrh_spe_filtered,
        low_mass_spectra=val_low_mass_spe,
        high_mass_spectra=val_high_mass_spe,
        smiles_list=val_smiles_list,
        auxiliary_data=val_aux_data,
        char2idx=char2idx,
        max_seq_length=max_seq_length,
        count_tasks=count_tasks,
        binary_tasks=binary_tasks,
        atom_types_list=atom_types_list_val, 
        coordinates_list=None,
    )

    # 创建测试集数据集
    test_dataset = SpectraDataset(
        ir_spectra=test_ir_spe_filtered,
        raman_spectra=test_raman_spe_filtered,
        c_spectra=test_nmrc_spe_filtered,
        h_spectra=test_nmrh_spe_filtered,
        low_mass_spectra=test_low_mass_spe,
        high_mass_spectra=test_high_mass_spe,
        smiles_list=test_smiles_list,
        auxiliary_data=test_aux_data,
        char2idx=char2idx,
        max_seq_length=max_seq_length,
        count_tasks=count_tasks,
        binary_tasks=binary_tasks,
        atom_types_list=atom_types_list_test, 
        coordinates_list=None
    )


    from torch.utils.data import DataLoader

    # 创建验证集数据加载器
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=128, 
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

    # 创建测试集数据加载器
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4
    )







    # 进行推理并计算BLEU得分，保存注意力分析结果
    avg_bleu_score = inference_with_analysis(
        model,
        test_dataloader,
        char2idx,
        idx2char,
        max_seq_length=100,
        save_dir='/root/workspace/smiles-transformer-master/csv/corr_draw'  # 保存注意力图的目录
    )

    # print(f"Average BLEU score on test set: {avg_bleu_score}")
