"""
SpectroMol Temperature Sampling Inference Module

This module provides temperature-based sampling inference for generating diverse
molecular structure candidates from spectral data.

Author: SpectroMol Team
"""

import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import re
import json
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

# Create character-to-index and index-to-character mappings
char2idx = {token: idx for idx, token in enumerate(SMILES_VOCAB)}
idx2char = {idx: token for idx, token in enumerate(SMILES_VOCAB)}


def inference_temperature_sampling(
    model, ir_spectrum, uv_spectrum, c_spectrum, h_spectrum,
    high_res_mass, char2idx, idx2char,
    max_seq_length=100, atom_types=None,
    required_atom_counts=None, temperatures=None, 
    num_candidates=50, device='cuda'
):
    """
    Generate molecular structure candidates using temperature sampling.
    
    This function generates multiple SMILES candidates by sampling from the model's
    output distribution at different temperatures, allowing for diverse predictions.
    
    Args:
        model: Trained SpectroMol model
        ir_spectrum, uv_spectrum, c_spectrum, h_spectrum, high_res_mass: Input spectral data
        char2idx (dict): Character to index mapping
        idx2char (dict): Index to character mapping
        max_seq_length (int): Maximum sequence length for generation
        atom_types: Atom type information
        required_atom_counts (dict): Required atom counts for filtering
        temperatures (list): List of temperatures for sampling
        num_candidates (int): Number of candidates to generate
        device (str): Device for computation
        
    Returns:
        list: List of generated SMILES candidates with scores
    """
    """
    Temperature sampling based inference function to generate multiple SMILES candidates
    
    Args:
        model: Trained model
        ir_spectrum, uv_spectrum, c_spectrum, h_spectrum, high_res_mass: Input spectral data
        char2idx, idx2char: Character mappings
        max_seq_length: Maximum sequence length
        atom_types: Atom type information
        required_atom_counts: Atom count constraints
        temperatures: Temperature list, if None, uses default temperature range
        num_candidates: Number of candidates to generate
        device: Device
    
    Returns:
        List of generated SMILES strings
    """
    model.eval()
    
    if temperatures is None:
        # 默认温度范围：从低到高
        temperatures = [0.6, 0.8, 1.0, 1.2, 1.5]
    
    with torch.no_grad():
        # 准备光谱特征
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

        batch_size = ir_spectrum.size(0)

        # Tokenize features
        tokens = model.tokenizer(features)  # Shape: [batch_size, total_N_features, d_model]
        # Permute for transformer input: [seq_len, batch_size, d_model]
        tokens = tokens.permute(1, 0, 2)
        # Apply transformer encoder
        memory, attention = model.transformer_encoder(tokens)  # Shape: [seq_len, batch_size, d_model]

        if required_atom_counts is None:
            required_atom_counts = [{'C':999,'N':999,'O':999,'F':999} for _ in range(batch_size)]

        all_candidates = []
        
        # Generate candidates for each sample in the batch  
        for b in range(batch_size):
            # Obtain memory and atom types for the current sample
            mem_single = memory[:, b:b+1, :] # [src_len, 1, d_model]
            if atom_types is not None:
                atom_single = atom_types[b:b+1, :] # [1, num_atom_types]
            else:
                atom_single = torch.zeros((1, 6), device=device)
            
            sample_candidates = []
            
            # Use temperature sampling to generate candidates
            for temperature in temperatures:
                # Compute the number of candidates to generate for this temperature
                candidates_per_temp = max(1, num_candidates // len(temperatures))
                if temperature == temperatures[-1]:  # The last temperature may need to fill up to num_candidates
                    candidates_per_temp = num_candidates - len(sample_candidates)
                
                for _ in range(candidates_per_temp):
                    if len(sample_candidates) >= num_candidates:
                        break
                        
                    candidate = generate_single_candidate_with_temperature(
                        model, mem_single, atom_single, char2idx, idx2char,
                        max_seq_length, required_atom_counts[b], temperature, device
                    )
                    
                    if candidate:  # if candidate is not empty
                        sample_candidates.append((candidate, temperature))
                
                if len(sample_candidates) >= num_candidates:
                    break
            
            all_candidates.append(sample_candidates)
        
        return all_candidates


def generate_single_candidate_with_temperature(
    model, memory, atom_types, char2idx, idx2char,
    max_seq_length, required_atom_counts, temperature, device
):
    """
    Using the temperature sampling method to generate a single SMILES candidate.
    """
    # <SOS> marker as the starting token
    tgt_indices = torch.full((1, 1), char2idx['<SOS>'], dtype=torch.long, device=device)
    
    generated_tokens = []
    
    for step in range(max_seq_length):
        tgt_mask = model.smiles_decoder.generate_square_subsequent_mask(tgt_indices.size(0)).to(device)
        
        output = model.smiles_decoder(
            tgt_indices,
            memory,
            tgt_mask=tgt_mask,
            atom_types=atom_types
        )
        
        output_logits = output[-1, 0, :]  # [vocab_size]
        
        # apply constraints for temperature sampling
        valid_mask = apply_constraints_for_temperature_sampling(
            [t.item() for t in tgt_indices[:, 0]], char2idx, idx2char, required_atom_counts
        )
        
        # set the invalid tokens to -inf
        masked_logits = output_logits.clone()
        for i, valid in enumerate(valid_mask):
            if not valid:
                masked_logits[i] = float('-inf')
        
        # apply temperature scaling
        if temperature > 0:
            probs = F.softmax(masked_logits / temperature, dim=-1)
            if torch.sum(probs) == 0:
                next_token = torch.tensor([char2idx['<EOS>']], device=device)
            else:
                next_token = torch.multinomial(probs, 1)
        else:
            next_token = masked_logits.argmax(dim=-1).unsqueeze(0)
        
        generated_tokens.append(next_token.item())
        tgt_indices = torch.cat([tgt_indices, next_token.unsqueeze(0)], dim=0)
        
        if next_token.item() == char2idx['<EOS>']:
            break
    
    # transform generated tokens to SMILES string
    tokens_list = []
    for idx in generated_tokens:
        if idx == char2idx['<EOS>']:
            break
        elif idx not in [char2idx['<PAD>'], char2idx['<SOS>']]:
            tokens_list.append(idx2char.get(idx, '<UNK>'))
    
    smiles_str = ''.join(tokens_list)
    return smiles_str


def apply_constraints_for_temperature_sampling(candidate_seq, char2idx, idx2char, required_atom_counts):
    """
    apply constraints to the candidate sequence for temperature sampling
    """
    # remove <PAD>,<SOS>,<EOS> from candidate_seq
    seq_chars = []
    for t in candidate_seq:
        if t == char2idx['<EOS>']:
            break
        if t not in [char2idx['<PAD>'], char2idx['<SOS>']]:
            seq_chars.append(idx2char[t])
    
    # initialize valid_tokens
    vocab_size = len(idx2char)
    valid_tokens = [True] * vocab_size

    current_atom_counts = {'C':0,'N':0,'O':0,'F':0}
    for tok in seq_chars:
        if tok in current_atom_counts:
            current_atom_counts[tok] += 1

    # check if current atom counts meet the required counts
    for atom, req_count in required_atom_counts.items():
        if current_atom_counts[atom] >= req_count:
            a_idx = char2idx[atom]
            valid_tokens[a_idx] = False

    # make sure the sequence does not end with an open parenthesis '('
    open_p = seq_chars.count('(')
    close_p = seq_chars.count(')')
    if close_p >= open_p:
        # cannot end with ')'
        rp_idx = char2idx.get(')', None)
        if rp_idx is not None:
            valid_tokens[rp_idx] = False

    # make sure the sequence does not end with a hash '#' or equal '='
    if len(seq_chars) > 0 and seq_chars[-1] in ['#','=']:
        # cannot end with '#' or '='
        hash_idx = char2idx['#']
        eq_idx = char2idx['=']
        valid_tokens[hash_idx] = False
        valid_tokens[eq_idx] = False

    return valid_tokens


def evaluate_candidates_and_select_best(true_smiles, candidates_with_temp):
    """
    evaluate the candidates generated by temperature sampling and select the best one.
    
    Args:
        true_smiles: real SMILES string for the sample
        candidates_with_temp: candidates generated by temperature sampling, each candidate is a tuple (smiles, temperature)
    
    Returns:
        dict: containing the best predicted SMILES, its temperature, BLEU score, and number of unique predictions
    """
    smoothie = SmoothingFunction().method4
    best_bleu = -1
    best_smiles = ""
    best_temperature = 0
    unique_smiles = set()
    
    for smiles, temp in candidates_with_temp:
        unique_smiles.add(smiles)
        
        reference = [list(true_smiles)]
        candidate = list(smiles)
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            best_smiles = smiles
            best_temperature = temp
    
    return {
        'best_predicted_smiles': best_smiles,
        'best_temperature': best_temperature,
        'best_bleu_score': best_bleu,
        'num_unique_predictions': len(unique_smiles)
    }


def inference_with_temperature_sampling_analysis(model, dataloader, char2idx, idx2char, 
                                                max_seq_length=100, save_dir='corr_draw',
                                                temperatures=[0.6, 0.8, 1.0, 1.2, 1.5],
                                                num_candidates=50):
    """
    Analyze the inference results using temperature sampling.
    """
    model.eval()
    
    results = []
    sample_id = 0
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Temperature Sampling Inference"):
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

            atom_counts_array = atom_types[:, 1:].cpu().numpy()  # [batch_size, 4]
            required_atom_counts = []
            for counts in atom_counts_array:
                req_dict = dict(zip(['C', 'N', 'O', 'F'], counts))
                required_atom_counts.append(req_dict)

            all_candidates = inference_temperature_sampling(
                model,
                ir_spectrum,
                uv_spectrum,
                c_spectrum,
                h_spectrum,
                high_res_mass,
                char2idx,
                idx2char,
                max_seq_length=max_seq_length,
                atom_types=atom_types,
                required_atom_counts=required_atom_counts,
                temperatures=temperatures,
                num_candidates=num_candidates,
                device=device
            )

            for i in range(batch_size):
                true_smiles = true_smiles_list[i]
                candidates_with_temp = all_candidates[i]
                
                best_result = evaluate_candidates_and_select_best(true_smiles, candidates_with_temp)
                
                result = {
                    'sample_id': sample_id,
                    'true_smiles': true_smiles,
                    'best_predicted_smiles': best_result['best_predicted_smiles'],
                    'best_temperature': best_result['best_temperature'],
                    'best_bleu_score': best_result['best_bleu_score'],
                    'num_unique_predictions': best_result['num_unique_predictions']
                }
                results.append(result)
                sample_id += 1

    csv_filename = os.path.join(save_dir, 'temperature_sampling_results.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['sample_id', 'true_smiles', 'best_predicted_smiles', 
                     'best_temperature', 'best_bleu_score', 'num_unique_predictions']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    json_filename = os.path.join(save_dir, 'temperature_sampling_results.json')
    with open(json_filename, 'w') as jsonfile:
        json.dump(results, jsonfile, indent=2)

    avg_bleu = np.mean([r['best_bleu_score'] for r in results])
    avg_unique = np.mean([r['num_unique_predictions'] for r in results])
    temp_distribution = Counter([r['best_temperature'] for r in results])
    
    print(f"Temperature Sampling Inference Complete!")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Average Unique Predictions: {avg_unique:.2f}")
    print(f"Best Temperature Distribution: {dict(temp_distribution)}")
    print(f"Results saved to: {csv_filename}")
    
    return results


def load_model(model_path, vocab_size, char2idx):
    model = AtomPredictionModel(vocab_size=vocab_size, count_tasks_classes=None, binary_tasks=None)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), True)
    model.eval()
    return model


if __name__ == "__main__":
    # model path
    model_path = './fangyang/spectromol/csv/weights_scaffold_semantic_simple/best_semantic_supervised.pth'
    
    # load the model
    model = load_model(model_path, vocab_size, char2idx)
    
    # SMILES vocabulary and size
    SMILES_VOCAB = ['<PAD>', '<SOS>', '<EOS>', '<UNK>',
                    'C', 'N', 'O', 'F',
                    '1', '2', '3', '4', '5',
                    '#', '=', '(', ')',
                    ]
    vocab_size = len(SMILES_VOCAB)

    # create character-to-index and index-to-character mappings
    char2idx = {token: idx for idx, token in enumerate(SMILES_VOCAB)}
    idx2char = {idx: token for idx, token in enumerate(SMILES_VOCAB)}

    # uv
    print('load uv file...')
    uv_max_value = 15.0
    uv_spe_filtered = pd.read_csv('./spectromol/qm9_all_raw_spe/uv.csv')
    peak_columns = [col for col in uv_spe_filtered.columns if 'peak' in col]
    uv_spe_filtered[peak_columns] = uv_spe_filtered[peak_columns] / uv_max_value
    uv_spe_filtered = uv_spe_filtered.to_numpy()
    print('uv_spe_filtered:', uv_spe_filtered.shape)

    # ir
    print('load ir file...')
    ir_max_value = 4000.0
    ir_spe_filtered = pd.read_csv('./spectromol/qm9_all_raw_spe/ir_82.csv')
    peak_columns = [col for col in ir_spe_filtered.columns if 'peak' in col]
    ir_spe_filtered[peak_columns] = ir_spe_filtered[peak_columns] / ir_max_value
    ir_spe_filtered = ir_spe_filtered.to_numpy()
    print('ir_spe_filtered:', ir_spe_filtered.shape)

    # c-nmr
    print('load 1dc-nmr with dept file...')
    cnmr_max_value = 220.0
    cnmr_min_value = -10.0
    nmrc_spe_filtered = pd.read_csv('./spectromol/t50_drug_database/C0_to_C9_C_DEPT_NMR.csv')
    peak_columns = [col for col in nmrc_spe_filtered.columns if 'peak' in col]
    nmrc_spe_filtered[peak_columns] = (nmrc_spe_filtered[peak_columns] - cnmr_min_value) / (cnmr_max_value - cnmr_min_value)
    nmrc_spe_filtered = nmrc_spe_filtered.to_numpy()

    print('load 2dc-nmr (c-c, c-x) file...')
    cnmr_2d_max_value = 450.0
    cnmr_2d_min_value = -400.0
    twoD_nmr = pd.read_csv('./spectromol/t50_drug_database/2d_cnmr.csv')
    peak_columns = [col for col in twoD_nmr.columns if 'peak' in col]
    twoD_nmr[peak_columns] = (twoD_nmr[peak_columns] - cnmr_2d_min_value) / (cnmr_2d_max_value - cnmr_2d_min_value)
    twoD_nmr = twoD_nmr.to_numpy()
    nmrc_spe_filtered = np.concatenate((nmrc_spe_filtered, twoD_nmr), axis=1)
    print('nmrc_spe_filtered:', nmrc_spe_filtered.shape)

    # h-nmr
    print('load 1d h-nmr file...')
    nmrh_max_value = 12.0
    nmrh_min_value = -2.0
    nmrh_spe_filtered = pd.read_csv('./spectromol/t50_drug_database/C0_to_C9_spin_H_NMR.csv')
    peak_columns = [col for col in nmrh_spe_filtered.columns if 'peak' in col]

    print('Filtering H-NMR samples with abnormal values...')
    threshold = 500.0
    nmrh_max_values = nmrh_spe_filtered[peak_columns].max(axis=1)
    h_nmr_abnormal_mask = nmrh_max_values > threshold
    h_nmr_abnormal_indices = set(np.where(h_nmr_abnormal_mask)[0])

    nmrh_spe_filtered[peak_columns] = (nmrh_spe_filtered[peak_columns] - nmrh_min_value) / (nmrh_max_value - nmrh_min_value)
    nmrh_spe_filtered = nmrh_spe_filtered.to_numpy()

    # HSQC
    hsqc_max_value = 400.0
    hsqc_min_value = -350.0
    hsqc = pd.read_csv('./spectromol/t50_drug_database/2D_H-X_HSQC.csv')
    peak_columns = [col for col in hsqc.columns if 'peak' in col]
    hsqc[peak_columns] = (hsqc[peak_columns] - hsqc_min_value) / (hsqc_max_value - hsqc_min_value)
    hsqc = hsqc.to_numpy()

    # COSY
    cosy_max_value = 14.0
    cosy_min_value = -2.0
    nmr_cosy = pd.read_csv('./spectromol/t50_drug_database/C0_to_C9_2D_COSY.csv')
    hxyh_columns = [col for col in nmr_cosy.columns if 'H_X_Y_H' in col]
    nmr_cosy = nmr_cosy[hxyh_columns]
    peak_columns = [col for col in nmr_cosy.columns if 'peak' in col]
    nmr_cosy[peak_columns] = (nmr_cosy[peak_columns] - cosy_min_value) / (cosy_max_value - cosy_min_value)
    nmr_cosy = nmr_cosy.to_numpy()

    # x-nmr
    print('load x-nmr file...')
    # F-NMR
    fnmr_max_value = 0.0001
    fnmr_min_value = -400.0
    nmrf_spe_filtered = pd.read_csv('./spectromol/t50_drug_database/C0_to_C9_F_NMR.csv')
    peak_columns = [col for col in nmrf_spe_filtered.columns if 'peak' in col]
    nmrf_spe_filtered[peak_columns] = (nmrf_spe_filtered[peak_columns] - fnmr_min_value) / (fnmr_max_value - fnmr_min_value)
    nmrf_spe_filtered = nmrf_spe_filtered.to_numpy()

    # N-NMR  
    nnmr_max_value = 400.0
    nnmr_min_value = -260.0
    nmrn_spe_filtered = pd.read_csv('./spectromol/t50_drug_database/C0_to_C9_N_NMR.csv')
    peak_columns = [col for col in nmrn_spe_filtered.columns if 'peak' in col]
    nmrn_spe_filtered[peak_columns] = (nmrn_spe_filtered[peak_columns] - nnmr_min_value) / (nnmr_max_value - nnmr_min_value)
    nmrn_spe_filtered = nmrn_spe_filtered.to_numpy()

    # O-NMR
    onmr_max_value = 460.0
    onmr_min_value = -385.0
    nmro_spe_filtered = pd.read_csv('./spectromol/t50_drug_database/C0_to_C9_O_NMR.csv')
    peak_columns = [col for col in nmro_spe_filtered.columns if 'peak' in col]
    nmro_spe_filtered[peak_columns] = (nmro_spe_filtered[peak_columns] - onmr_min_value) / (onmr_max_value - onmr_min_value)
    nmro_spe_filtered = nmro_spe_filtered.to_numpy()

    # combine all h-nmr and x-nmr features together
    nmrh_spe_filtered = np.concatenate((nmrh_spe_filtered, hsqc, nmr_cosy, nmrf_spe_filtered, nmrn_spe_filtered, nmro_spe_filtered), axis=1)
    print('nmrh_spe_filtered:', nmrh_spe_filtered.shape)

    # ms
    print('load high-mass file...')
    mass = pd.read_csv('./spectromol/t50_drug_database/MS.csv')
    high_mass_spe = mass.to_numpy()
    print('high-mass_spe:', high_mass_spe.shape)

    # atom type
    atom_type = high_mass_spe[:, 1:-1]
    print(f"Atom type shape: {atom_type.shape}")

    # smiles
    smiles_list = pd.read_csv('./spectromol/t50_drug_database/smiles.csv').values.tolist()
    smiles_lengths = [len(smiles[0]) for smiles in smiles_list]
    max_smiles_length = max(smiles_lengths)
    max_seq_length = max_smiles_length + 2
    print(f"SMILES 序列的最大长度为：{max_smiles_length}")
    print(f"模型中应使用的 max_seq_length 为：{max_seq_length}")

    # aux
    auxiliary_data = pd.read_csv('./spectromol/aligned_smiles_id_aux_task.csv').iloc[:, 2:]
    columns = auxiliary_data.columns.tolist()
    auxiliary_tasks = [col for col in columns]
    print(f"Auxiliary tasks: {auxiliary_tasks}")
    print(f"Number of ATs: {len(auxiliary_tasks)}")

    # count_tasks and binary_tasks
    count_tasks = [at for at in auxiliary_tasks if 'Has' not in at and 'Is' not in at]
    binary_tasks = [at for at in auxiliary_tasks if 'Has' in at or 'Is' in at]

    # for debugging, we can limit the dataset size
    test_size = min(100, len(nmrh_spe_filtered))  # use a smaller test size for debugging
    
    test_ir_spe_filtered = ir_spe_filtered[:test_size]
    test_uv_spe_filtered = uv_spe_filtered[:test_size]
    test_nmrh_spe_filtered = nmrh_spe_filtered[:test_size]
    test_nmrc_spe_filtered = nmrc_spe_filtered[:test_size]
    test_high_mass_spe = high_mass_spe[:test_size]
    test_smiles_list = smiles_list[:test_size]
    atom_types_list_test = atom_type[:test_size]

    # test dataset
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
        atom_types_list=atom_types_list_test, 
    )

    # create DataLoader for the test dataset
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,  # smaller batch size for testing
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    # apply the model to the test dataset
    print("Starting temperature sampling inference...")
    results = inference_with_temperature_sampling_analysis(
        model,
        test_dataloader,
        char2idx,
        idx2char,
        max_seq_length=100,
        save_dir='./fangyang/spectromol/csv/corr_draw',
        temperatures=[0.6, 0.8, 1.0, 1.2, 1.5],
        num_candidates=50
    )

    print("Temperature sampling inference completed!")
