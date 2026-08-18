"""
Training script for auxiliary task learning with multi-spectral molecular data.
"""

import os
import random
import math
import csv
from collections import Counter
from math import sqrt

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from Levenshtein import distance as lev

from model import *
from dataset import *

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Set device and data split mode
device = 'cuda'
data_split_mode = 'scaffold'

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
import math


# NOTE: SemanticSupervisedSMILESLoss has been removed.
#
# It was a discontinued experimental variant of the auxiliary-supervision loss
# that was never used to produce any result reported in the manuscript, and it
# could not execute: its __init__ accepted only (ignore_index,
# functional_penalty_weight) while the call site passed aux_weight/
# count_weight/binary_weight, and its forward() referenced attributes that
# __init__ never assigned. The auxiliary-supervision results were obtained
# through train_at() in train_all.py.


def get_adaptive_aux_weight(epoch, total_epochs, initial_weight=0.5, final_weight=0.1, decay_type='exponential'):
    """
    Adaptively adjust auxiliary task weights
    Higher auxiliary task weights in early training help learn semantics; lower weights in later stages focus on SMILES generation
    """
    if decay_type == 'exponential':
        # Exponential decay
        decay_rate = math.log(final_weight / initial_weight) / total_epochs
        weight = initial_weight * math.exp(decay_rate * epoch)
    elif decay_type == 'linear':
        # Linear decay
        weight = initial_weight - (initial_weight - final_weight) * epoch / total_epochs
    elif decay_type == 'cosine':
        # Cosine decay
        weight = final_weight + 0.5 * (initial_weight - final_weight) * (1 + math.cos(math.pi * epoch / total_epochs))
    else:
        weight = initial_weight
    
    return max(weight, final_weight)
from math import sqrt


def cosine_similarity(s1, s2):
    counter1, counter2 = Counter(s1), Counter(s2)
    intersection = set(counter1.keys()) & set(counter2.keys())
    numerator = sum([counter1[x] * counter2[x] for x in intersection])

    sum1 = sum([counter1[x] ** 2 for x in counter1.keys()])
    sum2 = sum([counter2[x] ** 2 for x in counter2.keys()])
    denominator = sqrt(sum1) * sqrt(sum2)

    return numerator / denominator if denominator != 0 else 0.0




def evaluate_at(epoch, model, dataloader, char2idx, idx2char, max_seq_length=100):
    model.eval()
    samples = []
    total_bleu_score = 0.0
    total_cos_score = 0.0
    total_correct = 0
    total_smiles = 0
    bad_sm_count = 0
    n_exact = 0
    maccs_sim, rdk_sim, morgan_sim, levs = [], [], [], []

    # Initialize auxiliary task metrics
    count_task_true = {task: [] for task in count_tasks}
    count_task_pred = {task: [] for task in count_tasks}
    binary_task_true = {task: [] for task in binary_tasks}
    binary_task_pred = {task: [] for task in binary_tasks}

    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoothie = SmoothingFunction().method4

    with torch.no_grad():
        for i, (ir, uv, c_spec, h_spec, high_mass, smiles_indices, auxiliary_targets, atom_types) in enumerate(tqdm(dataloader, desc="Evaluating", ncols=100)):
            # Move data to device
            ir = ir.to(device)
            uv = uv.to(device)
            c_spec = c_spec.to(device)
            h_spec = h_spec.to(device)
            high_mass = high_mass.to(device)
            smiles_indices = smiles_indices.to(device)
            auxiliary_targets = auxiliary_targets.to(device)
            atom_types = atom_types.to(device)
            batch_size = ir.size(0)

            # Get true SMILES
            true_smiles_list = []
            for j in range(batch_size):
                true_indices = smiles_indices[j]
                true_smiles_tokens = []
                for idx in true_indices:
                    idx = idx.item()
                    if idx == char2idx['<EOS>']:
                        break
                    elif idx not in [char2idx['<PAD>'], char2idx['<SOS>']]:
                        true_smiles_tokens.append(idx2char.get(idx, '<UNK>'))
                true_smiles_str = ''.join(true_smiles_tokens)
                true_smiles_list.append(true_smiles_str)


            # Greedy prediction
            predicted_smiles_list = predict_greedy(
                model, ir, uv, c_spec, h_spec, high_mass,
                char2idx, idx2char, max_seq_length=max_seq_length, atom_types=atom_types)



            for true_smiles_str, predicted_smiles_str in zip(true_smiles_list, predicted_smiles_list):
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

                # Compute BLEU score
                reference = [list(true_smiles_str)]
                candidate = list(predicted_smiles_str)
                bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
                total_bleu_score += bleu_score

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

                # Compute Cos-sim
                cos_sim = cosine_similarity(predicted_smiles_str, true_smiles_str)
                total_cos_score += cos_sim

                # Compute L-Distance
                l_dis = lev(predicted_smiles_str, true_smiles_str)
                levs.append(l_dis)


            # Get auxiliary task predictions
            # Prepare dummy target sequence (only for getting auxiliary outputs)
            tgt_seq = torch.full((1, batch_size), char2idx['<SOS>'], dtype=torch.long, device=device)
            tgt_mask = model.smiles_decoder.generate_square_subsequent_mask(1).to(device)


            _, _, _, count_task_outputs, binary_task_outputs = model(
                ir, uv, c_spec, h_spec, high_mass, tgt_seq, tgt_mask, atom_types=atom_types)

            # Collect auxiliary task predictions and true values
            # Count tasks
            for idx_task, task in enumerate(count_tasks):
                target = auxiliary_targets[:, idx_task].cpu().numpy()  # Shape: [batch_size]
                logits = count_task_outputs[task]  # Shape: [batch_size, num_classes]
                predictions = logits.argmax(dim=1).cpu().numpy()  # Shape: [batch_size]
                count_task_true[task].extend(target)
                count_task_pred[task].extend(predictions)

            # Binary tasks
            for idx_task, task in enumerate(binary_tasks):
                target = auxiliary_targets[:, len(count_tasks) + idx_task].cpu().numpy()
                logit = binary_task_outputs[task]
                predictions = (torch.sigmoid(logit) > 0.5).int().cpu().numpy()
                binary_task_true[task].extend(target)
                binary_task_pred[task].extend(predictions)



        avg_bleu_score = total_bleu_score / total_smiles
        accuracy = total_correct / total_smiles
        # Compute validity
        validity  = (total_smiles - bad_sm_count) / total_smiles
        cos_sim_all = total_cos_score / total_smiles
        exact = n_exact * 1.0 / total_smiles


        results_dict = {
            'BLEU': avg_bleu_score,
            'validity': validity,
            'Levenshtein': np.mean(levs),
            'Cosine Similarity': cos_sim_all,
            'Top1 Acc': accuracy,
            'Exact': exact,
            "MACCS FTS": np.mean(maccs_sim),
            "RDKit FTS": np.mean(rdk_sim),
            "Morgan FTS": np.mean(morgan_sim),
        }

        # Compute auxiliary task metrics
        from sklearn.metrics import accuracy_score, mean_absolute_error

        count_task_acc = {}
        count_task_mae = {}
        for task in count_tasks:
            y_true = count_task_true[task]
            y_pred = count_task_pred[task]
            acc = accuracy_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            count_task_acc[task] = acc
            count_task_mae[task] = mae

        binary_task_acc = {}
        for task in binary_tasks:
            y_true = binary_task_true[task]
            y_pred = binary_task_pred[task]
            acc = accuracy_score(y_true, y_pred)
            binary_task_acc[task] = acc

        # # Display sample results
        # for i, sample in enumerate(samples):
        #     print(f"\nSample {i+1}:")
        #     print(f"True SMILES: {sample['True SMILES']}")
        #     print(f"Predicted SMILES: {sample['Predicted SMILES']}")
        #     print(f"BLEU Score: {sample['BLEU Score']:.4f}")
        #     print("Auxiliary Task Predictions:")
        #     for task in count_tasks + binary_tasks:
        #         true_value = sample['Auxiliary True'][task]
        #         predicted_value = sample['Auxiliary Predicted'][task]
        #         print(f"  {task}: True = {true_value}, Predicted = {predicted_value}")
        #     print("-" * 50)

        # Return metrics
        val_aux_metrics = {
            'count_task_acc': 0,
            'count_task_mae': 0,
            'binary_task_acc': 0,
        }


        # if epoch % 10 == 0:
        #     os.makedirs(f"./results_new/", exist_ok=True)
        #     with open(f"results_new/epoch_{epoch}_results.csv", 'w', newline='') as file:
        #         writer = csv.DictWriter(file, fieldnames=['pred_smiles', 'true_smiles', 'BLEU'])
        #         writer.writeheader()
        #         writer.writerows(results)

        return results_dict, val_aux_metrics
        # return avg_bleu_score, accuracy, val_aux_metrics



def evaluate(epoch, model, dataloader, char2idx, idx2char, max_seq_length=100):
    model.eval()
    samples = []
    total_bleu_score = 0.0
    total_cos_score = 0.0
    total_correct = 0
    total_smiles = 0
    bad_sm_count = 0
    n_exact = 0
    maccs_sim, rdk_sim, morgan_sim, levs = [], [], [], []

    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoothie = SmoothingFunction().method4

    with torch.no_grad():
        for i, (ir, uv, c_spec, h_spec, high_mass, smiles_indices, auxiliary_targets, atom_types) in enumerate(tqdm(dataloader, desc="Evaluating", ncols=100)):
            # Move data to device
            ir = ir.to(device)
            uv = uv.to(device)
            c_spec = c_spec.to(device)
            h_spec = h_spec.to(device)
            high_mass = high_mass.to(device)
            smiles_indices = smiles_indices.to(device)
            auxiliary_targets = auxiliary_targets.to(device)
            atom_types = atom_types.to(device)
            batch_size = ir.size(0)

            # Get true SMILES
            true_smiles_list = []
            for j in range(batch_size):
                true_indices = smiles_indices[j]
                true_smiles_tokens = []
                for idx in true_indices:
                    idx = idx.item()
                    if idx == char2idx['<EOS>']:
                        break
                    elif idx not in [char2idx['<PAD>'], char2idx['<SOS>']]:
                        true_smiles_tokens.append(idx2char.get(idx, '<UNK>'))
                true_smiles_str = ''.join(true_smiles_tokens)
                true_smiles_list.append(true_smiles_str)


            # Greedy prediction for training (faster and efficient)
            predicted_smiles_list = predict_greedy(
                model, ir, uv, c_spec, h_spec, high_mass,
                char2idx, idx2char, max_seq_length=max_seq_length, atom_types=atom_types)


            for true_smiles_str, predicted_smiles_str in zip(true_smiles_list, predicted_smiles_list):
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

                # Compute BLEU score
                reference = [list(true_smiles_str)]
                candidate = list(predicted_smiles_str)
                bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
                total_bleu_score += bleu_score

                # Compute SMILES accuracy
                if predicted_smiles_str == true_smiles_str:
                    total_correct += 1
                total_smiles += 1

                # Compute Cos-sim
                cos_sim = cosine_similarity(predicted_smiles_str, true_smiles_str)
                total_cos_score += cos_sim

                # Compute L-Distance
                l_dis = lev(predicted_smiles_str, true_smiles_str)
                levs.append(l_dis)



        avg_bleu_score = total_bleu_score / total_smiles
        accuracy = total_correct / total_smiles
        # Compute validity
        validity  = (total_smiles - bad_sm_count) / total_smiles
        cos_sim_all = total_cos_score / total_smiles
        exact = n_exact * 1.0 / total_smiles


        results_dict = {
            'BLEU': avg_bleu_score,
            'validity': validity,
            'Levenshtein': np.mean(levs),
            'Cosine Similarity': cos_sim_all,
            'Top1 Acc': accuracy,
            'Exact': exact,
            "MACCS FTS": np.mean(maccs_sim),
            "RDKit FTS": np.mean(rdk_sim),
            "Morgan FTS": np.mean(morgan_sim),
        }

        # if epoch % 10 == 0:
        #     os.makedirs(f"./results_new/", exist_ok=True)
        #     with open(f"results_new/epoch_{epoch}_results.csv", 'w', newline='') as file:
        #         writer = csv.DictWriter(file, fieldnames=['pred_smiles', 'true_smiles', 'BLEU'])
        #         writer.writeheader()
        #         writer.writerows(results)

        return results_dict, None



import torch.nn.functional as F
def predict_greedy(model, ir, uv, c_spec, h_spectrum, high_mass, char2idx, idx2char, max_seq_length=100, atom_types=None):
    model.eval()
    with torch.no_grad():
        # Split h_spectrum into h_spectrum_part, f_spectrum, n_spectrum
        h_spectrum_part = h_spectrum[:, :382]
        f_spectrum = h_spectrum[:, 382:394]
        n_spectrum = h_spectrum[:, 394:408]
        o_spectrum = h_spectrum[:, 408:]

        # Prepare features
        features = {
            'ir': ir,
            'uv': uv,
            'nmr_c': c_spec,
            'nmr_h': h_spectrum_part,
            'f_spectrum': f_spectrum,
            'n_spectrum': n_spectrum,
            'o_spectrum': o_spectrum,
            'mass_high': high_mass
        }

        # Tokenize features
        tokens = model.tokenizer(features)  # Shape: [batch_size, total_N_features, d_model]

        # Permute for transformer input: [seq_len, batch_size, d_model]
        tokens = tokens.permute(1, 0, 2)

        # Apply transformer encoder
        memory, attention = model.transformer_encoder(tokens)  # Shape: [seq_len, batch_size, d_model]

        batch_size = ir.size(0)
        device = ir.device

        # Initialize input sequence with <SOS> tokens
        tgt_indices = torch.full((1, batch_size), char2idx['<SOS>'], dtype=torch.long, device=device)  # Shape: [1, batch_size]

        generated_tokens = []

        for _ in range(max_seq_length):
            # Generate target mask
            tgt_mask = model.smiles_decoder.generate_square_subsequent_mask(tgt_indices.size(0)).to(device)

            # Decode using the SMILES decoder
            output = model.smiles_decoder(
                tgt_indices,
                memory,
                tgt_mask=tgt_mask,
                atom_types=atom_types
            )  # Shape: [tgt_seq_len, batch_size, vocab_size]

            # Get the last timestep output
            output_logits = output[-1, :, :]  # Shape: [batch_size, vocab_size]

            # Greedy decoding: select the token with highest probability
            next_token = output_logits.argmax(dim=-1)  # Shape: [batch_size]

            generated_tokens.append(next_token.unsqueeze(0))  # Shape: [1, batch_size]

            # Append the next token to the target indices
            tgt_indices = torch.cat([tgt_indices, next_token.unsqueeze(0)], dim=0)  # Shape: [tgt_seq_len + 1, batch_size]

            # Stop if all sequences have generated <EOS>
            if (next_token == char2idx['<EOS>']).all():
                break

        # Convert generated token indices to SMILES strings
        generated_tokens = torch.cat(generated_tokens, dim=0)  # Shape: [generated_seq_len, batch_size]
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





# Define weight adjustment function
def get_smiles_weight(epoch, total_epochs, k=5):
    return 1 - math.exp(-k * epoch / total_epochs)




def train_at(model, smiles_loss_fn, optimizer, train_dataloader, val_dataloader, epochs=10, save_dir='./model_weights_smiles'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_bleu_score = 0.0  # For saving the best model
    feature_loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", total=len(train_dataloader), ncols=100)
        for i, (ir, uv, c_spec, h_spec, high_mass, smiles_indices, auxiliary_targets, atom_types) in enumerate(progress_bar):
            # Move data to device
            ir = ir.to(device)
            uv = uv.to(device)
            c_spec = c_spec.to(device)
            h_spec = h_spec.to(device)
            high_mass = high_mass.to(device)
            smiles_indices = smiles_indices.to(device)
            auxiliary_targets = auxiliary_targets.to(device)
            atom_types = atom_types.to(device)

            optimizer.zero_grad()

            # Prepare target sequence
            tgt_seq = smiles_indices.transpose(0, 1)[:-1]
            tgt_output = smiles_indices.transpose(0, 1)[1:]

            # Generate mask
            seq_len = tgt_seq.size(0)
            tgt_mask = model.smiles_decoder.generate_square_subsequent_mask(seq_len).to(device)

            # Forward pass
            output_spectra, output_mol, spectra_feat, count_task_outputs, binary_task_outputs = model(
                ir, uv, c_spec, h_spec, high_mass, tgt_seq, tgt_mask, atom_types)

            # Compute main task loss (from spectral features)
            output_flat = output_spectra.reshape(-1, output_spectra.size(-1))
            tgt_output_flat = tgt_output.reshape(-1)
            loss_smiles_spectra = smiles_loss_fn(output_flat, tgt_output_flat)

            # # Compute loss from molecular features
            # output_mol_flat = output_mol.reshape(-1, output_mol.size(-1))
            # loss_smiles_mol = smiles_loss_fn(output_mol_flat, tgt_output_flat)

            # # Compute feature loss between spectra_feat and mol_feat
            # feature_loss = feature_loss_fn(spectra_feat, mol_feat)

            # # Total loss
            # total_loss = loss_smiles_spectra

            # Calculate auxiliary task loss
            total_auxiliary_loss = 0.0

            # Count tasks
            for idx, task in enumerate(count_tasks):
                target = auxiliary_targets[:, idx].long()
                logits = count_task_outputs[task]
                loss = count_task_loss_fn(logits, target)
                total_auxiliary_loss += loss

            # 二元分类任务
            for idx, task in enumerate(binary_tasks):
                target = auxiliary_targets[:, len(count_tasks) + idx].float()
                logit = binary_task_outputs[task]
                loss = binary_task_loss_fn(logit, target)
                total_auxiliary_loss += loss

            # 总损失
            total_loss = loss_smiles_spectra + 0.1*total_auxiliary_loss
            # total_loss = total_auxiliary_loss
            # if epoch < 50:
            #     total_loss = total_auxiliary_loss
            # else:
            #     total_loss = loss_smiles_spectra
            # # 动态调整总损失
            # w_smiles = get_smiles_weight(epoch, 200, k=5)
            # total_loss = w_smiles * loss_smiles_spectra + (1 - w_smiles) * total_auxiliary_loss

            # 反向传播和优化
            total_loss.backward()
            optimizer.step()

            # 日志记录
            avg_loss = total_loss.item() / (i + 1)
            progress_bar.set_postfix({'Loss': avg_loss})

        # 每个 epoch 结束后在验证集上评估
        print(f"\nEpoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}")
        # print(f"\nEpoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}, Weight for the main task: {w_smiles:.4f}")
        print(f"Epoch: {epoch+1}, starting validation...")

        # 在验证集上评估
        val_metrics, val_aux_metrics = evaluate(
            epoch, model, val_dataloader, char2idx, idx2char, max_seq_length=max_seq_length)
        # val_bleu_score, val_acc = evaluate(
        #     model, val_dataloader, char2idx, idx2char, max_seq_length=max_seq_length)
        for key, value in val_metrics.items():
            print(f"{key}: {value:.4f}")

        # 打印辅助任务指标
        print("\nValidation Auxiliary Task Metrics:")
        for task in count_tasks:
            acc = val_aux_metrics['count_task_acc'][task]
            mae = val_aux_metrics['count_task_mae'][task]
            if acc < 0.9:
                print(f"Count Task - {task}: Accuracy = {acc:.4f}, MAE = {mae:.4f}")
        
        for task in binary_tasks:
            acc = val_aux_metrics['binary_task_acc'][task]
            if acc < 0.9:
                print(f"Binary Task - {task}: Accuracy = {acc:.4f}")

        # 保存最佳模型
        if val_metrics['BLEU'] > best_bleu_score:
            best_bleu_score = val_metrics['BLEU']
            torch.save(model.state_dict(), os.path.join(save_dir, 'ir_only_scaffold.pth'))
            print(f"Best model saved with BLEU Score: {best_bleu_score:.4f}")




# train_with_semantic_supervision() has been removed together with
# SemanticSupervisedSMILESLoss, which it depended on. The auxiliary-
# supervision training path is train_at() in train_all.py.


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


    

# split dataset
train_df = pd.read_csv(f'./spectromol/csv/dataset/{data_split_mode}/train.csv')
val_df = pd.read_csv(f'./spectromol/csv/dataset/{data_split_mode}/val.csv')
test_df = pd.read_csv(f'./spectromol/csv/dataset/{data_split_mode}/test.csv')

# process abnormal samples
if len(all_abnormal_indices) > 0:
    print(f"Processing {len(all_abnormal_indices)} abnormal samples...")
    
    abnormal_smiles = set()
    for idx in all_abnormal_indices:
        if idx < len(smiles_list):
            abnormal_smiles.add(smiles_list[idx][0])
    
    print(f"Found {len(abnormal_smiles)} unique abnormal SMILES")
    
    # reset indices for train, val, test sets
    original_train_size = len(train_df)
    original_val_size = len(val_df)
    original_test_size = len(test_df)
    
    train_df = train_df[~train_df['smiles'].isin(abnormal_smiles)].reset_index(drop=True)
    val_df = val_df[~val_df['smiles'].isin(abnormal_smiles)].reset_index(drop=True)
    test_df = test_df[~test_df['smiles'].isin(abnormal_smiles)].reset_index(drop=True)
    
    print(f"Train set: {original_train_size} -> {len(train_df)} (removed {original_train_size - len(train_df)})")
    print(f"Val set: {original_val_size} -> {len(val_df)} (removed {original_val_size - len(val_df)})")
    print(f"Test set: {original_test_size} -> {len(test_df)} (removed {original_test_size - len(test_df)})")
    
    total_samples = len(smiles_list)
    normal_mask = np.ones(total_samples, dtype=bool)
    normal_mask[all_abnormal_indices] = False
    
    ir_spe_filtered = ir_spe_filtered[normal_mask]
    uv_spe_filtered = uv_spe_filtered[normal_mask]
    nmrh_spe_filtered = nmrh_spe_filtered[normal_mask]
    nmrc_spe_filtered = nmrc_spe_filtered[normal_mask]
    high_mass_spe = high_mass_spe[normal_mask]
    atom_type = atom_type[normal_mask]
    
    original_smiles_list = smiles_list.copy()
    smiles_list = [original_smiles_list[i] for i in range(total_samples) if normal_mask[i]]
    auxiliary_data = auxiliary_data[normal_mask].reset_index(drop=True)
    
    print(f"Filtered dataset: {total_samples} -> {len(smiles_list)} samples")

print(f"Final dataset size: {len(smiles_list)}")
assert len(smiles_list) == len(auxiliary_data) == ir_spe_filtered.shape[0], "Data length mismatch after filtering!"

# create a mapping from SMILES to index
smiles_to_index = {smiles[0]: idx for idx, smiles in enumerate(smiles_list)}

train_indices, train_missing_smiles = get_indices(train_df['smiles'], smiles_to_index)
val_indices, val_missing_smiles = get_indices(val_df['smiles'], smiles_to_index)
test_indices, test_missing_smiles = get_indices(test_df['smiles'], smiles_to_index)


# train
train_ir_spe_filtered = ir_spe_filtered[train_indices]
train_uv_spe_filtered = uv_spe_filtered[train_indices]
train_nmrh_spe_filtered = nmrh_spe_filtered[train_indices]
train_nmrc_spe_filtered = nmrc_spe_filtered[train_indices]
train_high_mass_spe = high_mass_spe[train_indices]
train_smiles_list = [smiles_list[idx] for idx in train_indices]
train_aux_data = auxiliary_data.iloc[train_indices].reset_index(drop=True)
atom_types_list_train = atom_type[train_indices]



# val
val_ir_spe_filtered = ir_spe_filtered[val_indices]
val_uv_spe_filtered = uv_spe_filtered[val_indices]
val_nmrh_spe_filtered = nmrh_spe_filtered[val_indices]
val_nmrc_spe_filtered = nmrc_spe_filtered[val_indices]
val_high_mass_spe = high_mass_spe[val_indices]
val_smiles_list = [smiles_list[idx] for idx in val_indices]
val_aux_data = auxiliary_data.iloc[val_indices].reset_index(drop=True)
atom_types_list_val = atom_type[val_indices]

# test
test_ir_spe_filtered = ir_spe_filtered[test_indices]
test_uv_spe_filtered = uv_spe_filtered[test_indices]
test_nmrh_spe_filtered = nmrh_spe_filtered[test_indices]
test_nmrc_spe_filtered = nmrc_spe_filtered[test_indices]
test_high_mass_spe = high_mass_spe[test_indices]
test_smiles_list = [smiles_list[idx] for idx in test_indices]
test_aux_data = auxiliary_data.iloc[test_indices].reset_index(drop=True)
atom_types_list_test = atom_type[test_indices]



# count_tasks and binary_tasks
count_tasks = [at for at in auxiliary_tasks if 'Has' not in at and 'Is' not in at]
binary_tasks = [at for at in auxiliary_tasks if 'Has' in at or 'Is' in at]


# train
train_dataset = SpectraDataset(
    ir_spectra=train_ir_spe_filtered,
    uv_spectra=train_uv_spe_filtered,
    c_spectra=train_nmrc_spe_filtered,
    h_spectra=train_nmrh_spe_filtered,
    high_mass_spectra=train_high_mass_spe,
    smiles_list=train_smiles_list,
    auxiliary_data=train_aux_data,
    char2idx=char2idx,
    max_seq_length=max_seq_length,
    count_tasks=count_tasks,
    binary_tasks=binary_tasks,
    atom_types_list=atom_types_list_train, 
)

# val
val_dataset = SpectraDataset(
    ir_spectra=val_ir_spe_filtered,
    uv_spectra=val_uv_spe_filtered,
    c_spectra=val_nmrc_spe_filtered,
    h_spectra=val_nmrh_spe_filtered,
    high_mass_spectra=val_high_mass_spe,
    smiles_list=val_smiles_list,
    auxiliary_data=val_aux_data,
    char2idx=char2idx,
    max_seq_length=max_seq_length,
    count_tasks=count_tasks,
    binary_tasks=binary_tasks,
    atom_types_list=atom_types_list_val, 
)

# test
test_dataset = SpectraDataset(
    ir_spectra=test_ir_spe_filtered,
    uv_spectra=test_uv_spe_filtered,
    c_spectra=test_nmrc_spe_filtered,
    h_spectra=test_nmrh_spe_filtered,
    high_mass_spectra=test_high_mass_spe,
    smiles_list=test_smiles_list,
    auxiliary_data=test_aux_data,
    char2idx=char2idx,
    max_seq_length=max_seq_length,
    count_tasks=count_tasks,
    binary_tasks=binary_tasks,
    atom_types_list=atom_types_list_val, 
)


from torch.utils.data import DataLoader

# train loader
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# val loader
val_dataloader = DataLoader(
    val_dataset,
    batch_size=128, 
    shuffle=False,
    num_workers=4,
    drop_last=True,
)

# test loader
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)





# compute count_task_classes
count_task_classes = {}
for task in count_tasks:
    max_value = int(auxiliary_data[task].max())
    count_task_classes[task] = max_value + 1  # number of classes is max value + 1



def load_model(model_path, vocab_size, char2idx):
    model = AtomPredictionModel(vocab_size=vocab_size, count_tasks_classes=None, binary_tasks=None)
    model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


# def init_weights(m):
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
#         nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)
#     elif isinstance(m, nn.Embedding):
#         nn.init.uniform_(m.weight, -0.1, 0.1)
# model.apply(init_weights)

 # load the model from the trained weights from the previous training
model_path = './fangyang/spectromol/csv/weights_scaffold_at/0806_ft.pth'

# load
model = load_model(model_path, vocab_size, char2idx)

# criterion = ContrastiveLoss()
# criterion = AtomPredictionLoss()
ignore_index = char2idx['<PAD>']
criterion = SMILESLoss(ignore_index)

count_task_loss_fn = nn.CrossEntropyLoss()
binary_task_loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# The semantic-supervision training entry point has been removed together with
# SemanticSupervisedSMILESLoss. Use train_all.py for both the
# auxiliary-supervision configuration (train_at) and the SMILES-only ablation
# (train); see README.

