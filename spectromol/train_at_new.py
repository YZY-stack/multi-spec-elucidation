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


class SemanticSupervisedSMILESLoss(nn.Module):
    """
    Simplified SMILES loss function, mainly using functional group penalty for semantic supervision
    """
    def __init__(self, ignore_index, functional_penalty_weight=0.05):
        super(SemanticSupervisedSMILESLoss, self).__init__()
        self.smiles_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # Functional group penalty weight - smaller weight to avoid affecting main task stability
        self.functional_penalty_weight = functional_penalty_weight
    
    def forward(self, smiles_output, smiles_target, auxiliary_targets, count_task_outputs, 
                binary_task_outputs, count_tasks, binary_tasks):
        """
        Calculate total loss including SMILES loss and auxiliary task losses.
        
        Args:
            smiles_output: [seq_len * batch_size, vocab_size] - SMILES prediction output
            smiles_target: [seq_len * batch_size] - SMILES ground truth labels
            auxiliary_targets: [batch_size, num_aux_tasks] - Auxiliary task ground truth labels
            count_task_outputs: dict - Count task prediction outputs
            binary_task_outputs: dict - Binary task prediction outputs
            count_tasks: list - List of count task names
            binary_tasks: list - List of binary task names
            
        Returns:
            tuple: (total_loss, smiles_loss, auxiliary_loss)
        """
        # Primary SMILES loss
        smiles_loss = self.smiles_loss_fn(smiles_output, smiles_target)
        
        # Auxiliary task losses
        total_auxiliary_loss = 0.0
        num_aux_tasks = 0
        
        # Count task losses - focus on important chemical features
        count_loss = 0.0
        count_tasks_used = 0
        for idx, task in enumerate(count_tasks):
            if task in self.important_count_tasks:
                target = auxiliary_targets[:, idx].long()
                logits = count_task_outputs[task]
                loss = self.count_task_loss_fn(logits, target)
                count_loss += loss
                count_tasks_used += 1
        
        if count_tasks_used > 0:
            count_loss = count_loss / count_tasks_used  # Average loss
            total_auxiliary_loss += self.count_weight * count_loss
            num_aux_tasks += 1
        
        # Binary classification task losses - structural features
        binary_loss = 0.0
        binary_tasks_used = 0
        for idx, task in enumerate(binary_tasks):
            if task in self.important_binary_tasks:
                target = auxiliary_targets[:, len(count_tasks) + idx].float()
                logit = binary_task_outputs[task]
                loss = self.binary_task_loss_fn(logit, target)
                binary_loss += loss
                binary_tasks_used += 1
        
        if binary_tasks_used > 0:
            binary_loss = binary_loss / binary_tasks_used  # Average loss
            total_auxiliary_loss += self.binary_weight * binary_loss
            num_aux_tasks += 1
        
        # Total loss
        if num_aux_tasks > 0:
            total_auxiliary_loss = total_auxiliary_loss / num_aux_tasks  # Average auxiliary loss
            total_loss = smiles_loss + self.aux_weight * total_auxiliary_loss
        else:
            total_loss = smiles_loss
            
        return total_loss, smiles_loss, total_auxiliary_loss


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




def train_with_semantic_supervision(model, smiles_loss_fn, optimizer, train_dataloader, val_dataloader, 
                                   epochs=10, save_dir='./model_weights_smiles', use_adaptive_weight=True):
    """
    using semantic supervision to train the model with auxiliary tasks.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_bleu_score = 0.0
    
    if isinstance(smiles_loss_fn, SemanticSupervisedSMILESLoss):
        semantic_loss_fn = smiles_loss_fn
    else:
        ignore_index = char2idx['<PAD>']
        semantic_loss_fn = SemanticSupervisedSMILESLoss(ignore_index=ignore_index, aux_weight=0.1)

    for epoch in range(epochs):
        model.train()
        total_loss_sum = 0
        smiles_loss_sum = 0
        aux_loss_sum = 0
        
        if use_adaptive_weight:
            current_aux_weight = get_adaptive_aux_weight(epoch, epochs, initial_weight=0.5, final_weight=0.1)
            semantic_loss_fn.aux_weight = current_aux_weight
        
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
            output_spectra, attention, fusion_feat, count_task_outputs, binary_task_outputs = model(
                ir, uv, c_spec, h_spec, high_mass, tgt_seq, tgt_mask, atom_types)

            # Prepare inputs for semantic loss
            output_flat = output_spectra.reshape(-1, output_spectra.size(-1))
            tgt_output_flat = tgt_output.reshape(-1)

            # compute semantic loss
            total_loss, smiles_loss, aux_loss = semantic_loss_fn(
                output_flat, tgt_output_flat, auxiliary_targets, 
                count_task_outputs, binary_task_outputs, count_tasks, binary_tasks
            )

            total_loss.backward()
            optimizer.step()

            total_loss_sum += total_loss.item()
            smiles_loss_sum += smiles_loss.item()
            aux_loss_sum += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
            
            avg_total_loss = total_loss_sum / (i + 1)
            avg_smiles_loss = smiles_loss_sum / (i + 1)
            avg_aux_loss = aux_loss_sum / (i + 1)
            
            progress_bar.set_postfix({
                'Total': f'{avg_total_loss:.4f}',
                'SMILES': f'{avg_smiles_loss:.4f}',
                'Aux': f'{avg_aux_loss:.4f}',
                'Aux_W': f'{semantic_loss_fn.aux_weight:.3f}'
            })

        # every epoch ends, print training summary
        print(f"\nEpoch [{epoch+1}/{epochs}] Training Summary:")
        print(f"  Total Loss: {avg_total_loss:.4f}")
        print(f"  SMILES Loss: {avg_smiles_loss:.4f}")
        print(f"  Auxiliary Loss: {avg_aux_loss:.4f}")
        print(f"  Auxiliary Weight: {semantic_loss_fn.aux_weight:.3f}")
        print(f"Starting validation...")

        # evaluate on validation set
        val_metrics, val_aux_metrics = evaluate(
            epoch, model, val_dataloader, char2idx, idx2char, max_seq_length=max_seq_length)
        
        # print validation metrics
        for key, value in val_metrics.items():
            print(f"{key}: {value:.4f}")

        # Save best model
        val_bleu_score = val_metrics['BLEU']
        if val_bleu_score > best_bleu_score:
            best_bleu_score = val_bleu_score
            print(f"🎉 New best model with BLEU Score: {best_bleu_score:.4f}")
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_semantic_supervised.pth'))


import torch.nn.functional as F

SMILES_VOCAB = ['<PAD>', '<SOS>', '<EOS>', '<UNK>',
                'C', 'N', 'O', 'F',
                '1', '2', '3', '4', '5',
                '#', '=', '(', ')',
                ]
vocab_size = len(SMILES_VOCAB)

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
nmrc_spe_filtered = pd.read_csv('./spectromol/qm9_all_raw_spe/1d_cnmr_dept.csv')
peak_columns = [col for col in nmrc_spe_filtered.columns if 'peak' in col]
nmrc_spe_filtered[peak_columns] = (nmrc_spe_filtered[peak_columns] - cnmr_min_value) / (cnmr_max_value - cnmr_min_value)
nmrc_spe_filtered = nmrc_spe_filtered.to_numpy()

print('load 2dc-nmr (c-c, c-x) file...')
cnmr_2d_max_value = 450.0
cnmr_2d_min_value = -400.0
twoD_nmr = pd.read_csv('./spectromol/qm9_all_raw_spe/2d_cnmr_ina_chsqc.csv')
peak_columns = [col for col in twoD_nmr.columns if 'peak' in col]
twoD_nmr[peak_columns] = (twoD_nmr[peak_columns] - cnmr_2d_min_value) / (cnmr_2d_max_value - cnmr_2d_min_value)
twoD_nmr = twoD_nmr.to_numpy()
nmrc_spe_filtered = np.concatenate((nmrc_spe_filtered, twoD_nmr), axis=1)
print('nmrc_spe_filtered:', nmrc_spe_filtered.shape)



# h-nmr
print('load 1d h-nmr file...')
nmrh_max_value = 12.0
nmrh_min_value = -2.0
nmrh_spe_filtered = pd.read_csv('./spectromol/qm9_all_raw_spe/1d_hnmr.csv')
peak_columns = [col for col in nmrh_spe_filtered.columns if 'peak' in col]

# filter out samples with abnormal values
print('Filtering H-NMR samples with abnormal values...')
threshold = 500.0
nmrh_max_values = nmrh_spe_filtered[peak_columns].max(axis=1)
h_nmr_abnormal_mask = nmrh_max_values > threshold
h_nmr_abnormal_indices = set(np.where(h_nmr_abnormal_mask)[0])

# using the mask to filter out abnormal samples
nmrh_spe_filtered[peak_columns] = (nmrh_spe_filtered[peak_columns] - nmrh_min_value) / (nmrh_max_value - nmrh_min_value)
nmrh_spe_filtered = nmrh_spe_filtered.to_numpy()


# HSQC
hsqc_max_value = 400.0
hsqc_min_value = -350.0
hsqc = pd.read_csv('./spectromol/qm9_all_raw_spe/2d_hhsqc.csv')
peak_columns = [col for col in hsqc.columns if 'peak' in col]
hsqc[peak_columns] = (hsqc[peak_columns] - hsqc_min_value) / (hsqc_max_value - hsqc_min_value)
hsqc = hsqc.to_numpy()


# COSY
cosy_max_value = 14.0
cosy_min_value = -2.0
nmr_cosy = pd.read_csv('./spectromol/qm9_all_raw_spe/2d_hcosy.csv')
hxyh_columns = [col for col in nmr_cosy.columns if 'H_X_Y_H' in col]
nmr_cosy = nmr_cosy[hxyh_columns]
peak_columns = [col for col in nmr_cosy.columns if 'peak' in col]
nmr_cosy[peak_columns] = (nmr_cosy[peak_columns] - cosy_min_value) / (cosy_max_value - cosy_min_value)
nmr_cosy = nmr_cosy.to_numpy()

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
nmrf_spe_filtered = pd.read_csv('./spectromol/qm9_all_raw_spe/1d_fnmr.csv')
peak_columns = [col for col in nmrf_spe_filtered.columns if 'peak' in col]
nmrf_spe_filtered[peak_columns] = (nmrf_spe_filtered[peak_columns] - fnmr_min_value) / (fnmr_max_value - fnmr_min_value)
nmrf_spe_filtered = nmrf_spe_filtered.to_numpy()

# N-NMR  
nnmr_max_value = 400.0
nnmr_min_value = -260.0
nmrn_spe_filtered = pd.read_csv('./spectromol/qm9_all_raw_spe/1d_nnmr.csv')
peak_columns = [col for col in nmrn_spe_filtered.columns if 'peak' in col]
nmrn_spe_filtered[peak_columns] = (nmrn_spe_filtered[peak_columns] - nnmr_min_value) / (nnmr_max_value - nnmr_min_value)
nmrn_spe_filtered = nmrn_spe_filtered.to_numpy()

# O-NMR
onmr_max_value = 460.0
onmr_min_value = -385.0
nmro_spe_filtered = pd.read_csv('./spectromol/qm9_all_raw_spe/1d_onmr.csv')
peak_columns = [col for col in nmro_spe_filtered.columns if 'peak' in col]
nmro_spe_filtered[peak_columns] = (nmro_spe_filtered[peak_columns] - onmr_min_value) / (onmr_max_value - onmr_min_value)
nmro_spe_filtered = nmro_spe_filtered.to_numpy()

# combine all h-nmr and x-nmr features together
nmrh_spe_filtered = np.concatenate((nmrh_spe_filtered, hsqc, nmr_cosy, nmrf_spe_filtered, nmrn_spe_filtered, nmro_spe_filtered), axis=1)
# nmrh_spe_filtered = np.concatenate((nmrh_spe_filtered, hsqc, nmr_cosy, j2d, nmrf_spe_filtered, nmrn_spe_filtered, nmro_spe_filtered), axis=1)

print('nmrh_spe_filtered:', nmrh_spe_filtered.shape)


# zhipu
print('load high-mass file...')
mass = pd.read_csv('./spectromol/qm9_all_raw_spe/ms.csv')
high_mass_spe = mass.to_numpy()
print('high-mass_spe:', high_mass_spe.shape)


# atom type
atom_type = high_mass_spe[:, 1:-1]
print(f"Atom type shape: {atom_type.shape}")


# smiles
smiles_list = pd.read_csv('./spectromol/qm9_all_raw_spe/smiles.csv').values.tolist() ### [[smiles1], [smiles2], ...]
smiles_lengths = [len(smiles[0]) for smiles in smiles_list]
max_smiles_length = max(smiles_lengths)
max_seq_length = max_smiles_length + 2
print(f"max_seq_length：{max_seq_length}")



# load auxiliary tasks
auxiliary_data = pd.read_csv('./spectromol/aligned_smiles_id_aux_task.csv').iloc[:, 2:]

columns = auxiliary_data.columns.tolist()
auxiliary_tasks = [col for col in columns]
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


# using semantic supervision
ignore_index = char2idx['<PAD>']
semantic_criterion = SemanticSupervisedSMILESLoss(
    ignore_index=ignore_index, 
    aux_weight=0.2,  # auxiliary task weight
    count_weight=1.0,  # weight for count tasks
    binary_weight=1.0  # weight for binary tasks
)

print("🚀 start training")
print(f"📊 count task: {semantic_criterion.important_count_tasks}")
print(f"🔢 binary task: {semantic_criterion.important_binary_tasks}")

# Train the model with semantic supervision
train_with_semantic_supervision(
    model,
    semantic_criterion,
    optimizer,
    train_dataloader,
    val_dataloader,
    epochs=1000,
    save_dir=f'./fangyang/spectromol/csv/weights_{data_split_mode}_semantic',
    use_adaptive_weight=True  # using adaptive weight for auxiliary tasks
)
