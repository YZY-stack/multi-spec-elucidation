import os
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import *
from dataset import *
from sklearn.preprocessing import StandardScaler
import csv
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from Levenshtein import distance as lev
RDLogger.DisableLog('rdApp.*')

# import pickle
# path = '/root/workspace/smiles-transformer-master/csv/output_smiles_vectors.pkl'
# with open(path, 'rb') as f: 
#     atom_coor_smiles = pickle.load(f)
#     import pdb;pdb.set_trace()
#     print('a')


# from sm_trans.smiles_transformer.build_vocab import WordVocab
# from sm_trans.smiles_transformer.pretrain_trfm import TrfmSeq2seq

# vocab = WordVocab.load_vocab('/root/workspace/smiles-transformer-master/csv/vocab.pkl')
# trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
# trfm.load_state_dict(torch.load('/root/workspace/smiles-transformer-master/csv/trfm_12_23000.pkl'))
# trfm.eval()

device = 'cuda'

data_split_mode = 'scaffold'


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)



from sklearn.metrics import accuracy_score, mean_absolute_error
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch.utils.data as data


from collections import Counter
import math


class SemanticSupervisedSMILESLoss(nn.Module):
    """
    简化的SMILES损失函数，主要使用官能团惩罚进行语义监督
    """
    def __init__(self, ignore_index, functional_penalty_weight=0.05):
        super(SemanticSupervisedSMILESLoss, self).__init__()
        self.smiles_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # 官能团惩罚权重 - 较小的权重避免影响主任务稳定性
        self.functional_penalty_weight = functional_penalty_weight
    
    def forward(self, smiles_output, smiles_target, auxiliary_targets, count_task_outputs, 
                binary_task_outputs, count_tasks, binary_tasks):
        """
        计算总损失，包括SMILES损失和辅助任务损失
        
        Args:
            smiles_output: [seq_len * batch_size, vocab_size] - SMILES预测输出
            smiles_target: [seq_len * batch_size] - SMILES真实标签
            auxiliary_targets: [batch_size, num_aux_tasks] - 辅助任务真实标签
            count_task_outputs: dict - 计数任务预测输出
            binary_task_outputs: dict - 二元任务预测输出
            count_tasks: list - 计数任务名称列表
            binary_tasks: list - 二元任务名称列表
        """
        # 主要SMILES损失
        smiles_loss = self.smiles_loss_fn(smiles_output, smiles_target)
        
        # 辅助任务损失
        total_auxiliary_loss = 0.0
        num_aux_tasks = 0
        
        # 计数任务损失 - 重点关注重要的化学特征
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
            count_loss = count_loss / count_tasks_used  # 平均损失
            total_auxiliary_loss += self.count_weight * count_loss
            num_aux_tasks += 1
        
        # 二元分类任务损失 - 结构特征
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
            binary_loss = binary_loss / binary_tasks_used  # 平均损失
            total_auxiliary_loss += self.binary_weight * binary_loss
            num_aux_tasks += 1
        
        # 总损失
        if num_aux_tasks > 0:
            total_auxiliary_loss = total_auxiliary_loss / num_aux_tasks  # 平均辅助损失
            total_loss = smiles_loss + self.aux_weight * total_auxiliary_loss
        else:
            total_loss = smiles_loss
            
        return total_loss, smiles_loss, total_auxiliary_loss


def get_adaptive_aux_weight(epoch, total_epochs, initial_weight=0.5, final_weight=0.1, decay_type='exponential'):
    """
    自适应调整辅助任务权重
    早期训练时辅助任务权重较大，帮助学习语义；后期权重减小，专注SMILES生成
    """
    if decay_type == 'exponential':
        # 指数衰减
        decay_rate = math.log(final_weight / initial_weight) / total_epochs
        weight = initial_weight * math.exp(decay_rate * epoch)
    elif decay_type == 'linear':
        # 线性衰减
        weight = initial_weight - (initial_weight - final_weight) * epoch / total_epochs
    elif decay_type == 'cosine':
        # 余弦衰减
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





# 定义权重调节函数
def get_smiles_weight(epoch, total_epochs, k=5):
    """
    计算 SMILES 损失的权重，基于指数增长。
    Args:
        epoch (int): 当前的训练轮次
        total_epochs (int): 总训练轮次
        k (float): 控制增长速率的参数
    Returns:
        float: SMILES 损失的权重
    """
    return 1 - math.exp(-k * epoch / total_epochs)




def train_at(model, smiles_loss_fn, optimizer, train_dataloader, val_dataloader, epochs=10, save_dir='./model_weights_smiles'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_bleu_score = 0.0  # 用于保存最佳模型
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

            # 计算辅助任务损失
            total_auxiliary_loss = 0.0

            # 计数任务
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
    使用语义监督的训练函数，包含辅助任务损失
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_bleu_score = 0.0
    
    # 使用语义监督损失函数
    if isinstance(smiles_loss_fn, SemanticSupervisedSMILESLoss):
        semantic_loss_fn = smiles_loss_fn
    else:
        # 如果传入的是普通损失函数，创建语义监督版本
        ignore_index = char2idx['<PAD>']
        semantic_loss_fn = SemanticSupervisedSMILESLoss(ignore_index=ignore_index, aux_weight=0.1)

    for epoch in range(epochs):
        model.train()
        total_loss_sum = 0
        smiles_loss_sum = 0
        aux_loss_sum = 0
        
        # 自适应调整辅助任务权重
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

            # 计算语义监督损失
            total_loss, smiles_loss, aux_loss = semantic_loss_fn(
                output_flat, tgt_output_flat, auxiliary_targets, 
                count_task_outputs, binary_task_outputs, count_tasks, binary_tasks
            )

            # 反向传播和优化
            total_loss.backward()
            optimizer.step()

            # 累积损失用于日志
            total_loss_sum += total_loss.item()
            smiles_loss_sum += smiles_loss.item()
            aux_loss_sum += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
            
            # 更新进度条
            avg_total_loss = total_loss_sum / (i + 1)
            avg_smiles_loss = smiles_loss_sum / (i + 1)
            avg_aux_loss = aux_loss_sum / (i + 1)
            
            progress_bar.set_postfix({
                'Total': f'{avg_total_loss:.4f}',
                'SMILES': f'{avg_smiles_loss:.4f}',
                'Aux': f'{avg_aux_loss:.4f}',
                'Aux_W': f'{semantic_loss_fn.aux_weight:.3f}'
            })

        # 每个 epoch 结束后在验证集上评估
        print(f"\nEpoch [{epoch+1}/{epochs}] Training Summary:")
        print(f"  Total Loss: {avg_total_loss:.4f}")
        print(f"  SMILES Loss: {avg_smiles_loss:.4f}")
        print(f"  Auxiliary Loss: {avg_aux_loss:.4f}")
        print(f"  Auxiliary Weight: {semantic_loss_fn.aux_weight:.3f}")
        print(f"Starting validation...")

        # 在验证集上评估
        val_metrics, val_aux_metrics = evaluate(
            epoch, model, val_dataloader, char2idx, idx2char, max_seq_length=max_seq_length)
        
        # 打印主要指标
        for key, value in val_metrics.items():
            print(f"{key}: {value:.4f}")

        # Save best model
        val_bleu_score = val_metrics['BLEU']
        if val_bleu_score > best_bleu_score:
            best_bleu_score = val_bleu_score
            print(f"🎉 New best model with BLEU Score: {best_bleu_score:.4f}")
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_semantic_supervised.pth'))


def train(model, smiles_loss_fn, optimizer, train_dataloader, val_dataloader, epochs=10, save_dir='./model_weights_smiles'):
    """
    保持原有的训练函数作为向后兼容
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_bleu_score = 0.0

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
            output_spectra, attention, fusion_feat, count_task_outputs, binary_task_outputs = model(
                ir, uv, c_spec, h_spec, high_mass, tgt_seq, tgt_mask, atom_types)

            # Compute main task loss (from spectral features)
            output_flat = output_spectra.reshape(-1, output_spectra.size(-1))
            tgt_output_flat = tgt_output.reshape(-1)
            loss_smiles_spectra = smiles_loss_fn(output_flat, tgt_output_flat)

            # 总损失
            total_loss = loss_smiles_spectra

            # 反向传播和优化
            total_loss.backward()
            optimizer.step()

            # 日志记录
            avg_loss = total_loss.item() / (i + 1)
            progress_bar.set_postfix({'Loss': avg_loss})

        # 每个 epoch 结束后在验证集上评估
        print(f"\nEpoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}")
        print(f"Epoch: {epoch+1}, starting validation...")

        # 在验证集上评估
        val_metrics, val_aux_metrics = evaluate(
            epoch, model, val_dataloader, char2idx, idx2char, max_seq_length=max_seq_length)
        for key, value in val_metrics.items():
            print(f"{key}: {value:.4f}")

        # Save best model
        val_bleu_score = val_metrics['BLEU']
        if val_bleu_score > best_bleu_score:
            best_bleu_score = val_bleu_score
            print(f"Current best model with BLEU Score: {best_bleu_score:.4f}")
            torch.save(model.state_dict(), os.path.join(save_dir, 'tmp.pth'))





import torch.nn.functional as F

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
uv_spe_filtered = pd.read_csv('/data4/linkaiqing/sm_pretrained/gp/qm9_all_raw_spe/uv.csv')
peak_columns = [col for col in uv_spe_filtered.columns if 'peak' in col]
uv_spe_filtered[peak_columns] = uv_spe_filtered[peak_columns] / uv_max_value
uv_spe_filtered = uv_spe_filtered.to_numpy()
print('uv_spe_filtered:', uv_spe_filtered.shape)


# ir
print('load ir file...')
ir_max_value = 4000.0
ir_spe_filtered = pd.read_csv('/data4/linkaiqing/sm_pretrained/gp/qm9_all_raw_spe/ir_82.csv')
peak_columns = [col for col in ir_spe_filtered.columns if 'peak' in col]
ir_spe_filtered[peak_columns] = ir_spe_filtered[peak_columns] / ir_max_value
ir_spe_filtered = ir_spe_filtered.to_numpy()
print('ir_spe_filtered:', ir_spe_filtered.shape)


# c-nmr
print('load 1dc-nmr with dept file...')
cnmr_max_value = 220.0
cnmr_min_value = -10.0
nmrc_spe_filtered = pd.read_csv('/data4/linkaiqing/sm_pretrained/gp/qm9_all_raw_spe/1d_cnmr_dept.csv')
peak_columns = [col for col in nmrc_spe_filtered.columns if 'peak' in col]
nmrc_spe_filtered[peak_columns] = (nmrc_spe_filtered[peak_columns] - cnmr_min_value) / (cnmr_max_value - cnmr_min_value)
nmrc_spe_filtered = nmrc_spe_filtered.to_numpy()

print('load 2dc-nmr (c-c, c-x) file...')
cnmr_2d_max_value = 450.0
cnmr_2d_min_value = -400.0
twoD_nmr = pd.read_csv('/data4/linkaiqing/sm_pretrained/gp/qm9_all_raw_spe/2d_cnmr_ina_chsqc.csv')
peak_columns = [col for col in twoD_nmr.columns if 'peak' in col]
twoD_nmr[peak_columns] = (twoD_nmr[peak_columns] - cnmr_2d_min_value) / (cnmr_2d_max_value - cnmr_2d_min_value)
twoD_nmr = twoD_nmr.to_numpy()
nmrc_spe_filtered = np.concatenate((nmrc_spe_filtered, twoD_nmr), axis=1)
print('nmrc_spe_filtered:', nmrc_spe_filtered.shape)



# h-nmr
print('load 1d h-nmr file...')
nmrh_max_value = 12.0
nmrh_min_value = -2.0
nmrh_spe_filtered = pd.read_csv('/data4/linkaiqing/sm_pretrained/gp/qm9_all_raw_spe/1d_hnmr.csv')
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
hsqc = pd.read_csv('/data4/linkaiqing/sm_pretrained/gp/qm9_all_raw_spe/2d_hhsqc.csv')
peak_columns = [col for col in hsqc.columns if 'peak' in col]
hsqc[peak_columns] = (hsqc[peak_columns] - hsqc_min_value) / (hsqc_max_value - hsqc_min_value)
hsqc = hsqc.to_numpy()


# COSY
cosy_max_value = 14.0
cosy_min_value = -2.0
nmr_cosy = pd.read_csv('/data4/linkaiqing/sm_pretrained/gp/qm9_all_raw_spe/2d_hcosy.csv')
hxyh_columns = [col for col in nmr_cosy.columns if 'H_X_Y_H' in col]
nmr_cosy = nmr_cosy[hxyh_columns]
peak_columns = [col for col in nmr_cosy.columns if 'peak' in col]
nmr_cosy[peak_columns] = (nmr_cosy[peak_columns] - cosy_min_value) / (cosy_max_value - cosy_min_value)
nmr_cosy = nmr_cosy.to_numpy()

# # J2D
# j2d_max_value = 30.0
# j2d_min_value = -30.0
# j2d = pd.read_csv('/data4/linkaiqing/sm_pretrained/gp/qm9_all_raw_spe/2d_h_j2d.csv')
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
nmrf_spe_filtered = pd.read_csv('/data4/linkaiqing/sm_pretrained/gp/qm9_all_raw_spe/1d_fnmr.csv')
peak_columns = [col for col in nmrf_spe_filtered.columns if 'peak' in col]
nmrf_spe_filtered[peak_columns] = (nmrf_spe_filtered[peak_columns] - fnmr_min_value) / (fnmr_max_value - fnmr_min_value)
nmrf_spe_filtered = nmrf_spe_filtered.to_numpy()

# N-NMR  
nnmr_max_value = 400.0
nnmr_min_value = -260.0
nmrn_spe_filtered = pd.read_csv('/data4/linkaiqing/sm_pretrained/gp/qm9_all_raw_spe/1d_nnmr.csv')
peak_columns = [col for col in nmrn_spe_filtered.columns if 'peak' in col]
nmrn_spe_filtered[peak_columns] = (nmrn_spe_filtered[peak_columns] - nnmr_min_value) / (nnmr_max_value - nnmr_min_value)
nmrn_spe_filtered = nmrn_spe_filtered.to_numpy()

# O-NMR
onmr_max_value = 460.0
onmr_min_value = -385.0
nmro_spe_filtered = pd.read_csv('/data4/linkaiqing/sm_pretrained/gp/qm9_all_raw_spe/1d_onmr.csv')
peak_columns = [col for col in nmro_spe_filtered.columns if 'peak' in col]
nmro_spe_filtered[peak_columns] = (nmro_spe_filtered[peak_columns] - onmr_min_value) / (onmr_max_value - onmr_min_value)
nmro_spe_filtered = nmro_spe_filtered.to_numpy()

# combine all h-nmr and x-nmr features together
nmrh_spe_filtered = np.concatenate((nmrh_spe_filtered, hsqc, nmr_cosy, nmrf_spe_filtered, nmrn_spe_filtered, nmro_spe_filtered), axis=1)
# nmrh_spe_filtered = np.concatenate((nmrh_spe_filtered, hsqc, nmr_cosy, j2d, nmrf_spe_filtered, nmrn_spe_filtered, nmro_spe_filtered), axis=1)

print('nmrh_spe_filtered:', nmrh_spe_filtered.shape)


# zhipu
print('load high-mass file...')
mass = pd.read_csv('/data4/linkaiqing/sm_pretrained/gp/qm9_all_raw_spe/ms.csv')
high_mass_spe = mass.to_numpy()
print('high-mass_spe:', high_mass_spe.shape)


# atom type
atom_type = high_mass_spe[:, 1:-1]
print(f"Atom type shape: {atom_type.shape}")


# smiles
smiles_list = pd.read_csv('/data4/linkaiqing/sm_pretrained/gp/qm9_all_raw_spe/smiles.csv').values.tolist() ### [[smiles1], [smiles2], ...]
smiles_lengths = [len(smiles[0]) for smiles in smiles_list]
max_smiles_length = max(smiles_lengths)
max_seq_length = max_smiles_length + 2
print(f"SMILES 序列的最大长度为：{max_smiles_length}")
print(f"模型中应使用的 max_seq_length 为：{max_seq_length}")


# 获取所有辅助任务
# # Get the list of columns
# # auxiliary_data = pd.read_csv('/data4/linkaiqing/sm_pretrained/fangyang/gp/csv/smiles-transformer-master/aligned_smiles_id_aux_task_canonical.csv')
# auxiliary_data = pd.read_csv('/data4/linkaiqing/sm_pretrained/fangyang/gp/csv/smiles-transformer-master/aligned_smiles_id_aux_task.csv')
# columns = auxiliary_data.columns.tolist()
# # Exclude 'smiles' and 'id' columns to get auxiliary tasks
# auxiliary_tasks = [col for col in columns if col not in ['smiles', 'id']]
# print(f"Auxiliary tasks: {auxiliary_tasks}")



# file_prefixes = {
#     "c_nmr": '/data4/linkaiqing/sm_pretrained/fangyang/gp/csv/smiles-transformer-master/Auxiliary_Task/C_NMR_TA.csv',
#     "h_nmr": '/data4/linkaiqing/sm_pretrained/fangyang/gp/csv/smiles-transformer-master/Auxiliary_Task/H_NMR_TA.csv',
#     # "ir": '/data4/linkaiqing/sm_pretrained/fangyang/gp/csv/smiles-transformer-master/Auxiliary_Task/IR_TA.csv',
#     "ms": '/data4/linkaiqing/sm_pretrained/fangyang/gp/csv/smiles-transformer-master/Auxiliary_Task/MS_TA.csv',
# }
# auxiliary_data = pd.DataFrame()
# for prefix, filepath in file_prefixes.items():
#     df = pd.read_csv(filepath).iloc[:, 3:]
#     df.columns = [f"{prefix}_{col}" for col in df.columns]
#     auxiliary_data = pd.concat([auxiliary_data, df], axis=1)


auxiliary_data = pd.read_csv('/data4/linkaiqing/sm_pretrained/gp/aligned_smiles_id_aux_task.csv').iloc[:, 2:]


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


    

# 先读取数据集划分文件
train_df = pd.read_csv(f'/data4/linkaiqing/sm_pretrained/gp/csv/dataset/{data_split_mode}/train.csv')
val_df = pd.read_csv(f'/data4/linkaiqing/sm_pretrained/gp/csv/dataset/{data_split_mode}/val.csv')
test_df = pd.read_csv(f'/data4/linkaiqing/sm_pretrained/gp/csv/dataset/{data_split_mode}/test.csv')

# 如果有异常样本，先从数据划分文件中移除对应的SMILES
if len(all_abnormal_indices) > 0:
    print(f"Processing {len(all_abnormal_indices)} abnormal samples...")
    
    # 获取异常样本对应的SMILES
    abnormal_smiles = set()
    for idx in all_abnormal_indices:
        if idx < len(smiles_list):
            abnormal_smiles.add(smiles_list[idx][0])
    
    print(f"Found {len(abnormal_smiles)} unique abnormal SMILES")
    
    # 从各个数据集中移除异常SMILES
    original_train_size = len(train_df)
    original_val_size = len(val_df)
    original_test_size = len(test_df)
    
    train_df = train_df[~train_df['smiles'].isin(abnormal_smiles)].reset_index(drop=True)
    val_df = val_df[~val_df['smiles'].isin(abnormal_smiles)].reset_index(drop=True)
    test_df = test_df[~test_df['smiles'].isin(abnormal_smiles)].reset_index(drop=True)
    
    print(f"Train set: {original_train_size} -> {len(train_df)} (removed {original_train_size - len(train_df)})")
    print(f"Val set: {original_val_size} -> {len(val_df)} (removed {original_val_size - len(val_df)})")
    print(f"Test set: {original_test_size} -> {len(test_df)} (removed {original_test_size - len(test_df)})")
    
    # 然后过滤原始数据
    total_samples = len(smiles_list)
    normal_mask = np.ones(total_samples, dtype=bool)
    normal_mask[all_abnormal_indices] = False
    
    # 过滤所有数组
    ir_spe_filtered = ir_spe_filtered[normal_mask]
    uv_spe_filtered = uv_spe_filtered[normal_mask]
    nmrh_spe_filtered = nmrh_spe_filtered[normal_mask]
    nmrc_spe_filtered = nmrc_spe_filtered[normal_mask]
    high_mass_spe = high_mass_spe[normal_mask]
    atom_type = atom_type[normal_mask]
    
    # 使用原始的smiles_list创建过滤后的列表
    original_smiles_list = smiles_list.copy()
    smiles_list = [original_smiles_list[i] for i in range(total_samples) if normal_mask[i]]
    auxiliary_data = auxiliary_data[normal_mask].reset_index(drop=True)
    
    print(f"Filtered dataset: {total_samples} -> {len(smiles_list)} samples")

print(f"Final dataset size: {len(smiles_list)}")
print(f"确保数据一致性检查...")
assert len(smiles_list) == len(auxiliary_data) == ir_spe_filtered.shape[0], "Data length mismatch after filtering!"

# 创建 SMILES 到索引的映射
smiles_to_index = {smiles[0]: idx for idx, smiles in enumerate(smiles_list)}

# 4. 获取各数据集的索引
train_indices, train_missing_smiles = get_indices(train_df['smiles'], smiles_to_index)
val_indices, val_missing_smiles = get_indices(val_df['smiles'], smiles_to_index)
test_indices, test_missing_smiles = get_indices(test_df['smiles'], smiles_to_index)


# 划分训练集数据
train_ir_spe_filtered = ir_spe_filtered[train_indices]
train_uv_spe_filtered = uv_spe_filtered[train_indices]
train_nmrh_spe_filtered = nmrh_spe_filtered[train_indices]
train_nmrc_spe_filtered = nmrc_spe_filtered[train_indices]
train_high_mass_spe = high_mass_spe[train_indices]
train_smiles_list = [smiles_list[idx] for idx in train_indices]
train_aux_data = auxiliary_data.iloc[train_indices].reset_index(drop=True)
atom_types_list_train = atom_type[train_indices]



# 划分验证集数据
val_ir_spe_filtered = ir_spe_filtered[val_indices]
val_uv_spe_filtered = uv_spe_filtered[val_indices]
val_nmrh_spe_filtered = nmrh_spe_filtered[val_indices]
val_nmrc_spe_filtered = nmrc_spe_filtered[val_indices]
val_high_mass_spe = high_mass_spe[val_indices]
val_smiles_list = [smiles_list[idx] for idx in val_indices]
val_aux_data = auxiliary_data.iloc[val_indices].reset_index(drop=True)
atom_types_list_val = atom_type[val_indices]

# 划分测试集数据
test_ir_spe_filtered = ir_spe_filtered[test_indices]
test_uv_spe_filtered = uv_spe_filtered[test_indices]
test_nmrh_spe_filtered = nmrh_spe_filtered[test_indices]
test_nmrc_spe_filtered = nmrc_spe_filtered[test_indices]
test_high_mass_spe = high_mass_spe[test_indices]
test_smiles_list = [smiles_list[idx] for idx in test_indices]
test_aux_data = auxiliary_data.iloc[test_indices].reset_index(drop=True)
atom_types_list_test = atom_type[test_indices]



# 定义 count_tasks 和 binary_tasks
count_tasks = [at for at in auxiliary_tasks if 'Has' not in at and 'Is' not in at]
binary_tasks = [at for at in auxiliary_tasks if 'Has' in at or 'Is' in at]


# 创建训练集数据集
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

# 创建验证集数据集
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

# 创建测试集数据集
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

# 创建训练集数据加载器
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 创建验证集数据加载器
val_dataloader = DataLoader(
    val_dataset,
    batch_size=128, 
    shuffle=False,
    num_workers=4,
    drop_last=True,
)

# 创建测试集数据加载器（如果需要）
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)





# 计算每个计数任务的类别数
count_task_classes = {}
for task in count_tasks:
    max_value = int(auxiliary_data[task].max())
    count_task_classes[task] = max_value + 1  # 类别数



# # 实例化模型时，传递 count_task_classes
# model = AtomPredictionModel(vocab_size, count_task_classes, binary_tasks).to(device)


def load_model(model_path, vocab_size, char2idx):
    """
    加载预训练模型
    """
    # 初始化模型
    model = AtomPredictionModel(vocab_size=vocab_size, count_tasks_classes=None, binary_tasks=None)
    model.to(device)

    # 加载模型权重
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

 # 定义模型文件路径
model_path = '/data4/linkaiqing/sm_pretrained/fangyang/gp/csv/weights_scaffold_at/0806_ft.pth'

# 加载模型
model = load_model(model_path, vocab_size, char2idx)

# criterion = ContrastiveLoss()
# criterion = AtomPredictionLoss()
ignore_index = char2idx['<PAD>']
criterion = SMILESLoss(ignore_index)

count_task_loss_fn = nn.CrossEntropyLoss()
binary_task_loss_fn = nn.BCEWithLogitsLoss()

# 定义优化器，仅优化新增的 count_task_heads 和 binary_task_heads
# trainable_params = [param for name, param in model.named_parameters() if param.requires_grad]
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# 选择训练方式
use_semantic_supervision = True  # 设置为True使用语义监督，False使用原始训练

if use_semantic_supervision:
    # 使用语义监督的损失函数
    ignore_index = char2idx['<PAD>']
    semantic_criterion = SemanticSupervisedSMILESLoss(
        ignore_index=ignore_index, 
        aux_weight=0.2,  # 辅助任务权重
        count_weight=1.0,  # 计数任务权重
        binary_weight=1.0  # 二元任务权重
    )
    
    print("🚀 开始使用语义监督的训练...")
    print(f"📊 重要的计数任务: {semantic_criterion.important_count_tasks}")
    print(f"🔢 重要的二元任务: {semantic_criterion.important_binary_tasks}")
    
    # Train the model with semantic supervision
    train_with_semantic_supervision(
        model,
        semantic_criterion,
        optimizer,
        train_dataloader,
        val_dataloader,
        epochs=1000,
        save_dir=f'/data4/linkaiqing/sm_pretrained/fangyang/gp/csv/weights_{data_split_mode}_semantic',
        use_adaptive_weight=True  # 使用自适应权重调整
    )
else:
    # 使用原始训练函数（仅SMILES损失）
    print("📚 使用原始训练方式（仅SMILES损失）...")
    train(
        model,
        criterion,
        optimizer,
        train_dataloader,
        val_dataloader,
        epochs=1000,
        save_dir=f'/data4/linkaiqing/sm_pretrained/fangyang/gp/csv/weights_{data_split_mode}_original'
    )
