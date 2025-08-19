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



from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch.utils.data as data


from collections import Counter
from math import sqrt


def cosine_similarity(s1, s2):
    counter1, counter2 = Counter(s1), Counter(s2)
    intersection = set(counter1.keys()) & set(counter2.keys())
    numerator = sum([counter1[x] * counter2[x] for x in intersection])

    sum1 = sum([counter1[x] ** 2 for x in counter1.keys()])
    sum2 = sum([counter2[x] ** 2 for x in counter2.keys()])
    denominator = sqrt(sum1) * sqrt(sum2)

    return numerator / denominator if denominator != 0 else 0.0




def test(epoch, model, dataloader, char2idx, idx2char, max_seq_length=100):
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
        for i, (ir, raman, c_spec, h_spec, low_mass, high_mass, smiles_indices, auxiliary_targets, atom_types, coordinates) in enumerate(tqdm(dataloader, desc="Evaluating", ncols=100)):
            # Move data to device
            ir = ir.to(device)
            raman = raman.to(device)
            c_spec = c_spec.to(device)
            h_spec = h_spec.to(device)
            low_mass = low_mass.to(device)
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

            # Prediction
            predicted_smiles_list = predict_greedy(
                model, ir, raman, c_spec, h_spec, low_mass, high_mass,
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
                    maccs_sim.append(DataStructs.FingerprintSimilarity(
                        MACCSkeys.GenMACCSKeys(mol_output), MACCSkeys.GenMACCSKeys(mol_gt),
                        metric=DataStructs.TanimotoSimilarity))
                    rdk_sim.append(DataStructs.FingerprintSimilarity(
                        Chem.RDKFingerprint(mol_output), Chem.RDKFingerprint(mol_gt),
                        metric=DataStructs.TanimotoSimilarity))
                    morgan_sim.append(DataStructs.TanimotoSimilarity(
                        AllChem.GetMorganFingerprint(mol_output, 2),
                        AllChem.GetMorganFingerprint(mol_gt, 2)))
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

                # Compute Cos-sim (assuming you have a function for this)
                cos_sim = cosine_similarity(predicted_smiles_str, true_smiles_str)
                total_cos_score += cos_sim

                # Compute Levenshtein Distance
                l_dis = lev(predicted_smiles_str, true_smiles_str)
                levs.append(l_dis)

        avg_bleu_score = total_bleu_score / total_smiles
        accuracy = total_correct / total_smiles
        # Compute validity
        validity = (total_smiles - bad_sm_count) / total_smiles
        cos_sim_all = total_cos_score / total_smiles
        exact = n_exact / total_smiles

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

        return results_dict, None



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
        for i, (ir, raman, c_spec, h_spec, low_mass, high_mass, smiles_indices, auxiliary_targets, atom_types, coordinates) in enumerate(tqdm(dataloader, desc="Evaluating", ncols=100)):
            # Move data to device
            ir = ir.to(device)
            raman = raman.to(device)
            c_spec = c_spec.to(device)
            h_spec = h_spec.to(device)
            low_mass = low_mass.to(device)
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

            if epoch < 1000:
                # Greedy prediction
                predicted_smiles_list = predict_greedy(
                    model, ir, raman, c_spec, h_spec, low_mass, high_mass,
                    char2idx, idx2char, max_seq_length=max_seq_length, atom_types=atom_types)
            else:
                predicted_smiles_list = predict_beam_search_batch(
                    model, ir, raman, c_spec, h_spec, low_mass, high_mass, 
                    char2idx, idx2char, max_seq_length=100, beam_width=5, atom_types=atom_types
                )



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
def predict_greedy(model, ir, raman, c_spec, h_spectrum, low_mass, high_mass, char2idx, idx2char, max_seq_length=100, atom_types=None):
    model.eval()
    with torch.no_grad():
        # Split h_spectrum into h_spectrum_part, f_spectrum, n_spectrum
        # h_spectrum_part = h_spectrum[:, :300]
        # f_spectrum = h_spectrum[:, 300:312]
        # n_spectrum = h_spectrum[:, 312:326]
        # o_spectrum = h_spectrum[:, 326:]

        # h_spectrum_part = h_spectrum[:, :105]
        # f_spectrum = h_spectrum[:, 105:117]
        # n_spectrum = h_spectrum[:, 117:131]
        # o_spectrum = h_spectrum[:, 131:]

        # h_spectrum_part = h_spectrum[:, :490]
        # f_spectrum = h_spectrum[:, 490:502]
        # n_spectrum = h_spectrum[:, 502:516]
        # o_spectrum = h_spectrum[:, 516:]

        # # 1D NMR only with HSQC
        # h_spectrum_part = h_spectrum[:, :225]
        # f_spectrum = h_spectrum[:, 225:237]
        # n_spectrum = h_spectrum[:, 237:251]
        # o_spectrum = h_spectrum[:, 251:]

        # # 1D NMR only with HSQC with COSY
        # h_spectrum_part = h_spectrum[:, :405]
        # f_spectrum = h_spectrum[:, 405:417]
        # n_spectrum = h_spectrum[:, 417:431]
        # o_spectrum = h_spectrum[:, 431:]

        # 1D NMR only with HSQC with COSY with J2D
        h_spectrum_part = h_spectrum[:, :595]
        f_spectrum = h_spectrum[:, 595:607]
        n_spectrum = h_spectrum[:, 607:621]
        o_spectrum = h_spectrum[:, 621:]

        # Prepare features
        features = {
            'ir': ir,
            'raman': raman,
            'nmr_c': c_spec,
            'nmr_h': h_spectrum_part,
            'f_spectrum': f_spectrum,
            'n_spectrum': n_spectrum,
            'o_spectrum': o_spectrum,
            # 'mass_low': low_mass,
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



def predict_beam_search_batch(model, ir, raman, c_spec, h_spec, low_mass, high_mass,
                              char2idx, idx2char, max_seq_length=100, beam_width=5, atom_types=None):
    model.eval()
    batch_size = ir.size(0)
    device = ir.device
    with torch.no_grad():
        # Get fusion features
        spectra_feat = model.spectra_encoder(ir, raman)
        nmr_feat = model.nmr_encoder(c_spec, h_spec)
        mass_feat = model.mass_spectrum_encoder(low_res=low_mass, high_res=high_mass, atom_types=atom_types)
        fusion_feat = model.fusion_encoder(spectra_feat, mass_feat, nmr_feat)  # Shape: [batch_size, d_model]

        # Prepare memory for the decoder
        memory = fusion_feat.unsqueeze(0)  # Shape: [1, batch_size, d_model]

        # Initialize beams for each batch element
        beams = []
        for i in range(batch_size):
            beams.append([{'seq': torch.tensor([char2idx['<SOS>']], device=device),
                           'score': 0.0,
                           'is_end': False}])

        for _ in range(max_seq_length):
            is_done = True  # Flag to check if all sequences have ended
            for i in range(batch_size):
                candidates = []
                for beam_item in beams[i]:
                    seq = beam_item['seq']
                    score = beam_item['score']
                    is_end = beam_item['is_end']

                    if is_end:
                        # Keep the completed sequence in the beam
                        candidates.append(beam_item)
                        continue

                    is_done = False  # At least one sequence is still active

                    # Prepare input for the decoder
                    tgt_indices = seq.unsqueeze(1)  # [seq_len, 1]
                    tgt_mask = model.smiles_decoder.generate_square_subsequent_mask(tgt_indices.size(0)).to(device)

                    # Memory for current batch item
                    current_memory = memory[:, i, :].unsqueeze(1)  # [1, 1, d_model]

                    # Adjust atom_types if necessary
                    if atom_types is not None:
                        atom_type = atom_types[i].unsqueeze(0)  # [1, feature_size]
                    else:
                        atom_type = None

                    # Pass through the decoder
                    output = model.smiles_decoder(tgt_indices, current_memory, tgt_mask=tgt_mask, atom_types=atom_type)  # [seq_len, 1, vocab_size]
                    output_logits = output[-1, 0, :]  # [vocab_size]
                    log_probs = F.log_softmax(output_logits, dim=-1)

                    # Get top k candidates
                    topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

                    for k in range(beam_width):
                        next_token = topk_indices[k].item()
                        next_score = score + topk_log_probs[k].item()
                        next_seq = torch.cat([seq, torch.tensor([next_token], device=device)])
                        candidates.append({
                            'seq': next_seq,
                            'score': next_score,
                            'is_end': next_token == char2idx['<EOS>']
                        })

                # Sort candidates and select top beam_width sequences
                candidates.sort(key=lambda x: x['score'], reverse=True)
                beams[i] = candidates[:beam_width]

            if is_done:
                break  # Exit if all sequences have completed

        # Prepare final outputs
        generated_smiles_list = []
        for i in range(batch_size):
            # Select the sequence with the highest score
            beams[i].sort(key=lambda x: x['score'], reverse=True)
            best_seq = beams[i][0]['seq']

            # Convert indices to SMILES string
            token_indices = best_seq.cpu().numpy()
            tokens = []
            for idx in token_indices:
                if idx == char2idx['<EOS>']:
                    break
                elif idx not in [char2idx['<PAD>'], char2idx['<SOS>']]:
                    tokens.append(idx2char.get(idx, '<UNK>'))
            smiles_str = ''.join(tokens)
            generated_smiles_list.append(smiles_str)

        return generated_smiles_list





def train(model, smiles_loss_fn, optimizer, train_dataloader, val_dataloader, epochs=10, save_dir='./model_weights_smiles'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_bleu_score = 0.0  # 用于保存最佳模型
    feature_loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", total=len(train_dataloader), ncols=100)
        for i, (ir, raman, c_spec, h_spec, low_mass, high_mass, smiles_indices, auxiliary_targets, atom_types, coordinates) in enumerate(progress_bar):
            # Move data to device
            ir = ir.to(device)
            raman = raman.to(device)
            c_spec = c_spec.to(device)
            h_spec = h_spec.to(device)
            low_mass = low_mass.to(device)
            high_mass = high_mass.to(device)
            smiles_indices = smiles_indices.to(device)
            auxiliary_targets = auxiliary_targets.to(device)
            atom_types = atom_types.to(device)
            coordinates = coordinates.to(device)

            optimizer.zero_grad()

            # Prepare target sequence
            tgt_seq = smiles_indices.transpose(0, 1)[:-1]
            tgt_output = smiles_indices.transpose(0, 1)[1:]

            # Generate mask
            seq_len = tgt_seq.size(0)
            tgt_mask = model.smiles_decoder.generate_square_subsequent_mask(seq_len).to(device)



            # Forward pass with return_mol_output=True
            output_spectra, output_mol, spectra_feat, mol_feat, count_task_outputs, binary_task_outputs = model(
                ir, raman, c_spec, h_spec, low_mass, high_mass, tgt_seq, tgt_mask, atom_types, coordinates, return_mol_output=True)

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
            # if val_bleu_score > 0.89:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ir_cnmr_for_attn.pth'))



import torch.nn.functional as F

# 预先定义的 SMILES 字符集
SMILES_VOCAB = ['<PAD>', '<SOS>', '<EOS>', '<UNK>',
#                 'C', '#', '1', '(', '=', 'O', 
#                 ')', 'n', 'c', 'N', '2', '[nH]', 
#                 '3', 'o', 'F', '4', '[N+]', '[O-]', '5', '-'
# ]
                'C', 'N', 'O', 'F',
                '1', '2', '3', '4', '5',
                '#', '=', '(', ')',
                ]

                # 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
                # 'H', 'B', 'Si', 'Se', 'se',
                # 'c', 'n', 'o', 's', 'p',
                # '(', ')', '[', ']', '=', '#', '-', '+', '@', '.', '/', '\\',
                # '%', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
vocab_size = len(SMILES_VOCAB)

# 创建字符到索引的映射和索引到字符的映射
char2idx = {token: idx for idx, token in enumerate(SMILES_VOCAB)}
idx2char = {idx: token for idx, token in enumerate(SMILES_VOCAB)}




scaler = StandardScaler()

# ir and raman
print('load raman file...')
raman_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/sparse_raman_wsmiles.csv').iloc[:, 1:].to_numpy()
# raman_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/raman_spe_filtered_values.csv', header=None).to_numpy()
print('load ir file...')
# ir_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/ir_spe_filtered_values.csv', header=None).to_numpy()
ir_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/sparse_ir_wsmiles.csv').iloc[:, 1:].to_numpy()
print('raman_spe_filtered:', raman_spe_filtered.shape)
print('ir_spe_filtered:', ir_spe_filtered.shape)

# nmr
# nmrh_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/nmrh_spe_filtered_values.csv', header=None).to_numpy()
# nmrc_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/nmrc_spe_filtered_values.csv', header=None).to_numpy()
nmrc_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/sparse_cnmr_new_new.csv').to_numpy()
# twoD_nmr = pd.read_csv('/root/workspace/smiles-transformer-master/csv/2d_c_nmr.csv')
# peak_columns = [col for col in twoD_nmr.columns if 'peak' in col]
# twoD_nmr[peak_columns] = scaler.fit_transform(twoD_nmr[peak_columns])
# twoD_nmr = twoD_nmr.to_numpy()
# twod_twod = pd.read_csv('/root/workspace/smiles-transformer-master/csv/2D_NMR/2D_2D/13C_13C_INADEQUATE_DEPT/13C_13C_INADEQUATE_DEPT.csv').iloc[:, 8:]
# peak_columns = [col for col in twod_twod.columns if 'peak' in col]
# twod_twod[peak_columns] = scaler.fit_transform(twod_twod[peak_columns])
# twod_twod = twod_twod.to_numpy()
# # nmrc_spe_filtered = np.concatenate((nmrc_spe_filtered, twoD_nmr, twod_twod), axis=1)
# nmrc_spe_filtered = np.concatenate((nmrc_spe_filtered, twoD_nmr), axis=1)

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
nmrh_spe_filtered = np.concatenate((nmrh_spe_filtered, hsqc, nmrf_spe_filtered, nmrn_spe_filtered), axis=1)
# nmrh_spe_filtered = np.concatenate((nmrh_spe_filtered, hsqc, nmr_cosy, j2d, nmrf_spe_filtered, nmrn_spe_filtered), axis=1)
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



# import pdb;pdb.set_trace()


# 确保光谱数据、SMILES 列表和辅助任务数据长度一致
assert len(smiles_list) == len(auxiliary_data) == ir_spe_filtered.shape[0], "Data length mismatch!"





# 创建 SMILES 到索引的映射
# smiles_to_index = {smiles: idx for idx, smiles in enumerate(smiles_list)}
smiles_to_index = {smiles[0]: idx for idx, smiles in enumerate(smiles_list)}


# 加载训练集、验证集和测试集
train_df = pd.read_csv(f'/root/workspace/smiles-transformer-master/csv/dataset/{data_split_mode}/train.csv')
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

# 获取训练集索引
train_indices, train_missing_smiles = get_indices(train_df['smiles'], smiles_to_index)
# 获取验证集索引
val_indices, val_missing_smiles = get_indices(val_df['smiles'], smiles_to_index)
# 获取测试集索引
test_indices, test_missing_smiles = get_indices(test_df['smiles'], smiles_to_index)

# 打印缺失的 SMILES（如果有）
if train_missing_smiles:
    print(f"Missing smiles in train set: {train_missing_smiles}")
if val_missing_smiles:
    print(f"Missing smiles in val set: {val_missing_smiles}")
if test_missing_smiles:
    print(f"Missing smiles in test set: {test_missing_smiles}")


# 划分训练集数据
train_ir_spe_filtered = ir_spe_filtered[train_indices]
train_raman_spe_filtered = raman_spe_filtered[train_indices]
train_nmrh_spe_filtered = nmrh_spe_filtered[train_indices]
train_nmrc_spe_filtered = nmrc_spe_filtered[train_indices]
train_low_mass_spe = low_mass_spe[train_indices]
train_high_mass_spe = high_mass_spe[train_indices]
train_smiles_list = [smiles_list[idx] for idx in train_indices]
train_aux_data = auxiliary_data.iloc[train_indices].reset_index(drop=True)
atom_types_list_train = atom_type[train_indices]

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





# # Create a mapping from SMILES to (atom_types_vector, coordinates_vector)
# smiles_to_vectors = {}
# for item in atom_coor_smiles:
#     smiles = item['smiles']
#     atom_types_vector = item['atom_types_vector']
#     coordinates_vector = item['coordinates_vector']
#     smiles_to_vectors[smiles] = (atom_types_vector, coordinates_vector)


# # Initialize lists to hold atom types and coordinates
# atom_types_list_val = []
# coordinates_list_val = []
# missing_smiles_val = []

# for smiles in val_smiles_list:
#     smiles = smiles[0]
#     if smiles in smiles_to_vectors:
#         atom_types_vector, coordinates_vector = smiles_to_vectors[smiles]
#         atom_types_list_val.append(atom_types_vector)
#         coordinates_list_val.append(coordinates_vector)
#     else:
#         missing_smiles_val.append(smiles)

# # Check if there are any missing SMILES
# if missing_smiles_val:
#     print(f"Missing SMILES in training set: {missing_smiles_val}")


# atom_types_list_train = np.load('/root/workspace/smiles-transformer-master/csv/train_type.npy')
# coordinates_list_train = np.load('/root/workspace/smiles-transformer-master/csv/train_xyz.npy')
coordinates_list_train = atom_types_list_train

# atom_types_list_val = np.load('/root/workspace/smiles-transformer-master/csv/val_type.npy')
# coordinates_list_val = np.load('/root/workspace/smiles-transformer-master/csv/val_xyz.npy')
coordinates_list_val = atom_types_list_val

# Instantiate the model, loss, and optimizer
# model = ContrastiveLearningModel(trfm=trfm, vocab=vocab)


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


# 创建训练集数据集
train_dataset = SpectraDataset(
    ir_spectra=train_ir_spe_filtered,
    raman_spectra=train_raman_spe_filtered,
    c_spectra=train_nmrc_spe_filtered,
    h_spectra=train_nmrh_spe_filtered,
    low_mass_spectra=train_low_mass_spe,
    high_mass_spectra=train_high_mass_spe,
    smiles_list=train_smiles_list,
    auxiliary_data=train_aux_data,
    char2idx=char2idx,
    max_seq_length=max_seq_length,
    count_tasks=count_tasks,
    binary_tasks=binary_tasks,
    atom_types_list=atom_types_list_train, 
    coordinates_list=coordinates_list_train
)

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

# 创建测试集数据加载器
test_dataloader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4
)





# 计算每个计数任务的类别数
count_task_classes = {}
for task in count_tasks:
    max_value = int(auxiliary_data[task].max())
    count_task_classes[task] = max_value + 1  # 类别数

# 实例化模型时，传递 count_task_classes
model = AtomPredictionModel(vocab_size, count_task_classes, binary_tasks).to(device)



def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.1, 0.1)
model.apply(init_weights)

# criterion = ContrastiveLoss()
# criterion = AtomPredictionLoss()
ignore_index = char2idx['<PAD>']
criterion = SMILESLoss(ignore_index)

count_task_loss_fn = nn.CrossEntropyLoss()
binary_task_loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# Train the model
train(
    model,
    criterion,
    optimizer,
    train_dataloader,
    val_dataloader,
    epochs=1000,
    save_dir=f'/root/workspace/smiles-transformer-master/csv/weights_{data_split_mode}'
)


# Evaluate on test set
test_metrics, _ = test(
    0, model, test_dataloader, char2idx, idx2char, max_seq_length=max_seq_length)

for key, value in test_metrics.items():
    print(f"Test {key}: {value:.4f}")

print('all done.')