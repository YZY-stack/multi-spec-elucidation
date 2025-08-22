import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import torch
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
RDLogger.DisableLog('rdApp.*')
from collections import Counter
from math import sqrt

from metrics import *




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_split_mode = 'scaffold'




# Predefined SMILES character set
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

# Create character to index mapping and index to character mapping
char2idx = {token: idx for idx, token in enumerate(SMILES_VOCAB)}
idx2char = {idx: token for idx, token in enumerate(SMILES_VOCAB)}




def extract_transformer_memory(model,
                               ir_spectrum, 
                               raman_spectrum, 
                               c_spectrum, 
                               h_spectrum,
                               low_res_mass,
                               high_res_mass,
                               atom_types):
    """
    Given a single batch input, return the corresponding memory representation for that batch.
    """
    model.eval()
    with torch.no_grad():
        h_spectrum_part = h_spectrum[:, :595]
        f_spectrum = h_spectrum[:, 595:607]
        n_spectrum = h_spectrum[:, 607:621]
        o_spectrum = h_spectrum[:, 621:]

        # Keep consistent with the model's tokenizer part
        features = {
            # Add IR/Raman if your model uses them
            # "ir": ir_spectrum,
            # "raman": raman_spectrum,
            "nmr_c": c_spectrum,
            "nmr_h": h_spectrum_part,
            "f_spectrum": f_spectrum,
            "n_spectrum": n_spectrum,
            "o_spectrum": o_spectrum,
            # "mass_low": low_res_mass,
            "mass_high": high_res_mass
        }

        # Assume the model has a tokenizer that converts different spectral information into token embeddings
        tokens = model.tokenizer(features)  # [batch_size, seq_len, d_model] etc.

        # 如果模型的 transformer_encoder 需要的是 [seq_len, batch_size, d_model]
        # 则需要做一个 permute
        tokens = tokens.permute(1, 0, 2)  # [seq_len, batch_size, d_model]

        # 得到 transformer encoder 输出
        memory, attention = model.transformer_encoder(tokens)  # [seq_len, batch_size, d_model]
        
        # memory 的形状: [seq_len, batch_size, d_model]
        # 如果你想要一个向量表示，可以对 seq_len 维度做平均池化或取最后时刻等
        memory_avg = memory.mean(dim=0)  # [batch_size, d_model]
        return memory_avg


def visualize_model_features(model,
                             dataloader,
                             char2idx,
                             idx2char,
                             method='tsne',
                             save_plot='feature_space.png'):
    """
    遍历整个 dataloader，收集模型的 memory 表示，然后用 PCA/t-SNE 做 2D 降维并画图。
    
    参数:
    ----
    model: 你的模型
    dataloader: 测试/验证集 dataloader
    char2idx, idx2char: SMILES token 的映射字典
    method: 'tsne' 或 'pca'，选择可视化的方法
    save_plot: 保存图片的文件名
    """
    model.eval()
    
    # 用于存储所有样本的特征和标签
    all_features = []
    all_carbon_counts = []  # 例如，可以用碳原子数量作为可视化时的颜色标签
    ring_count_list = []
    # 也可以存其他信息，如是否预测正确、分子量等
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            # 从 batch 解包
            (ir_spectrum, raman_spectrum, c_spectrum, h_spectrum,
             low_res_mass, high_res_mass,
             smiles_indices, auxiliary_targets, atom_types, ring_count) = batch

            # 将数据转到 GPU (如果有的话)
            device = next(model.parameters()).device
            ir_spectrum = ir_spectrum.to(device)
            raman_spectrum = raman_spectrum.to(device)
            c_spectrum = c_spectrum.to(device)
            h_spectrum = h_spectrum.to(device)
            low_res_mass = low_res_mass.to(device)
            high_res_mass = high_res_mass.to(device)
            if atom_types is not None:
                atom_types = atom_types.to(device)

            # 得到当前 batch 的 memory 表示
            memory_avg = extract_transformer_memory(
                model,
                ir_spectrum, 
                raman_spectrum, 
                c_spectrum, 
                h_spectrum,
                low_res_mass,
                high_res_mass,
                atom_types
            )  # shape [batch_size, d_model]

            # 拼接到总数组中
            all_features.append(memory_avg.cpu().numpy())

            # 也可以存一些标签，用于画图时上色或区分
            # 比如，这里假设 atom_types[:, 2] 就是碳原子数
            # （注：具体要看你 atom_types 里的数据排布，如果不是这样，就要改）
            if atom_types is not None:
                for i in range(atom_types.size(0)):
                    # 假设 atom_types[i, 2] 就是碳原子数量
                    c_count = atom_types[i, 2].item()
                    all_carbon_counts.append(c_count)
                    ring = ring_count[i]
                    ring_count_list.append(ring+1)

            else:
                # 如果没有 atom_types，可选: 用 0 代替
                batch_size = memory_avg.size(0)
                all_carbon_counts.extend([0]*batch_size)

    # 整合所有 batch 的特征
    all_features = np.concatenate(all_features, axis=0)  # shape: [N, d_model]
    N = all_features.shape[0]
    indices = np.random.choice(N, 2000, replace=False)  # 不放回抽样
    sampled_features = all_features[indices]

    # 选择降维方法
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method {method}, please use 'tsne' or 'pca'.")

    print(f"Running {method.upper()} on {all_features.shape[0]} samples with dim={all_features.shape[1]}...")
    # X_embedded = reducer.fit_transform(all_features)  # shape: [N, 2]
    X_embedded = reducer.fit_transform(sampled_features)  # shape: [N, 2]
    # ring_count_list = [int(d.cpu().numpy()) for d in ring_count_list]
    all_carbon_counts = [int(d.cpu().numpy()) for d in all_carbon_counts]

    # 开始画图
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=np.array(ring_count_list)[indices],
        cmap='viridis',
        alpha=0.6
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label('Number of C atoms')
    plt.title(f'Feature Space Visualization ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # 保存图片
    plt.tight_layout()
    plt.savefig(save_plot, dpi=300)
    plt.close()
    print(f"Visualization saved to: {save_plot}")








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
    model_path = '/root/workspace/smiles-transformer-master/csv/weights_scaffold/1d_with_hsqc_cosy_j2d_2dCnmr_1d2d_scaffold.pth'

    # 加载模型
    model = load_model(model_path, vocab_size, char2idx)

    
    





    scaler = StandardScaler()

    # ir and raman
    print('load raman file...')
    raman_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/sparse_raman_wsmiles.csv').iloc[:, 1:].to_numpy()
    print('load ir file...')
    ir_spe_filtered = pd.read_csv('/root/workspace/smiles-transformer-master/csv/sparse_ir_wsmiles.csv').iloc[:, 1:].to_numpy()
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
    nmrc_spe_filtered = np.concatenate((nmrc_spe_filtered, twoD_nmr, twod_twod), axis=1)

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
    nmrh_spe_filtered = np.concatenate((nmrh_spe_filtered, hsqc, nmr_cosy, j2d, nmrf_spe_filtered, nmrn_spe_filtered), axis=1)
    # nmrc_spe_filtered = np.load('/root/workspace/smiles-transformer-master/csv/spin_nmrc_values.npy')
    # nmrh_spe_filtered = np.load('/root/workspace/smiles-transformer-master/csv/spin_nmrh_values.npy')


    # zhipu
    mass = pd.read_csv('/root/workspace/smiles-transformer-master/csv/sparse_ms_spectra_with_normalized_inten.csv')
    high_mass_spe = mass.iloc[:, 1:8].to_numpy()
    print('load high-mass file...')
    print('load low-mass file...')
    low_mass_spe = mass.iloc[:, 8:]
    peak_columns = [col for col in low_mass_spe.columns if 'peak' in col]
    low_mass_spe[peak_columns] = scaler.fit_transform(low_mass_spe[peak_columns])
    low_mass_spe = low_mass_spe.to_numpy()
    print('low-mass_spe:', low_mass_spe.shape)
    print('high-mass_spe:', high_mass_spe.shape)

    # smiles
    smiles_list = pd.read_csv('/root/workspace/smiles-transformer-master/csv/aligned_smiles.csv').values.tolist()

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
    auxiliary_data = pd.read_csv('/root/workspace/smiles-transformer-master/aligned_smiles_id_aux_task.csv')
    columns = auxiliary_data.columns.tolist()
    # Exclude 'smiles' and 'id' columns to get auxiliary tasks
    auxiliary_tasks = [col for col in columns if col not in ['smiles', 'id']]
    print(f"Auxiliary tasks: {auxiliary_tasks}")






    atom_type = high_mass_spe[:, :-1]
    print(f"Atom type shape: {atom_type.shape}")
    ring_count = pd.read_csv('/root/workspace/smiles-transformer-master/aligned_smiles_id_with_properties.csv')['ring_count'].to_numpy()







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
    ring_count_val = ring_count[val_indices]

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
    ring_count_test = ring_count[test_indices]




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
        coordinates_list=ring_count_val,
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
        coordinates_list=ring_count_test
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



    visualize_model_features(
        model,
        dataloader=test_dataloader,
        char2idx=char2idx,
        idx2char=idx2char,
        method='tsne',
        save_plot='tsne_feature_space.png'
    )