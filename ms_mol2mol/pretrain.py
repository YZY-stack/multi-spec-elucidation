import os
import re
import math
import random
import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler, IterableDataset
from torch.nn.parallel import DistributedDataParallel as DDP



from rdkit import Chem, RDLogger, DataStructs
RDLogger.DisableLog('rdApp.*')


import re
import torch
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from rdkit import Chem


# ============================
# Tokenizer 与数据预处理工具函数（只针对 C, N, O, F）
# ============================
def tokenize_smiles(smiles):
    pattern = r'''
        (\[[CNOF][^\]]*\]) |    # 匹配方括号内的原子，如 [C+], [N-] 等（要求第一个字母为 C、N、O、F）
        (%\d{2})         |      # 匹配两位数的环闭标记，如 %12
        ([CNOF])        |       # 匹配单个原子符号 C, N, O, F
        (\d)           |       # 匹配环闭数字（一个或多个数字）
        ([=#\-\+\(\)/\\])       # 匹配化学键、括号和斜杠等符号
    '''
    tokens = re.findall(pattern, smiles, re.VERBOSE)
    # 每个匹配返回的是一个元组，取其中非空的部分
    token_list = [next(filter(None, t)).strip() for t in tokens if any(t)]
    return token_list

def smiles_to_indices(smiles, char2idx, max_length):
    """
    将 SMILES 转换为 token decode
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
    unk_token = char2idx.get('<UNK>')

    masked_indices = indices.copy()
    mask_positions = [False] * len(indices)

    for i in range(1, len(indices)-1):
        # 跳过特殊 token
        if indices[i] in [pad_token, sos_token, eos_token, unk_token]:
            continue
        if random.random() < mask_prob:
            mask_positions[i] = True
            if random.random() < 0.3:
                masked_indices[i] = mask_token_idx
            # 70% 保持原样
    return masked_indices, mask_positions





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
            continue
            #mol_weight += Descriptors.AtomicWeight(symbol)
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







# ============================
# 3. In-memory 数据集（Dataset）
# ============================
import glob

class SMILESPretokenDataset(Dataset):
    def __init__(self, csv_path, char2idx, max_seq_length, compute_atom_types_fn, mask_prob=1.0):
        """
        csv_path: CSV 文件路径或文件夹。
                  如果是文件夹，则加载该文件夹下所有 CSV 文件（每个 CSV 文件第一行为标题，列名为 "SMILES"）；
                  如果是 CSV 文件，则只加载该文件的数据。
        char2idx: SMILES 到索引的映射字典
        max_seq_length: 最大序列长度（含 <SOS>, <EOS>, padding）
        compute_atom_types_fn: 根据 SMILES 返回 atom-type 特征向量的函数
        mask_prob: mask 的概率
        """
        self.smiles_list = []
        # 如果传入的是文件夹，则加载所有 csv 文件；如果是单个 csv 文件，则只加载该文件
        if os.path.isdir(csv_path):
            csv_files = sorted(glob.glob(os.path.join(csv_path, "*.csv")))
        else:
            csv_files = [csv_path]
        print(f"Found {len(csv_files)} CSV file(s). Loading...")
        for csv_file in tqdm(csv_files):
            df = pd.read_csv(csv_file)
            smiles = df["SMILES"].tolist()
            self.smiles_list.extend(smiles)
        print(f"Total molecules loaded: {len(self.smiles_list)}")
        self.mask_prob = mask_prob
        self.char2idx = char2idx
        self.max_seq_length = max_seq_length
        self.compute_atom_types = compute_atom_types_fn
        self.mask_token_idx = char2idx.get('<MASK>')
        # 假设所有 SMILES 均有效
        self.valid_indices = list(range(len(self.smiles_list)))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        smiles = self.smiles_list[real_idx]
        true_indices = smiles_to_indices(smiles, self.char2idx, self.max_seq_length)
        src_indices, mask_positions = mask_smiles_indices(true_indices, self.mask_token_idx, self.char2idx, mask_prob=self.mask_prob)
        src_indices = torch.tensor(src_indices, dtype=torch.long)
        true_indices = torch.tensor(true_indices, dtype=torch.long)
        mask_positions = torch.tensor(mask_positions, dtype=torch.bool)
        atom_types = self.compute_atom_types(smiles)
        atom_types = torch.tensor(atom_types, dtype=torch.float32)
        return src_indices, true_indices, atom_types, mask_positions


    




# ============================
# 4. Positional Encoding
# ============================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)






# ============================
# 5. Encoder 增强模块：局部卷积
# ============================
class EncoderWithConv(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super(EncoderWithConv, self).__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        # x: [seq_len, B, d_model]
        x = x.permute(1, 2, 0)  # [B, d_model, seq_len]
        x = self.conv(x)         # [B, d_model, seq_len]
        x = x.permute(2, 0, 1)     # [seq_len, B, d_model]
        return x






# ============================
# 6. 模型（Transformer Encoder-Decoder）
# ============================

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(CustomTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, **kwargs):
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt




class MoleculePretrainingModel(nn.Module):
    def __init__(self, vocab_size, atom_type_dim, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1):
        super(MoleculePretrainingModel, self).__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        # 采用卷积层增强局部特征，然后堆叠 Transformer Encoder
        self.encoder = nn.Sequential(
            EncoderWithConv(d_model),
            nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        )
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_linear = nn.Linear(d_model, vocab_size)
        
        # atom_type 投影
        self.atom_type_proj = nn.Linear(atom_type_dim, d_model)
        # 用于将 atom_type 信息作为全局特征与 decoder 初始状态融合
        self.decoder_init_proj = nn.Linear(d_model, d_model)
    
    def forward(self, src_seq, tgt_seq, atom_types, tgt_mask=None, memory_key_padding_mask=None):
        batch_size, src_len = src_seq.size()
        batch_size, tgt_len = tgt_seq.size()
        
        # Encoder
        src_emb = self.embedding(src_seq) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        src_emb = src_emb.transpose(0, 1)  # [src_len, B, d_model]
        memory = self.encoder(src_emb)       # [src_len, B, d_model]
        
        # Decoder
        tgt_emb = self.embedding(tgt_seq) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)  # [tgt_len, B, d_model]
        
        # 将 atom_type 作为全局特征，经过投影后作为 decoder 初始 token
        atom_emb = self.atom_type_proj(atom_types)   # [B, d_model]
        decoder_init = self.decoder_init_proj(atom_emb).unsqueeze(0)  # [1, B, d_model]
        # 将 decoder 初始状态拼接到 tgt_emb 前面
        tgt_emb = torch.cat([decoder_init, tgt_emb], dim=0)  # [1+tgt_len, B, d_model]
        
        # 生成 tgt_mask 时需考虑序列长度已增加1
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(0)).to(tgt_seq.device)
        
        decoder_output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                                      memory_key_padding_mask=memory_key_padding_mask)
        # 去掉第一个 token（decoder 初始状态），还原为 [tgt_len, B, d_model]
        decoder_output = decoder_output[1:, :, :].transpose(0, 1)  # [B, tgt_len, d_model]
        output_logits = self.output_linear(decoder_output)         # [B, tgt_len, vocab_size]
        return output_logits





# ============================
# 3. SMILES 词表与映射（仅包含 C、N、O、F 及相关符号）
# ============================
SMILES_VOCAB = [
    # 特殊 Token
    '<PAD>', '<SOS>', '<EOS>', '<UNK>', '<MASK>',
    
    # 原子符号（基础）
    'C', 'N', 'O', 'F',
    
    # 带电原子形式（补充完整）
    '[C]', '[CH]', '[CH2]', '[CH3]', 
    '[N+]', '[N-]', '[NH+]', '[NH2+]', '[NH3+]',
    '[O-]', '[OH+]',
    
    # 化学符号
    '(', ')', '[', ']', '=', '#', '-', '+', '/', '\\',
    
    # 环闭标记（两位数）
    *[f'%{i}' for i in range(10, 100)],
    
    # 数字（0-9 和 10-99）
    #*[str(i) for i in range(100)],
    *[str(i) for i in range(10)],

    # 补充常见同位素标记
    '[13C]', '[14C]', '[15N]'
]


vocab_size = len(SMILES_VOCAB)
char2idx = {token: idx for idx, token in enumerate(SMILES_VOCAB)}
idx2char = {idx: token for idx, token in enumerate(SMILES_VOCAB)}

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
# 8. 训练函数（DDP + tqdm + 学习率 Warmup + 动态梯度裁剪 + 有效性检查）
# ============================
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import logging


############################################
# 设置日志记录器
############################################
def setup_logger(log_file):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
    return logger


def get_mask_prob(epoch, total_epochs, start_ratio=0.0, end_ratio=1.0, power=0.5):
    """
    计算当前 epoch 的 mask_ratio，采用非线性上升策略。

    参数:
      - epoch: 当前 epoch（从 0 开始计数）
      - total_epochs: 总的 epoch 数量
      - start_ratio: 起始 mask_ratio，默认为 0.0
      - end_ratio: 最终 mask_ratio，默认为 1.0
      - power: 指数幂参数，用于控制上升速度。小于 1 会使前期上升较快，建议设置为 0.5

    例如，若 total_epochs=10，则：
      - epoch=0 时：mask_ratio = 0
      - epoch=2 时：mask_ratio ≈ ((2/9)^0.5) ≈ 0.47
      - epoch=3 时：mask_ratio ≈ ((3/9)^0.5) ≈ 0.58
      - epoch=9 时：mask_ratio = 1.0
    """
    if total_epochs <= 1:
        return end_ratio
    # 注意：这里 epoch 从 0 到 total_epochs-1
    ratio = (epoch / (total_epochs - 1)) ** power
    return start_ratio + (end_ratio - start_ratio) * ratio


############################################
def train(rank, world_size, csv_folder, num_epochs=10, batch_size=128, max_seq_length=300):
    # 设置多卡训练的环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10314'
    
    # 初始化分布式进程组
    dist.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)
    torch.manual_seed(42)
    device = torch.device('cuda', rank)
    
    # 仅在 rank==0 初始化 TensorBoard writer 和日志记录器
    if rank == 0:
        writer = SummaryWriter(log_dir="runs/exp")
        logger = setup_logger("training.log")
        logger.info("Training started.")
    
    # 获取 CSV 文件列表（假设 csv_folder 下正好有 10 个 CSV 文件）
    csv_files = sorted(glob.glob(os.path.join(csv_folder, "*.csv")))
    assert len(csv_files) == num_epochs, f"Expected {num_epochs} CSV files, but found {len(csv_files)}"
    
    # 预先计算所有 epoch 的总步数（仅供 lr scheduler 使用）
    total_steps = 0
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        num_samples = len(df)
        num_batches = math.ceil(num_samples / batch_size)
        total_steps += num_batches

    global_step = 0

    # 初始化模型、损失、优化器、调度器（模型保持不变，所有 epoch 共用）
    atom_type_dim = len(compute_atom_types("C"))
    print(atom_type_dim)
    model = MoleculePretrainingModel(vocab_size, atom_type_dim, d_model=512, nhead=8,
                                     num_encoder_layers=6, num_decoder_layers=6,
                                     dim_feedforward=2048, dropout=0.1)
    

    checkpoint_path = "./512_molecule_pretraining_model_epoch4.pth"
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Rank {rank} loaded pretrained weights.")
    else:
        print(f"Rank {rank} checkpoint not found.")

    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx.get('<PAD>'), reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    def lr_lambda(step):
        warmup_steps = 10000
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, (total_steps - warmup_steps))))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # 每个 epoch 使用不同的 dataloader（同一模型）
    for epoch in range(4, num_epochs):
        current_csv_file = csv_files[epoch]
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs} using CSV file: {current_csv_file}")
        
        
        # 根据当前 epoch 动态计算 mask_ratio
        current_mask_prob = get_mask_prob(epoch, num_epochs, start_ratio=0.0, end_ratio=1.0, power=0.7)

        # 重新初始化当前 epoch 的 dataset、sampler、dataloader
        dataset = SMILESPretokenDataset(current_csv_file, char2idx, max_seq_length, compute_atom_types, mask_prob=current_mask_prob)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        
        # 设置 sampler 的 epoch，以便打乱顺序
        sampler.set_epoch(epoch)
        
        total_loss = 0.0
        if rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)
        else:
            pbar = dataloader
        
        for i, (src_seq, tgt_seq, atom_types, mask_positions) in enumerate(pbar):
            src_seq = src_seq.to(device)
            tgt_seq = tgt_seq.to(device)
            atom_types = atom_types.to(device)
            mask_positions = mask_positions.to(device)
            
            # Teacher Forcing：decoder 输入为 tgt_seq 去掉最后一个 token，标签为 tgt_seq 去掉第一个 token
            tgt_input = tgt_seq[:, :-1]
            tgt_output = tgt_seq[:, 1:]
            mask_positions_target = mask_positions[:, 1:]
            
            # 由于 decoder 输入中 prepend 了 atom_type 特征，故 mask 长度为 1+tgt_len
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1) + 1).to(device)
            
            optimizer.zero_grad()
            output_logits = model(src_seq, tgt_input, atom_types, tgt_mask=tgt_mask)
            loss_tensor = criterion(output_logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
            loss = torch.mean(loss_tensor)
            
            loss.backward()
            current_max_norm = 1.0 - 0.9 * (global_step / total_steps)
            current_max_norm = max(0.1, current_max_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=current_max_norm)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            global_step += 1
            
            if rank == 0:
                writer.add_scalar("Loss/iteration", loss.item(), global_step)
                writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], global_step)
                pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
                if (i + 1) % 100 == 0:
                    pred_tokens = torch.argmax(output_logits[0], dim=-1).cpu().tolist()
                    true_tokens = tgt_output[0].cpu().tolist()
                    mask_tokens = src_seq[0].cpu().tolist()
                    mask_positions_list = mask_positions[0].cpu().tolist()  # bool列表，True表示该位置被mask

                    # 利用辅助函数将 token 索引转换为 SMILES 字符串
                    pred_smiles = decode_indices(pred_tokens, idx2char)
                    true_smiles = decode_indices(true_tokens, idx2char)
                    mask_smiles = decode_indices(mask_tokens, idx2char)

                    # 额外构造一个被mask掉的 token 列表，展示哪些 token 在原始序列中被mask了
                    masked_true_tokens = [idx2char[t] for t, m in zip(true_tokens, mask_positions_list) if m]

                    log_msg = (
                        f"Step {global_step}: Loss {loss.item():.4f}\n"
                        f"True SMILES   : {true_smiles}\n"
                        f"Masked SMILES : {mask_smiles}\n"
                        f"Masked Tokens : {masked_true_tokens}\n"
                        f"Pred SMILES   : {pred_smiles}"
                    )
                    print("\n" + log_msg)
                    logger.info(log_msg)
                    writer.add_text("Predictions", log_msg, global_step)

        
        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            epoch_msg = f"Epoch [{epoch+1}/{num_epochs}] finished, Avg Loss: {avg_loss:.4f}"
            print(epoch_msg)
            logger.info(epoch_msg)
            writer.add_scalar("Loss/epoch", avg_loss, epoch)
            # 保存模型权重（同一模型）
            model_path = f"512_molecule_pretraining_model_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
            writer.add_text("Model Save", f"Model saved to {model_path}", epoch)
            writer.flush()
    
    if rank == 0:
        writer.close()
    dist.destroy_process_group()


# ============================
# 9. DDP 多卡训练入口
# ============================
def main():
    csv_folder = "./zinc20_data"  # CSV 文件所在目录（10个 CSV 文件）
    world_size = 8  # 使用 8 张 GPU
    mp.spawn(train, args=(world_size, csv_folder, 10, 128, 300), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

