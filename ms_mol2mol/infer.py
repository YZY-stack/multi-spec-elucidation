import os
import math
import random
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger, DataStructs
RDLogger.DisableLog('rdApp.*')

from model import *
from utils import *

# -------------------------------
# 模型与词表定义
# -------------------------------

# 定义 SMILES 词表（与训练时一致）
SMILES_VOCAB = [
    # 特殊 Token
    '<PAD>', '<SOS>', '<EOS>', '<UNK>', '<MASK>',
    
    # 原子符号（基础）
    'C', 'N', 'O', 'F',
    
    # 带电原子形式（示例补充）
    '[C]', '[CH]', '[CH2]', '[CH3]', 
    '[N+]', '[N-]', '[NH+]', '[NH2+]', '[NH3+]',
    '[O-]', '[OH+]',
    
    # 化学符号
    '(', ')', '[', ']', '=', '#', '-', '+', '/', '\\',
    
    # 环闭标记（两位数）
    *[f'%{i}' for i in range(10, 100)],
    
    # 数字（0-9 和 10-99）
    *[str(i) for i in range(100)],
    
    # 补充常见同位素标记
    '[13C]', '[14C]', '[15N]'
]

#print(SMILES_VOCAB)

vocab_size = len(SMILES_VOCAB)
char2idx = {token: idx for idx, token in enumerate(SMILES_VOCAB)}
idx2char = {idx: token for idx, token in enumerate(SMILES_VOCAB)}
# atom_type 维度根据 compute_atom_types("C") 的长度
atom_type_dim = len(compute_atom_types("C"))




# -------------------------------
# 3. 定义用于自回归生成的辅助函数
# -------------------------------
def generate_smiles(model, src_seq, atom_types, max_length, char2idx, additional_info, device):
    """
    采用贪心解码策略生成完整 SMILES 序列：
      - src_seq: [1, seq_len]，为 mask 后的 SMILES 索引序列
      - atom_types: [1, atom_type_dim]，原始 SMILES 对应的 atom type 特征
      - max_length: 生成序列最大长度（不含 <SOS>）
      - additional_info: [max_C, max_N, max_O, max_F]，每个原子允许的最大数量
    返回生成的 token 索引列表（不含初始的 <SOS>）。
    """
    model.eval()
    with torch.no_grad():
        # 1. Encoder 部分
        src_emb = model.embedding(src_seq) * math.sqrt(model.d_model)
        src_emb = model.pos_encoder(src_emb)
        # Transformer Encoder 要求输入 shape 为 [seq_len, batch, d_model]
        src_emb = src_emb.transpose(0, 1)  # [seq_len, 1, d_model]
        memory = model.encoder(src_emb)      # [seq_len, 1, d_model]

        # 2. Decoder 部分初始设置：以 <SOS> 开始
        input_tgt = torch.tensor([[char2idx['<SOS>']]], device=device)  # [1, 1]

        for _ in range(max_length):
            # 当前 decoder 输入 embedding 与位置编码，shape: [tgt_len, 1, d_model]
            tgt_emb = model.embedding(input_tgt) * math.sqrt(model.d_model)
            tgt_emb = model.pos_encoder(tgt_emb)
            tgt_emb = tgt_emb.transpose(0, 1)  # [tgt_len, 1, d_model]

            # 利用 atom type 特征计算 decoder 初始 token
            atom_emb = model.atom_type_proj(atom_types)  # [1, d_model]
            decoder_init = model.decoder_init_proj(atom_emb).unsqueeze(0)  # [1, 1, d_model]

            # 拼接 decoder 初始 token 与当前 tgt_emb
            decoder_input = torch.cat([decoder_init, tgt_emb], dim=0)  # [1+tgt_len, 1, d_model]

            # 生成自回归 mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.size(0)).to(device)

            # Decoder 计算输出
            decoder_output = model.decoder(decoder_input, memory, tgt_mask=tgt_mask)
            # 去掉 decoder 初始 token
            decoder_output = decoder_output[1:, 0, :]  # [tgt_len, d_model]

            # 取最后一个位置的输出，得到 logits 分布
            logits = model.output_linear(decoder_output[-1, :])  # [vocab_size]

            # -------------------------------
            # 根据 additional_info 限制生成原子数量
            # -------------------------------
            # 当前生成序列（去除 <SOS>）
            current_tokens = input_tgt.squeeze(0).tolist()[1:]
            # 统计各原子数量（注意此处假设 'C','N','O','F' 在词表中对应唯一 token）
            count_C = current_tokens.count(char2idx['C'])
            count_N = current_tokens.count(char2idx['N'])
            count_O = current_tokens.count(char2idx['O'])
            count_F = current_tokens.count(char2idx['F'])

            # 如果达到数量限制，则将对应 atom 的 logits 设为 -∞
            if count_C >= additional_info[0]:
                logits[char2idx['C']] = -float('inf')
            if count_N >= additional_info[1]:
                logits[char2idx['N']] = -float('inf')
            if count_O >= additional_info[2]:
                logits[char2idx['O']] = -float('inf')
            if count_F >= additional_info[3]:
                logits[char2idx['F']] = -float('inf')

            # 选择下一个 token（采样策略增加随机性）
            # 使用 temperature sampling
            temperature = 1.0
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).unsqueeze(0)  # [1, 1]

            # 拼接到输入序列中
            input_tgt = torch.cat([input_tgt, next_token], dim=1)

            # 如果生成 <EOS>，则提前结束
            if next_token.item() == char2idx['<EOS>']:
                break

        # 返回生成的 token 序列（去除 <SOS>）
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


def inference(input_smiles, atom_types, additional_info, model, device, max_seq_length, char2idx, idx2char, mask_prob=0.15):
    """
    推理函数：
      - 将 SMILES 转为 token 索引序列
      - 随机 mask 掉部分 token（作为 encoder 输入）
      - 计算 atom type 特征
      - 利用自回归生成完整 SMILES 序列
    返回：
      - generated_smiles: 模型生成的 SMILES 字符串
      - src_indices: mask 后的 token 索引序列
      - true_indices: 原始完整 token 索引序列
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



def compute_additional_info(smiles):
    """
    计算给定 SMILES 分子中 C、N、O、F 四种原子的个数，
    返回一个列表，顺序为 [C, N, O, F]。
    若 SMILES 无法解析，则返回 [0, 0, 0, 0]。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0, 0, 0, 0]
    
    count_C = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    count_N = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
    count_O = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
    count_F = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
    
    return [count_C, count_N, count_O, count_F]




def compute_atom_count(smiles):
    """
    计算 SMILES 字符串中 C, N, O, F 的数量
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # 如果 RDKit 解析失败，直接使用字符串方法统计 C、N、O、F 的数量
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



# -------------------------------
# 5. 推理主函数
# -------------------------------
if __name__ == '__main__':
    
    infer_one_sample = False  # 是否进行单个样本推理，否则进行批量推理
    
    # 若有 GPU 则使用 GPU，否则使用 CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 超参数设置（与训练时保持一致）
    max_seq_length = 400
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1

    # 实例化模型，并加载预训练权重
    model = MoleculePretrainingModel(
        vocab_size, atom_type_dim, d_model=d_model, nhead=nhead,
        num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward, dropout=dropout
    )
    model = model.to(device)

    # 加载预训练模型权重
    #checkpoint_path = "pre_large_weights/molecule_pretraining_model_epoch7.pth"
    # checkpoint_path = "finetuned_molecule_pretraining_model_epoch10.pth"
    checkpoint_path = "/data4/linkaiqing/sm_pretrained/repair_old_finetuned_molecule_pretraining_model_epoch10.pth"
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
        input_file = "/data4/linkaiqing/sm_pretrained/fangyang/gp/csv/corr_draw/temperature_sampling_results.csv"
        output_file = "/data4/linkaiqing/sm_pretrained/fangyang/gp/csv/corr_draw/inference_topN_candidates.csv"
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


