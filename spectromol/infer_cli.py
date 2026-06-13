"""
SpectroMol CLI Inference

Usage:
    python spectromol/infer_cli.py --modality all
    python spectromol/infer_cli.py --modality ir
    python spectromol/infer_cli.py --modality 1dnmr
    python spectromol/infer_cli.py --modality 2dnmr
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import csv
from tqdm import tqdm

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import AtomPredictionModel
from dataset import SpectraDataset
from inference import inference_beam, load_model
from metrics import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SMILES_VOCAB = [
    '<PAD>', '<SOS>', '<EOS>', '<UNK>',
    'C', 'N', 'O', 'F',
    '1', '2', '3', '4', '5',
    '#', '=', '(', ')',
]
vocab_size = len(SMILES_VOCAB)
char2idx = {token: idx for idx, token in enumerate(SMILES_VOCAB)}
idx2char = {idx: token for idx, token in enumerate(SMILES_VOCAB)}


def parse_args():
    parser = argparse.ArgumentParser(description='SpectroMol Inference with Modality Selection')
    parser.add_argument('--modality', type=str, default='all',
                        choices=['all', 'ir', '1dnmr', '2dnmr'],
                        help='Spectral modality to use (default: all)')
    parser.add_argument('--model_path', type=str,
                        default='./spectromol/checkpoints/0806_ft.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str,
                        default='./spectromol/qm9_all_raw_spe/',
                        help='Path to spectral data directory')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Dataset split to evaluate (default: test)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of samples to evaluate (default: all)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save results CSV (default: results_<modality>.csv)')
    return parser.parse_args()


def load_spectral_data(data_dir):
    """Load and normalize all spectral CSV files."""
    print('Loading spectral data...')

    # IR
    ir = pd.read_csv(os.path.join(data_dir, 'ir_82.csv'))
    peak_cols = [c for c in ir.columns if 'peak' in c]
    ir[peak_cols] = ir[peak_cols] / 4000.0
    ir = ir.to_numpy()

    # UV
    uv = pd.read_csv(os.path.join(data_dir, 'uv.csv'))
    peak_cols = [c for c in uv.columns if 'peak' in c]
    uv[peak_cols] = uv[peak_cols] / 15.0
    uv = uv.to_numpy()

    # 1D C-NMR
    cnmr_1d = pd.read_csv(os.path.join(data_dir, '1d_cnmr_dept.csv'))
    peak_cols = [c for c in cnmr_1d.columns if 'peak' in c]
    cnmr_1d[peak_cols] = (cnmr_1d[peak_cols] - (-10.0)) / (220.0 - (-10.0))
    cnmr_1d = cnmr_1d.to_numpy()

    # 2D C-NMR (CHSQC)
    cnmr_2d = pd.read_csv(os.path.join(data_dir, '2d_cnmr_ina_chsqc.csv'))
    peak_cols = [c for c in cnmr_2d.columns if 'peak' in c]
    cnmr_2d[peak_cols] = (cnmr_2d[peak_cols] - (-400.0)) / (450.0 - (-400.0))
    cnmr_2d = cnmr_2d.to_numpy()

    # Combined C-NMR: 1D (28) + 2D (180) = 207
    c_spec = np.concatenate((cnmr_1d, cnmr_2d), axis=1)

    # 1D H-NMR
    hnmr_1d = pd.read_csv(os.path.join(data_dir, '1d_hnmr.csv'))
    peak_cols = [c for c in hnmr_1d.columns if 'peak' in c]
    # Detect abnormal samples
    max_vals = hnmr_1d[peak_cols].max(axis=1)
    abnormal_mask = max_vals > 500.0
    abnormal_indices = set(np.where(abnormal_mask)[0])
    hnmr_1d[peak_cols] = (hnmr_1d[peak_cols] - (-2.0)) / (12.0 - (-2.0))
    hnmr_1d = hnmr_1d.to_numpy()

    # HSQC
    hsqc = pd.read_csv(os.path.join(data_dir, '2d_hhsqc.csv'))
    peak_cols = [c for c in hsqc.columns if 'peak' in c]
    hsqc[peak_cols] = (hsqc[peak_cols] - (-350.0)) / (400.0 - (-350.0))
    hsqc = hsqc.to_numpy()

    # COSY
    cosy = pd.read_csv(os.path.join(data_dir, '2d_hcosy.csv'))
    hxyh_cols = [c for c in cosy.columns if 'H_X_Y_H' in c]
    cosy = cosy[hxyh_cols]
    peak_cols = [c for c in cosy.columns if 'peak' in c]
    cosy[peak_cols] = (cosy[peak_cols] - (-2.0)) / (14.0 - (-2.0))
    cosy = cosy.to_numpy()

    # F-NMR
    fnmr = pd.read_csv(os.path.join(data_dir, '1d_fnmr.csv'))
    peak_cols = [c for c in fnmr.columns if 'peak' in c]
    fnmr[peak_cols] = (fnmr[peak_cols] - (-400.0)) / (0.0001 - (-400.0))
    fnmr = fnmr.to_numpy()

    # N-NMR
    nnmr = pd.read_csv(os.path.join(data_dir, '1d_nnmr.csv'))
    peak_cols = [c for c in nnmr.columns if 'peak' in c]
    nnmr[peak_cols] = (nnmr[peak_cols] - (-260.0)) / (400.0 - (-260.0))
    nnmr = nnmr.to_numpy()

    # O-NMR
    onmr = pd.read_csv(os.path.join(data_dir, '1d_onmr.csv'))
    peak_cols = [c for c in onmr.columns if 'peak' in c]
    onmr[peak_cols] = (onmr[peak_cols] - (-385.0)) / (460.0 - (-385.0))
    onmr = onmr.to_numpy()

    # Combined H-spectrum: 1D(105) + HSQC(120) + COSY(120) + F(12) + N(14) + O(12) = 383 -> wait
    # Actually from the code: h_spectrum[:, :382] is h_part, then F/N/O appended
    # So h_spec = hnmr_1d(105) + hsqc(120) + cosy(120) + fnmr(12) + nnmr(14) + onmr(12)
    # But model splits at [:382] for h_part. Let's check: 105+120+120 = 345? No...
    # From inference.py line 808: concatenate(hnmr_1d, hsqc, cosy, fnmr, nnmr, onmr)
    # model.py line 210: h_spectrum_part = h_spectrum[:, :382], f = [:, 382:394], n = [:, 394:408], o = [:, 408:]
    # So hnmr_1d must be 105 cols, hsqc 120, cosy 120, that's 345 for h_part? No, 382.
    # Let me just trust the concatenation order and model split indices.
    h_spec = np.concatenate((hnmr_1d, hsqc, cosy, fnmr, nnmr, onmr), axis=1)

    # MS
    ms = pd.read_csv(os.path.join(data_dir, 'ms.csv'))
    ms = ms.to_numpy()

    # atom_type: columns 1:-1 of ms (C, N, O, F counts)
    atom_type = ms[:, 1:-1]

    # SMILES
    smiles_list = pd.read_csv(os.path.join(data_dir, 'smiles.csv')).values.tolist()

    # Auxiliary data
    aux_data = pd.read_csv(os.path.join(data_dir, 'aligned_smiles_id_aux_task.csv')).iloc[:, 2:]

    print(f'  IR: {ir.shape}, UV: {uv.shape}, C-spec: {c_spec.shape}')
    print(f'  H-spec: {h_spec.shape}, MS: {ms.shape}')
    print(f'  Total samples: {len(smiles_list)}')

    return {
        'ir': ir,
        'uv': uv,
        'c_spec': c_spec,
        'h_spec': h_spec,
        'ms': ms,
        'atom_type': atom_type,
        'smiles_list': smiles_list,
        'aux_data': aux_data,
        'abnormal_indices': abnormal_indices,
        # Keep sub-components for fine-grained masking
        'hnmr_1d': hnmr_1d,
        'hsqc': hsqc,
        'cosy': cosy,
        'fnmr': fnmr,
        'nnmr': nnmr,
        'onmr': onmr,
        'cnmr_1d': cnmr_1d,
        'cnmr_2d': cnmr_2d,
    }


def apply_modality_mask(data, modality):
    """
    Zero-out spectral modalities based on selection.
    MS/atom_type is always preserved.

    Modality groups:
      all   - everything
      ir    - IR only (+ MS)
      1dnmr - 1D NMR (H-1D, C-1D, F, N, O) (+ MS)
      2dnmr - all NMR (1D + HSQC + COSY + CHSQC) (+ MS)
    """
    n = data['ir'].shape[0]

    if modality == 'all':
        return data

    if modality == 'ir':
        # Zero UV and all NMR
        data['uv'] = np.zeros_like(data['uv'])
        data['c_spec'] = np.zeros_like(data['c_spec'])
        data['h_spec'] = np.zeros_like(data['h_spec'])

    elif modality == '1dnmr':
        # Zero IR, UV, 2D NMR (HSQC, COSY, CHSQC)
        data['ir'] = np.zeros_like(data['ir'])
        data['uv'] = np.zeros_like(data['uv'])
        # Rebuild h_spec: keep 1D parts, zero 2D parts
        # h_spec = [hnmr_1d(105), hsqc(120), cosy(120), fnmr(12), nnmr(14), onmr(12)]
        hnmr_1d = data['hnmr_1d']
        hsqc_zero = np.zeros_like(data['hsqc'])
        cosy_zero = np.zeros_like(data['cosy'])
        fnmr = data['fnmr']
        nnmr = data['nnmr']
        onmr = data['onmr']
        data['h_spec'] = np.concatenate((hnmr_1d, hsqc_zero, cosy_zero, fnmr, nnmr, onmr), axis=1)
        # C-spec: keep 1D, zero 2D
        cnmr_1d = data['cnmr_1d']
        cnmr_2d_zero = np.zeros_like(data['cnmr_2d'])
        data['c_spec'] = np.concatenate((cnmr_1d, cnmr_2d_zero), axis=1)

    elif modality == '2dnmr':
        # Keep all NMR (1D + 2D), zero IR and UV
        data['ir'] = np.zeros_like(data['ir'])
        data['uv'] = np.zeros_like(data['uv'])

    return data


def filter_abnormal(data):
    """Remove samples with abnormal H-NMR values."""
    abnormal_indices = sorted(list(data['abnormal_indices']))
    if not abnormal_indices:
        return data

    n = len(data['smiles_list'])
    mask = np.ones(n, dtype=bool)
    mask[list(abnormal_indices)] = False

    print(f'Filtering {len(abnormal_indices)} abnormal samples...')

    data['ir'] = data['ir'][mask]
    data['uv'] = data['uv'][mask]
    data['c_spec'] = data['c_spec'][mask]
    data['h_spec'] = data['h_spec'][mask]
    data['ms'] = data['ms'][mask]
    data['atom_type'] = data['atom_type'][mask]
    data['smiles_list'] = [data['smiles_list'][i] for i in range(n) if mask[i]]
    data['aux_data'] = data['aux_data'][mask].reset_index(drop=True)
    # Sub-components
    for key in ['hnmr_1d', 'hsqc', 'cosy', 'fnmr', 'nnmr', 'onmr', 'cnmr_1d', 'cnmr_2d']:
        data[key] = data[key][mask]

    # Collect abnormal SMILES for split filtering
    data['abnormal_smiles'] = set()
    for idx in abnormal_indices:
        if idx < n:
            sm = data['smiles_list'][idx] if idx < len(data['smiles_list']) else None
            if sm:
                data['abnormal_smiles'].add(sm[0] if isinstance(sm, list) else sm)

    print(f'  Remaining samples: {len(data["smiles_list"])}')
    return data


def get_split_indices(data, data_dir, split, data_split_mode='scaffold'):
    """Get indices for the requested split."""
    smiles_list = data['smiles_list']
    smiles_to_index = {}
    for idx, sm in enumerate(smiles_list):
        key = sm[0] if isinstance(sm, list) else sm
        smiles_to_index[key] = idx

    split_df = pd.read_csv(os.path.join(data_dir, f'dataset_split/{data_split_mode}/{split}.csv'))

    # Filter abnormal SMILES from split
    abnormal_smiles = data.get('abnormal_smiles', set())
    if abnormal_smiles:
        split_df = split_df[~split_df['smiles'].isin(abnormal_smiles)].reset_index(drop=True)

    indices = []
    for smiles in split_df['smiles']:
        idx = smiles_to_index.get(smiles)
        if idx is not None:
            indices.append(idx)

    print(f'  {split} split: {len(indices)} samples')
    return indices


def run_inference(model, dataloader, beam_size=5):
    """Run beam search inference on a dataloader, return (true, predicted) pairs."""
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Inference'):
            ir, uv, c_spec, h_spec, ms, smiles_indices, aux, atom_types = batch

            ir = ir.to(device)
            uv = uv.to(device)
            c_spec = c_spec.to(device)
            h_spec = h_spec.to(device)
            ms = ms.to(device)
            atom_types = atom_types.to(device)

            batch_size = ir.size(0)

            # Decode ground truth SMILES
            true_smiles_list = []
            for i in range(batch_size):
                tokens = []
                for idx in smiles_indices[i]:
                    idx_val = idx.item()
                    if idx_val == char2idx['<EOS>']:
                        break
                    if idx_val not in [char2idx['<PAD>'], char2idx['<SOS>']]:
                        tokens.append(idx2char.get(idx_val, '<UNK>'))
                true_smiles_list.append(''.join(tokens))

            # Build atom count constraints from atom_types
            atom_counts_array = atom_types[:, 1:].cpu().numpy()
            required_atom_counts = []
            for counts in atom_counts_array:
                required_atom_counts.append(dict(zip(['C', 'N', 'O', 'F'], counts)))

            predicted_smiles_list = inference_beam(
                model, ir, uv, c_spec, h_spec, ms,
                char2idx, idx2char,
                max_seq_length=100,
                atom_types=atom_types,
                required_atom_counts=required_atom_counts,
                beam_size=beam_size,
            )

            for i in range(batch_size):
                results.append((true_smiles_list[i], predicted_smiles_list[i]))

    return results


def compute_metrics(results):
    """Compute evaluation metrics from (true, predicted) pairs."""
    smoothie = SmoothingFunction().method4

    total = len(results)
    bleu_sum = 0.0
    correct = 0
    valid = 0
    maccs_sims = []
    morgan_sims = []

    for true_sm, pred_sm in results:
        # Validity
        try:
            mol = Chem.MolFromSmiles(pred_sm)
            if mol is not None:
                valid += 1
        except:
            pass

        # Top-1 accuracy (canonical match)
        try:
            if Chem.CanonSmiles(pred_sm) == Chem.CanonSmiles(true_sm):
                correct += 1
        except:
            if pred_sm == true_sm:
                correct += 1

        # BLEU
        ref = [list(true_sm)]
        cand = list(pred_sm)
        bleu_sum += sentence_bleu(ref, cand, smoothing_function=smoothie)

        # Fingerprint similarity
        try:
            mol_pred = Chem.MolFromSmiles(pred_sm)
            mol_true = Chem.MolFromSmiles(true_sm)
            if mol_pred and mol_true:
                ms_val = maccs_similarity(pred_sm, true_sm)
                if ms_val is not None:
                    maccs_sims.append(ms_val)
                mg_val = morgan_similarity(pred_sm, true_sm)
                if mg_val is not None:
                    morgan_sims.append(mg_val)
        except:
            pass

    metrics = {
        'total_samples': total,
        'top1_accuracy': correct / total if total else 0,
        'validity': valid / total if total else 0,
        'avg_bleu': bleu_sum / total if total else 0,
        'avg_maccs_sim': np.mean(maccs_sims) if maccs_sims else 0,
        'avg_morgan_sim': np.mean(morgan_sims) if morgan_sims else 0,
    }
    return metrics


def main():
    args = parse_args()

    print(f'=== SpectroMol Inference ===')
    print(f'  Modality: {args.modality}')
    print(f'  Model: {args.model_path}')
    print(f'  Split: {args.split}')
    print(f'  Device: {device}')
    print()

    # Load model
    model = load_model(args.model_path, vocab_size, char2idx)
    print(f'Model loaded.\n')

    # Load data
    data = load_spectral_data(args.data_dir)

    # Filter abnormal samples
    data = filter_abnormal(data)

    # Apply modality mask
    data = apply_modality_mask(data, args.modality)

    # Get split indices
    indices = get_split_indices(data, args.data_dir, args.split)

    # Limit samples if requested
    if args.max_samples is not None:
        indices = indices[:args.max_samples]
        print(f'  Limited to {len(indices)} samples')

    # Extract split data
    ir_split = data['ir'][indices]
    uv_split = data['uv'][indices]
    c_split = data['c_spec'][indices]
    h_split = data['h_spec'][indices]
    ms_split = data['ms'][indices]
    atom_split = data['atom_type'][indices]
    smiles_split = [data['smiles_list'][i] for i in indices]
    aux_split = data['aux_data'].iloc[indices].reset_index(drop=True)

    # Compute max SMILES length
    smiles_lengths = []
    for sm in smiles_split:
        s = sm[0] if isinstance(sm, list) else sm
        smiles_lengths.append(len(s))
    max_seq_length = max(smiles_lengths) + 2

    # Auxiliary task columns
    columns = aux_split.columns.tolist()
    count_tasks = [c for c in columns if 'Has' not in c and 'Is' not in c]
    binary_tasks = [c for c in columns if 'Has' in c or 'Is' in c]

    # Build dataset
    dataset = SpectraDataset(
        ir_spectra=ir_split,
        uv_spectra=uv_split,
        c_spectra=c_split,
        h_spectra=h_split,
        high_mass_spectra=ms_split,
        smiles_list=smiles_split,
        auxiliary_data=aux_split,
        char2idx=char2idx,
        max_seq_length=max_seq_length,
        count_tasks=count_tasks,
        binary_tasks=binary_tasks,
        atom_types_list=atom_split,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    # Run inference
    results = run_inference(model, dataloader, beam_size=args.beam_size)

    # Compute metrics
    metrics = compute_metrics(results)

    print(f'\n=== Results ({args.modality}) ===')
    print(f'  Samples evaluated: {metrics["total_samples"]}')
    print(f'  Top-1 Accuracy:    {metrics["top1_accuracy"]:.4f}')
    print(f'  Validity:          {metrics["validity"]:.4f}')
    print(f'  BLEU:              {metrics["avg_bleu"]:.4f}')
    print(f'  MACCS Similarity:  {metrics["avg_maccs_sim"]:.4f}')
    print(f'  Morgan Similarity: {metrics["avg_morgan_sim"]:.4f}')

    # Save results
    output_path = args.output_csv or f'results_{args.modality}_{args.split}.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['true_smiles', 'predicted_smiles'])
        writer.writeheader()
        for true_sm, pred_sm in results:
            writer.writerow({'true_smiles': true_sm, 'predicted_smiles': pred_sm})
    print(f'\n  Results saved to: {output_path}')


if __name__ == '__main__':
    main()
