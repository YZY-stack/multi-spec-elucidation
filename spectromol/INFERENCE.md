# SpectroMol Inference

## Quick Start

Run inference with all spectral modalities on the test set:

```bash
cd /path/to/multi-spec-elucidation
python spectromol/infer_cli.py --modality all --split test
```

## CLI Usage

```bash
python spectromol/infer_cli.py [OPTIONS]
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--modality` | `all` | Spectral modality: `all`, `ir`, `1dnmr`, `2dnmr` |
| `--model_path` | `./spectromol/checkpoints/0806_ft.pth` | Path to model checkpoint |
| `--data_dir` | `./spectromol/qm9_all_raw_spe/` | Path to spectral data directory |
| `--split` | `test` | Dataset split: `val` or `test` |
| `--batch_size` | `128` | Batch size for inference |
| `--beam_size` | `5` | Beam search width |
| `--max_samples` | `None` | Limit number of samples (for quick testing) |
| `--output_csv` | `results_<modality>_<split>.csv` | Output file path |

### Modality Groups

| Modality | Retained Data | Zeroed Data |
|----------|--------------|-------------|
| `all` | IR + UV + all NMR + MS | None |
| `ir` | IR + MS | UV + all NMR |
| `1dnmr` | 1D NMR (H/C/F/N/O) + MS | IR + UV + 2D NMR |
| `2dnmr` | All NMR (1D + 2D) + MS | IR + UV |

Note: MS/atom_types (molecular formula: C/N/O/F counts) is always retained.

## Examples

```bash
# Full evaluation on test set with all modalities
python spectromol/infer_cli.py --modality all --split test

# Quick test with 20 samples, IR-only
python spectromol/infer_cli.py --modality ir --max_samples 20 --beam_size 3

# 1D NMR only on validation set
python spectromol/infer_cli.py --modality 1dnmr --split val
```

## Expected Results (test set, 20 samples)

| Modality | Top-1 Acc | Validity | BLEU | Morgan Sim |
|----------|-----------|----------|------|------------|
| all | 0.90 | 1.00 | 0.96 | 0.95 |
| ir | 0.00 | 0.25 | 0.21 | 0.09 |

## Checkpoint

The default checkpoint is located at `spectromol/checkpoints/0806_ft.pth`.

## Data

Spectral data should be placed in `spectromol/qm9_all_raw_spe/` with the following CSV files:
- `ir_82.csv`, `uv.csv`
- `1d_cnmr_dept.csv`, `2d_cnmr_ina_chsqc.csv`
- `1d_hnmr.csv`, `2d_hhsqc.csv`, `2d_hcosy.csv`
- `1d_fnmr.csv`, `1d_nnmr.csv`, `1d_onmr.csv`
- `ms.csv`, `smiles.csv`
- `aligned_smiles_id_aux_task.csv`
- `dataset_split/scaffold/{train,val,test}.csv`
