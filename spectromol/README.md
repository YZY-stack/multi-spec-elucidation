# SpectroMol: Multi-Modal Spectral Data to Molecular Structure Prediction

SpectroMol is a deep learning framework for molecular structure elucidation from multi-modal spectral data. It uses transformer-based architecture to predict SMILES molecular representations from various spectroscopic inputs including IR, UV-Vis, NMR, and mass spectrometry data.

## Overview

This framework tackles the challenging problem of determining molecular structures from spectroscopic evidence - a task traditionally requiring expert knowledge and manual interpretation. SpectroMol automates this process using deep learning to learn the complex relationships between spectral signatures and molecular structures.

## Features

- **Multi-Modal Input Processing**: Handles diverse spectroscopic data types:
  - Infrared (IR) spectroscopy
  - UV-Visible spectroscopy  
  - Carbon-13 NMR (1D and 2D)
  - Proton NMR (1D, HSQC, COSY, J-coupling)
  - Fluorine, Nitrogen, and Oxygen NMR
  - High-resolution mass spectrometry

- **Advanced Architecture**: 
  - Custom transformer encoder for spectral feature processing
  - SMILES decoder for molecular structure generation
  - Multi-task learning with auxiliary chemical property prediction
  - Attention mechanisms for interpretable predictions

- **Comprehensive Evaluation**: Multiple metrics including:
  - BLEU scores for sequence similarity
  - Molecular fingerprint similarity (MACCS, RDKit, Morgan)
  - Chemical validity and exact match accuracy
  - Levenshtein distance and cosine similarity

## Project Structure

```
spectromol/
├── __init__.py                          # Package initialization
├── dataset.py                           # Dataset classes and data loading utilities
├── model.py                            # Core model architecture definitions
├── train_all.py                        # Main training script with data preprocessing
├── train_at_new.py                     # Advanced training with auxiliary tasks
├── inference.py                        # Model inference and evaluation
├── inference_drug.py                   # Drug-specific inference pipeline
├── inference_temperature_sampling.py   # Temperature-based sampling inference
├── analysis/                           # Analysis and evaluation tools
│   ├── inference_robustness.py        # Robustness analysis
│   └── inference_tsne.py              # t-SNE visualization
└── tmp_utils/                          # Utility functions and notebooks
    ├── metrics.py                      # Evaluation metrics
    ├── prepare_spe_data_new.ipynb     # Data preparation notebook
    ├── prepare_spe_data.ipynb         # Legacy data preparation
    └── process_HNMR.ipynb             # H-NMR processing notebook
```

## Core Components

### 1. Data Processing (`dataset.py`)
- **SpectraDataset**: Custom PyTorch dataset for multi-modal spectral data
- **SMILES Tokenization**: Converts molecular SMILES strings to token sequences
- **Feature Normalization**: Standardizes spectral data across different modalities
- **Auxiliary Tasks**: Handles molecular property prediction tasks

### 2. Model Architecture (`model.py`)
- **AtomPredictionModel**: Main model combining spectral analysis and SMILES generation
- **Custom Transformer Layers**: Specialized attention mechanisms for spectral data
- **Multi-Task Heads**: Separate prediction heads for auxiliary chemical properties
- **Tokenizer**: Converts spectral data into transformer-compatible tokens

### 3. Training Pipeline (`train_all.py`, `train_at_new.py`)
- **Multi-Modal Training**: Simultaneous learning from all spectral modalities
- **Auxiliary Task Learning**: Joint optimization with chemical property prediction
- **Advanced Loss Functions**: Semantic supervision and functional group penalties
- **Data Splitting**: Scaffold-based molecular splits for realistic evaluation

### 4. Inference and Evaluation (`inference.py`, `inference_*.py`)
- **Greedy Decoding**: Fast molecular structure generation
- **Temperature Sampling**: Diverse structure generation with controlled randomness
- **Comprehensive Metrics**: Chemical validity, similarity, and accuracy measures
- **Robustness Analysis**: Model performance under various conditions

## Quick Start

### Prerequisites
```bash
# Core dependencies
torch>=1.9.0
torchvision
numpy
pandas
scikit-learn

# Chemistry libraries
rdkit-pypi
nltk

# Visualization and analysis
matplotlib
seaborn
tqdm

# Text processing
Levenshtein
```

### Data Preparation
1. **Spectral Data**: Prepare CSV files for each spectral modality
   - IR: `ir_82.csv` (82 features)
   - UV: `uv.csv` (10 features)
   - C-NMR: `1d_cnmr_dept.csv`, `2d_cnmr_ina_chsqc.csv`
   - H-NMR: `1d_hnmr.csv`, `2d_hhsqc.csv`, `2d_hcosy.csv`
   - X-NMR: `1d_fnmr.csv`, `1d_nnmr.csv`, `1d_onmr.csv`
   - Mass Spec: `ms.csv`

2. **Molecular Data**: 
   - SMILES: `smiles.csv`
   - Auxiliary tasks: `aligned_smiles_id_aux_task.csv`

3. **Dataset Splits**: Train/validation/test CSV files with molecular scaffolds

### Training
```python
from spectromol import *

# Load and preprocess data
# (Data loading code from train_all.py)

# Initialize model
model = AtomPredictionModel(
    vocab_size=vocab_size,
    count_tasks_classes=count_task_classes,
    binary_tasks=binary_tasks
)

# Train model
train(
    model=model,
    smiles_loss_fn=criterion,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=1000,
    save_dir='./model_weights'
)
```

### Inference
```python
# Load trained model
model.load_state_dict(torch.load('model_weights/best_model.pth'))

# Predict molecular structures
predicted_smiles = predict_greedy(
    model, ir_data, uv_data, cnmr_data, hnmr_data, mass_data,
    char2idx, idx2char, max_seq_length=100
)

# Evaluate predictions
metrics = evaluate(
    model, test_dataloader, char2idx, idx2char
)
```

## Model Performance

The model is evaluated on multiple metrics:
- **Validity**: Percentage of chemically valid SMILES generated
- **Exact Match**: Exact molecular structure matches
- **BLEU Score**: Sequence similarity measure
- **Fingerprint Similarity**: Chemical similarity using molecular fingerprints
- **Auxiliary Tasks**: Accuracy on molecular property prediction

## Key Features Explained

### Multi-Modal Learning
SpectroMol processes multiple types of spectroscopic data simultaneously, learning complementary information from each modality. This approach mimics how chemists use multiple analytical techniques for structure determination.

### Auxiliary Task Learning
The model jointly learns to predict molecular properties (ring counts, functional groups, etc.) alongside SMILES generation. This multi-task approach improves the model's chemical understanding and prediction accuracy.

### Attention Mechanisms
Custom attention layers allow the model to focus on relevant spectral features when generating specific parts of the molecular structure, providing interpretability.

### Robust Evaluation
Comprehensive evaluation metrics ensure the model generates chemically meaningful and accurate molecular structures, not just syntactically correct SMILES.

## Data Requirements

### Spectral Data Format
- **CSV files** with peak information for each spectral modality
- **Normalized features** (handled automatically during preprocessing)
- **Consistent sample ordering** across all spectral files

### Molecular Data Format
- **SMILES strings** in canonical form
- **Auxiliary task labels** for molecular properties
- **Dataset splits** maintaining molecular scaffold diversity

## Advanced Features

### Temperature Sampling
```python
# Generate diverse molecular candidates
candidates = inference_temperature_sampling(
    model, spectral_data, temperature=0.8, num_samples=10
)
```

### Robustness Analysis
```python
# Evaluate model robustness to noise
robustness_metrics = analyze_robustness(
    model, test_data, noise_levels=[0.1, 0.2, 0.3]
)
```

### Visualization
```python
# t-SNE visualization of learned representations
plot_tsne_analysis(model, spectral_data, molecular_labels)
```

## Contributing

When contributing to SpectroMol:
1. Follow the existing code structure and documentation style
2. Add comprehensive tests for new features
3. Update this README for significant changes
4. Ensure chemical validity of any new evaluation metrics

## License

[License information to be added]

## Support

For questions, issues, or contributions, please contact [contact information] or open an issue in the repository.

---

*SpectroMol bridges the gap between analytical chemistry and artificial intelligence, enabling automated molecular structure elucidation from spectroscopic data.*
