# MS Mol2Mol Module

This module implements a transformer-based molecular representation learning system with mass spectrometry integration for molecular structure elucidation tasks.

## Overview

The MS Mol2Mol module provides:
- **Molecular Pre-training**: Self-supervised learning on large molecular datasets
- **Fine-tuning**: Task-specific adaptation for structure-property relationships
- **Inference**: High-performance molecular generation and property prediction
- **Mass Spectrometry Integration**: Leveraging MS data for enhanced molecular understanding

## Architecture

### Core Components

1. **Transformer Model** (`model.py`)
   - Multi-head attention mechanisms for molecular sequence understanding
   - Encoder-decoder architecture with positional encoding
   - Atomic feature integration for enhanced molecular representation

2. **Pre-training Framework** (`pretrain.py`)
   - Self-supervised masked language modeling for molecules
   - Large-scale molecular dataset processing
   - Distributed training support with gradient accumulation

3. **Fine-tuning System** (`finetune.py`)
   - Task-specific adaptation capabilities
   - SMILES corruption and reconstruction training
   - Advanced data augmentation strategies

4. **Inference Engine** (`infer.py`)
   - Autoregressive molecular generation
   - Beam search and sampling strategies
   - Constraint-guided generation

5. **Utilities** (`utils.py`)
   - SMILES tokenization and processing
   - Molecular feature computation
   - Data loading and preprocessing utilities

## Key Features

### Advanced Tokenization
- Multi-character element support (Br, Cl, etc.)
- Bracketed atom handling for charged species
- Ring closure and bond type recognition
- Special token support for model training

### Robust Data Processing
- **SMILES Corruption**: Multiple perturbation strategies for robust training
  - Token-level masking, deletion, and replacement
  - Atom swapping and functional group modifications
  - Ring structure perturbations
- **Atomic Features**: Comprehensive molecular descriptors
  - Degree of unsaturation (DBE) calculation
  - Precise molecular weight computation
  - Element count normalization

### Training Optimizations
- **Distributed Training**: Multi-GPU support with DDP
- **Dynamic Learning Rate**: Cosine annealing with warmup
- **Gradient Clipping**: Adaptive gradient normalization
- **NaN Detection**: Robust error handling and recovery

## Usage

### Pre-training

```python
# Initialize model and start pre-training
python pretrain.py --data_path /path/to/molecular/data \
                   --epochs 100 \
                   --batch_size 256 \
                   --learning_rate 1e-4
```

### Fine-tuning

```python
# Fine-tune on specific molecular tasks
python finetune.py --pretrained_model /path/to/checkpoint \
                   --task_data /path/to/task/data \
                   --epochs 50 \
                   --batch_size 128
```

### Inference

```python
# Generate molecular structures
python infer.py --model_path /path/to/trained/model \
                --input_smiles "CCO" \
                --output_file results.csv
```

## Model Architecture Details

### Encoder-Decoder Transformer
- **Embedding Dimension**: 512
- **Attention Heads**: 8
- **Encoder Layers**: 6
- **Decoder Layers**: 6
- **Feed-forward Dimension**: 2048

### Atomic Feature Integration
- Atomic weight-based molecular descriptors
- Log-normalized element counts
- DBE (Degree of Unsaturation) calculation
- NaN-resistant feature computation

### Vocabulary
- Comprehensive SMILES token coverage
- Special tokens for model control
- Extended element support
- Ring closure handling

## Data Requirements

### Input Formats
- **CSV Files**: SMILES column required
- **Molecular Properties**: Optional atomic features
- **Mass Spectra**: MS/MS fragmentation patterns (future integration)

### Preprocessing
- SMILES validation and canonicalization
- Vocabulary coverage checking
- Sequence length normalization
- Batch padding and truncation

## Performance Optimizations

### Memory Efficiency
- Gradient checkpointing for large models
- Mixed precision training support
- Efficient data loading with prefetching

### Training Stability
- Layer normalization for stable gradients
- Residual connections in transformer blocks
- Dropout regularization
- Early stopping based on validation metrics

## Applications

### Molecular Generation
- De novo drug design
- Chemical space exploration
- Structure-based optimization

### Property Prediction
- ADMET property forecasting
- Toxicity assessment
- Synthetic accessibility scoring

### Structure Elucidation
- MS/MS spectrum interpretation
- Unknown compound identification
- Structural confirmation

## Dependencies

- PyTorch ≥ 1.8.0
- RDKit ≥ 2021.03
- pandas ≥ 1.2.0
- numpy ≥ 1.20.0
- tqdm ≥ 4.60.0
- tensorboard (optional, for training monitoring)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines on:
- Code style and formatting
- Testing requirements
- Documentation standards
- Pull request procedures

## Support

For questions and support:
- GitHub Issues: Report bugs and feature requests
- Documentation: Comprehensive API reference available
- Examples: Sample notebooks and scripts provided

---

**Note**: This module is part of the multi-spec-elucidation project, focusing on integrating multiple spectroscopic techniques for enhanced molecular structure determination.
