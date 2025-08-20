"""
SpectroMol Dataset Module

This module defines the dataset class for handling multi-modal spectral data
and SMILES molecular representations for deep learning tasks.

Author: SpectroMol Team
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import re

class SpectraDataset(Dataset):
    """
    Dataset class for handling multi-modal spectral data and SMILES molecular representations.
    
    This dataset handles five types of spectral data (IR, UV, C-NMR, H-NMR, and high-resolution 
    mass spectra) along with SMILES strings and auxiliary task data for molecular property prediction.
    
    Args:
        ir_spectra (array-like): IR spectral data
        uv_spectra (array-like): UV spectral data  
        c_spectra (array-like): Carbon-13 NMR spectral data
        h_spectra (array-like): Proton NMR spectral data
        high_mass_spectra (array-like): High-resolution mass spectral data
        smiles_list (list): List of SMILES string representations
        auxiliary_data (DataFrame): DataFrame containing auxiliary task targets
        char2idx (dict): Character to index mapping for SMILES tokenization
        max_seq_length (int): Maximum sequence length for SMILES encoding
        count_tasks (list): List of count-based auxiliary task names
        binary_tasks (list): List of binary auxiliary task names
        atom_types_list (list, optional): List of atom type features for each molecule
    """
    def __init__(
        self, ir_spectra, uv_spectra, c_spectra, h_spectra,
        high_mass_spectra, smiles_list, auxiliary_data,
        char2idx, max_seq_length, count_tasks, binary_tasks,
        atom_types_list=None
    ):
        # Store spectral data
        self.ir_spectra = ir_spectra
        self.uv_spectra = uv_spectra
        self.c_spectra = c_spectra
        self.h_spectra = h_spectra
        self.high_mass_spectra = high_mass_spectra
        
        # Store molecular representations and metadata
        self.smiles_list = smiles_list  # List of SMILES strings
        self.auxiliary_data = auxiliary_data  # DataFrame containing auxiliary tasks
        self.char2idx = char2idx
        self.max_seq_length = max_seq_length
        self.atom_types_list = atom_types_list  # List of atom types for each molecule

        # Task definitions
        self.count_tasks = count_tasks
        self.binary_tasks = binary_tasks

        # Validate and process auxiliary data
        if auxiliary_data is not None:
            required_columns = self.count_tasks + self.binary_tasks
            if not all(col in self.auxiliary_data.columns for col in required_columns):
                raise ValueError("Auxiliary data does not contain all required columns.")

            # Calculate number of classes for each count task
            self.count_task_classes = {}
            for task in self.count_tasks:
                max_value = self.auxiliary_data[task].max()
                self.count_task_classes[task] = int(max_value) + 1  # Number of classes

            # Extract and process auxiliary targets
            auxiliary_targets = self.auxiliary_data[required_columns].values  # Shape: [num_samples, num_auxiliary_tasks]
            
            # Convert count tasks to integer class indices
            for idx, task in enumerate(self.count_tasks):
                auxiliary_targets[:, idx] = auxiliary_targets[:, idx].astype(int)
                
            # Ensure binary tasks are integers (0 or 1)
            for idx, task in enumerate(self.binary_tasks):
                task_idx = len(self.count_tasks) + idx
                auxiliary_targets[:, task_idx] = auxiliary_targets[:, task_idx].astype(int)

            self.auxiliary_targets = auxiliary_targets  # Numpy array
        else:
            self.auxiliary_targets = atom_types_list

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.smiles_list)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: A tuple containing:
                - ir (torch.Tensor): IR spectral data
                - uv (torch.Tensor): UV spectral data  
                - c_spec (torch.Tensor): Carbon-13 NMR spectral data
                - h_spec (torch.Tensor): Proton NMR spectral data
                - high_mass (torch.Tensor): High-resolution mass spectral data
                - smiles_indices (torch.Tensor): Tokenized SMILES sequence
                - auxiliary_targets (torch.Tensor): Auxiliary task targets
                - atom_types (torch.Tensor): Atom type features
        """
        # Get spectra data
        ir = torch.tensor(self.ir_spectra[idx], dtype=torch.float32)
        uv = torch.tensor(self.uv_spectra[idx], dtype=torch.float32)
        c_spec = torch.tensor(self.c_spectra[idx], dtype=torch.float32)
        h_spec = torch.tensor(self.h_spectra[idx], dtype=torch.float32)
        high_mass = torch.tensor(self.high_mass_spectra[idx], dtype=torch.float32)

        # Get SMILES sequence and validate format
        sm = self.smiles_list[idx]
        if isinstance(sm, list):
            smiles = sm[0]  # Extract string from list format
        elif isinstance(sm, str):
            smiles = sm
        else:
            raise ValueError(f"Invalid SMILES format at index {idx}: {type(sm)}")
            
        # Convert SMILES to token indices
        smiles_indices = smiles_to_indices(smiles, self.char2idx, self.max_seq_length)
        smiles_indices = torch.tensor(smiles_indices, dtype=torch.long)

        # Get auxiliary targets
        auxiliary_targets = self.auxiliary_targets[idx]
        auxiliary_targets = torch.tensor(auxiliary_targets, dtype=torch.float32)  # Shape: [num_auxiliary_tasks]

        # Get atom types
        atom_types = torch.tensor(self.atom_types_list[idx], dtype=torch.float32)

        return ir, uv, c_spec, h_spec, high_mass, smiles_indices, auxiliary_targets, atom_types




import re
def tokenize_smiles(smiles):
    """
    Tokenize a SMILES string into individual chemical tokens.
    
    This function uses regex pattern matching to identify and extract
    chemical tokens including atoms, bonds, and special symbols.
    
    Args:
        smiles (str): SMILES string representation of a molecule
        
    Returns:
        list: List of chemical tokens extracted from the SMILES string
    """
    pattern = r'(\[.*?\]|Br|Cl|Si|se|c|n|o|s|[B-DF-Zb-df-z]|\%\d\d|.)'
    tokens = re.findall(pattern, smiles)
    return tokens

def smiles_to_indices(smiles, char2idx, max_length):
    """
    Convert a SMILES string to a sequence of token indices.
    
    This function tokenizes the SMILES string and converts each token
    to its corresponding index in the vocabulary, adding special tokens
    for sequence start/end and padding to a fixed length.
    
    Args:
        smiles (str): SMILES string representation of a molecule
        char2idx (dict): Dictionary mapping tokens to indices
        max_length (int): Maximum sequence length for padding/truncation
        
    Returns:
        list: List of token indices representing the SMILES sequence
    """
    # Tokenize SMILES and build index sequence
    tokens = tokenize_smiles(smiles)
    indices = [char2idx.get('<SOS>')]  # Start of sequence token
    
    # Convert each token to its index
    for token in tokens:
        if token in char2idx:
            indices.append(char2idx[token])
        else:
            indices.append(char2idx.get('<UNK>'))  # Unknown token
            
    indices.append(char2idx.get('<EOS>'))  # End of sequence token
    
    # Apply padding or truncation to reach max_length
    if len(indices) < max_length:
        indices += [char2idx.get('<PAD>')] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    return indices
