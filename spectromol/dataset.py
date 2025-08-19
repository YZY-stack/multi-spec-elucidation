import torch
import torch.utils.data as data
import random


import torch
from torch.utils.data import Dataset
import pandas as pd

class SpectraDataset(Dataset):
    def __init__(
        self, ir_spectra, uv_spectra, c_spectra, h_spectra,
        high_mass_spectra, smiles_list, auxiliary_data,
        char2idx, max_seq_length, count_tasks, binary_tasks,
        atom_types_list=None
    ):
        self.ir_spectra = ir_spectra
        self.uv_spectra = uv_spectra
        self.c_spectra = c_spectra
        self.h_spectra = h_spectra
        self.high_mass_spectra = high_mass_spectra
        self.smiles_list = smiles_list  # List of SMILES strings
        self.auxiliary_data = auxiliary_data  # DataFrame containing auxiliary tasks
        self.char2idx = char2idx
        self.max_seq_length = max_seq_length
        self.atom_types_list = atom_types_list  # List of atom types for each molecule

        # Tasks
        self.count_tasks = count_tasks
        self.binary_tasks = binary_tasks

        # Ensure that the auxiliary data contains the necessary columns
        if auxiliary_data is not None:
            required_columns = self.count_tasks + self.binary_tasks
            if not all(col in self.auxiliary_data.columns for col in required_columns):
                raise ValueError("Auxiliary data does not contain all required columns.")

            # Calculate number of classes for each count task
            self.count_task_classes = {}
            for task in self.count_tasks:
                max_value = self.auxiliary_data[task].max()
                self.count_task_classes[task] = int(max_value) + 1  # Number of classes

            # Extract auxiliary targets
            auxiliary_targets = self.auxiliary_data[required_columns].values  # Shape: [num_samples, num_auxiliary_tasks]
            # Convert count tasks to integer class indices
            for idx, task in enumerate(self.count_tasks):
                auxiliary_targets[:, idx] = auxiliary_targets[:, idx].astype(int)
            # Ensure binary tasks are integers (0 or 1)
            for idx, task in enumerate(self.binary_tasks):
                auxiliary_targets[:, len(self.count_tasks) + idx] = auxiliary_targets[:, len(self.count_tasks) + idx].astype(int)

            self.auxiliary_targets = auxiliary_targets  # Numpy array
        else:
            self.auxiliary_targets = atom_types_list

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        # Get spectra data
        ir = torch.tensor(self.ir_spectra[idx], dtype=torch.float32)
        uv = torch.tensor(self.uv_spectra[idx], dtype=torch.float32)
        c_spec = torch.tensor(self.c_spectra[idx], dtype=torch.float32)
        h_spec = torch.tensor(self.h_spectra[idx], dtype=torch.float32)
        high_mass = torch.tensor(self.high_mass_spectra[idx], dtype=torch.float32)

        # Get SMILES sequence
        sm = self.smiles_list[idx]
        if type(sm) == list:
            smiles = sm[0]  # Assuming smiles_list is a list of strings
        elif type(sm) == str:
            smiles = sm
        else:
            raise ValueError
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
    pattern = r'(\[.*?\]|Br|Cl|Si|se|c|n|o|s|[B-DF-Zb-df-z]|\%\d\d|.)'
    tokens = re.findall(pattern, smiles)
    return tokens

def smiles_to_indices(smiles, char2idx, max_length):
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
        indices = indices[:max_length]
    return indices
