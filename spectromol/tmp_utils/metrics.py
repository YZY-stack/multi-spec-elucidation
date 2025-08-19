from rdkit import Chem
from rdkit.Chem import Descriptors, rdFMCS
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from collections import Counter, defaultdict
from math import sqrt
import re
import numpy as np
import os


# 宏观指标函数
def normalize_smiles(s):
    mol = Chem.MolFromSmiles(s)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        return None

def top1_accuracy(s1, s2):
    norm_s1 = normalize_smiles(s1)
    norm_s2 = normalize_smiles(s2)
    if norm_s1 and norm_s2:
        return 1 if norm_s1 == norm_s2 else 0
    return 0

def get_molecular_formula(s):
    mol = Chem.MolFromSmiles(s)
    if mol:
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
        element_counts = {element: int(count) if count else 1 for element, count in elements}
        return element_counts
    return None

def molecular_formula_accuracy(s1, s2):
    formula1 = get_molecular_formula(s1)
    formula2 = get_molecular_formula(s2)
    if formula1 and formula2:
        return 1 if formula1 == formula2 else 0
    return 0

def cosine_similarity(s1, s2):
    counter1, counter2 = Counter(s1), Counter(s2)
    intersection = set(counter1.keys()) & set(counter2.keys())
    numerator = sum([counter1[x] * counter2[x] for x in intersection])

    sum1 = sum([counter1[x] ** 2 for x in counter1.keys()])
    sum2 = sum([counter2[x] ** 2 for x in counter2.keys()])
    denominator = sqrt(sum1) * sqrt(sum2)

    return numerator / denominator if denominator != 0 else 0.0

def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n +1) for _ in range(m +1)]

    for i in range(m +1):
        dp[i][0] = i
    for j in range(n +1):
        dp[0][j] = j

    for i in range(1, m +1):
        for j in range(1, n +1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] +1,
                dp[i][j-1] +1,
                dp[i-1][j-1] + cost
            )
    return dp[m][n]

def molecular_weight(s):
    mol = Chem.MolFromSmiles(s)
    if mol:
        return Descriptors.MolWt(mol)
    return None

def molecular_weight_difference(s1, s2):
    mw1 = molecular_weight(s1)
    mw2 = molecular_weight(s2)
    if mw1 is not None and mw2 is not None:
        return abs(mw1 - mw2)
    return None

def get_species_counts(s, hydrogens=False):
    mol = Chem.MolFromSmiles(s)
    species_counts = defaultdict(int)
    if mol:
        for atom in mol.GetAtoms():
            element = atom.GetSymbol()
            if not hydrogens and element == 'H':
                continue
            species_counts[element] += 1
    return species_counts

def compare_species_counts(s1, s2, hydrogens=False):
    species_counts_1 = get_species_counts(s1, hydrogens)
    species_counts_2 = get_species_counts(s2, hydrogens)
    errors = 0
    for spe in species_counts_1.keys():
        counts1 = species_counts_1[spe]
        counts2 = species_counts_2.get(spe, 0)
        errors += abs(counts1 - counts2)
    for spe in species_counts_2.keys():
        if spe not in species_counts_1:
            errors += species_counts_2[spe]
    return errors

def get_formulas_avg_distance(s1, s2, hydrogens=False):
    try:
        dist = compare_species_counts(s1, s2, hydrogens=hydrogens)
        return dist
    except Exception as e:
        return None

def calculate_maccs_fingerprint(s):
    mol = Chem.MolFromSmiles(s)
    if mol:
        maccs_fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        return maccs_fp
    return None

def calculate_rdkit_fingerprint(s):
    mol = Chem.MolFromSmiles(s)
    if mol:
        rdkit_fp = Chem.RDKFingerprint(mol)
        return rdkit_fp
    return None

def calculate_morgan_fingerprint(s, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(s)
    if mol:
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return morgan_fp
    return None

def calculate_tanimoto_similarity(fp1, fp2):
    if fp1 is None or fp2 is None:
        return None
    return DataStructs.FingerprintSimilarity(fp1, fp2)

def maccs_similarity(s1, s2):
    fp1 = calculate_maccs_fingerprint(s1)
    fp2 = calculate_maccs_fingerprint(s2)
    return calculate_tanimoto_similarity(fp1, fp2)

def rdkit_similarity(s1, s2):
    fp1 = calculate_rdkit_fingerprint(s1)
    fp2 = calculate_rdkit_fingerprint(s2)
    return calculate_tanimoto_similarity(fp1, fp2)

def morgan_similarity(s1, s2, radius=2, nBits=2048):
    fp1 = calculate_morgan_fingerprint(s1, radius, nBits)
    fp2 = calculate_morgan_fingerprint(s2, radius, nBits)
    return calculate_tanimoto_similarity(fp1, fp2)

# 微观指标函数
def get_max_mcs(s1, s2):
    true_mol = Chem.MolFromSmiles(s1)
    true_atoms = true_mol.GetNumAtoms()

    predicted_mol = Chem.MolFromSmiles(s2)
    predicted_atoms = predicted_mol.GetNumAtoms()

    mols = [true_mol, predicted_mol]
    mcs = rdFMCS.FindMCS(mols, ringMatchesRingOnly=True,
                         atomCompare=Chem.rdFMCS.AtomCompare.CompareElements,
                         bondCompare=Chem.rdFMCS.BondCompare.CompareOrder,
                         timeout=60)
    mcs_atoms = mcs.numAtoms

    if mcs_atoms == 0:
        mcs_ratio = 0
        mcs_tanimoto = 0
        mcs_coefficient = 0
    else:
        mcs_ratio = mcs_atoms / true_atoms
        mcs_tanimoto = mcs_atoms / (true_atoms + predicted_atoms - mcs_atoms)
        mcs_coefficient = mcs_atoms / min(true_atoms, predicted_atoms)
    return mcs_ratio, mcs_tanimoto, mcs_coefficient

# 灵敏性和特异性函数
def calculate_sensitivity_specificity(s1, s2):
    count_s1 = Counter(s1)
    count_s2 = Counter(s2)
    sensitivity = {}
    specificity = {}

    all_elements = set(s1) | set(s2)

    for element in all_elements:
        TP = min(count_s1[element], count_s2[element])
        FN = count_s2[element] - TP
        FP = count_s1[element] - TP
        TN = 0

        sensitivity[element] = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity[element] = TP / (TP + FP) if (TP + FP) > 0 else 0

    return sensitivity, specificity
