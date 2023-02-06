# -*- coding: utf-8 -*-
# Author: xinghd


import torch
import numpy as np
from rdkit import Chem

def data_collate(compounds, adjs, proteins, labels=None, device='cpu'):
    """
    Collate data into a batch

    Args:
        compounds : data of compounds
        adjs: data of adjacency matrix
        proteins: data of proteins
        labels: data of labels(optional: if not given,  you should use Predictor)
        device_params (torch device): Device to use.

    Returns:
        tuple of tensors: (compounds, adjs, proteins, labels, valid_mask of compounds and proteins)
    """
    # must have same length and same dim_length
    assert len(compounds) == len(proteins)
    assert compounds[0].dim() == proteins[0].dim() == 2
    N = len(compounds)

    # default 46, 100
    atom_dim, protein_dim = compounds[0].shape[1], proteins[0].shape[1]
    
    # all compounds legnth and all proteins length
    atom_nums = [atom.shape[0] for atom in compounds]
    protein_nums = [protein.shape[0] for protein in proteins]

    compounds_max_length = max(atom_nums)
    proteins_max_length = max(protein_nums)

    compounds_new = torch.zeros((N, compounds_max_length, atom_dim), device=device)
    for i, atom in enumerate(compounds):
        compounds_new[i, :atom_nums[i], :] = atom

    adjs_new = torch.zeros((N, compounds_max_length, compounds_max_length), device=device)
    for i, adj in enumerate(adjs):
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        adjs_new[i, :a_len, :a_len] = adj

    proteins_new = torch.zeros((N, proteins_max_length, protein_dim), device=device)
    for i, protein in enumerate(proteins):
        proteins_new[i, :protein_nums[i], :] = protein
    

    atom_nums = torch.tensor(atom_nums, dtype=torch.int).to(device)
    protein_nums = torch.tensor(protein_nums, dtype=torch.int).to(device)
    if labels is not None:
        labels_new = torch.tensor(labels, dtype=torch.long, device=device)
        return (compounds_new, adjs_new, proteins_new, labels_new, atom_nums, protein_nums)
    return compounds_new, adjs_new, proteins_new, atom_nums, protein_nums
    

def collate_fn(batch):
    """
    Args:
        batch: list of data, each atom, adj, protein, (label)
    Note:
        if label is not given, you should use Predictor
        else, you should use Trainer
    """
    collate_data = zip(*batch)
    if len(batch[0]) == 3:
        compounds, adjs, proteins = collate_data
        labels = None
    elif len(batch[0]) == 4:
        compounds, adjs, proteins, labels = collate_data
    else:
        raise ValueError("Wrong collate_fn input")
    return data_collate(compounds, adjs, proteins, labels)


class AtomFeatures:
    def atom_features(self, atom,explicit_H=False,use_chirality=True):
        """Generate atom features including atom symbol(10),degree(7),formal charge,
        radical electrons,hybridization(6),aromatic(1),Chirality(3)
        """
        symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
        degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
        hybridizationType = [Chem.rdchem.HybridizationType.SP,
                                Chem.rdchem.HybridizationType.SP2,
                                Chem.rdchem.HybridizationType.SP3,
                                Chem.rdchem.HybridizationType.SP3D,
                                Chem.rdchem.HybridizationType.SP3D2,
                                'other']   # 6-dim
        results = self.one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                    self.one_of_k_encoding(atom.GetDegree(),degree) + \
                    [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                    self.one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + \
                    self.one_of_k_encoding_unk(atom.GetExplicitValence(), [1,2,3,4,5,6]) +\
                    self.one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5]) +\
                    [atom.GetIsAromatic()]  # 10+7+2+6+6+6+1=38
#                     self.one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]  # 10+7+2+6+1=26

        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + self.one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                        [0, 1, 2, 3, 4])   # 38+5=43
        if use_chirality:
            try:
                results = results + self.one_of_k_encoding_unk(
                        atom.GetProp('_CIPCode'),
                        ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 43+3 =46
        return results
    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(
                x, allowable_set))
        return [x == s for s in allowable_set]
    def one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return [x == s for s in allowable_set]

    def adjacent_matrix(self, mol):
        adjacency = Chem.GetAdjacencyMatrix(mol)
        return np.array(adjacency)
    def mol_features(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            raise RuntimeError("SMILES cannot been parsed!")
        mol = Chem.AddHs(mol)
        atom_feat = np.zeros((mol.GetNumAtoms(), self.atom_dim))
        for atom in mol.GetAtoms():
            atom_feat[atom.GetIdx(), :] = self.atom_features(atom)
        adj_matrix = self.adjacent_matrix(mol)
        return atom_feat, adj_matrix
def get_shuffle_data(df, length=None, test_length=20000):
    """"
    input dataframe and return train test data
    """
    if not length:
        length = len(df)
    arr = np.arange(length)
    np.random.shuffle(arr)
    test_i, train_i = arr[:test_length], arr[test_length:]
    return df.iloc[train_i].reset_index(drop=True), df.iloc[test_i].reset_index(drop=True)