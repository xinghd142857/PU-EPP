# -*- coding: utf-8 -*-
# Author: xinghd

from model.model import *
from .data_parser import data_collate, AtomFeatures
from .word2vec import seq_to_kmers, get_protein_embedding
from .tools import d2D,load_word2vec
import os
from functools import lru_cache
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def get_model_vis(cfg):
    encoder = Encoder(cfg.protein_dim, cfg.hid_dim, cfg.norm_shape)
    decoder = Decoder(cfg.atom_dim, cfg.hid_dim, cfg.norm_shape)
    model = ModelCat(encoder, decoder)
    model = model.to(cfg.DEVICE)
    model = nn.DataParallel(model, device_ids = list(range(cfg.gpu_number)))
#     if cfg.gpu_number == 1:
#         model = model.module
    if cfg.state_dict_path is not None:
        if os.path.exists(cfg.state_dict_path):
            model.load_state_dict(torch.load(cfg.state_dict_path))
            print('success load state dict')
        else:
            raise ValueError('Wrong path')

    else:
        raise ValueError('No state dict path')
    if cfg.gpu_number == 1:
        model = model.module
    model.to(cfg.DEVICE)
    return model


class SmilesSeqEmbedding(AtomFeatures):
    def __init__(self, cfg) -> None:
        self.seq_emb_model = load_word2vec(cfg.word2vec_path)
        self.atom_dim = cfg.atom_dim
    @lru_cache(maxsize=None)
    def smiles_embedding(self, smiles):
        """
        smiles embedding
        """
        atom_feature, adj = self.mol_features(smiles)
        atom_feature = torch.FloatTensor(atom_feature)
        adj = torch.FloatTensor(adj)
        return atom_feature, adj
    @lru_cache(maxsize=None)
    def sequence_embedding(self, seq): 
        """
        protein sequence embedding
        """
        sequence = seq#pro2seq[seq]
        protein_embedding = get_protein_embedding(self.seq_emb_model, seq_to_kmers(sequence))
        protein = torch.FloatTensor(protein_embedding)
        return protein

def get_feature_embedding(cfg):
    """
    get feature embedding
    """
    return SmilesSeqEmbedding(cfg)

def count_decoder_attetion(chem_attention, attetion_index:int, layer:list, head:list):
    """

    Args:
        chem_attention (tensor): chem_attention
        attetion_index (int): attention index
        layer (list): layer [x,y] contains y
        head (list): head [x,y] contains y
    Example:
        chem_attention = count_decoder_attetion(chem_attention, attetion_index=0, layer=[0,5], head=[4,4])
    Returns:
        np.array: decoder_attention
    """
    assert 0<=attetion_index<2, 'attention index should be 0 or 1'
    assert 0<=layer[0]<=layer[1]<12, 'layer should be 0-11'
    assert 0<=head[0]<=head[1]<8, 'head should be 0-7'
    chem_attention = torch.stack(chem_attention[attetion_index])[layer[0]:layer[1]+1,head[0]:head[1],...]
    chem_attention = torch.sum(chem_attention, dim=0)
    chem_attention = torch.sum(chem_attention,dim=0).numpy()
    chem_attention =np.sum(chem_attention,0)
    chem_attention = MinMaxScaler().fit_transform(chem_attention.reshape(-1, 1))
    chem_attention = np.sum(chem_attention,1)
    return chem_attention

def count_encoder_attetion(seq_attention, layer:list, head:list):
    """
    Args: 
        seq_attention (tensor): seq_attention 
        layer (list): layer [x,y] contains y
        head (list): head [x,y] contains y
    Example:
        seq_attention = count_en_attetion(seq_attention, layer=[0,5], head=[4,4])
    Returns:
        np.array: en_attention
    """
    assert 0<=layer[0]<=layer[1]<12, 'layer should be 0-11'
    assert 0<=head[0]<=head[1]<8, 'head should be 0-7'
    seq_attention = torch.stack(seq_attention)[layer[0]:layer[1]+1,head[0]:head[1],...]
    seq_attention = torch.sum(seq_attention, dim=0)
    seq_attention = torch.sum(seq_attention,dim=0).numpy()
    seq_attention =np.sum(seq_attention,0)
    seq_attention = MinMaxScaler().fit_transform(seq_attention.reshape(-1, 1))
    seq_attention = np.sum(seq_attention,1)
    return seq_attention