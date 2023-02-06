# -*- coding: utf-8 -*-
# Author: xinghd


"""
trianer = Trainer(model, CFG)
"""

from functools import lru_cache
import torch
import torch.nn.functional as F
from torch import nn

from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
import numpy as np
import pandas as pd

import math
import os
from rdkit import Chem


from collections import defaultdict

from .lookahead import Lookahead
from .word2vec import seq_to_kmers, get_protein_embedding
from .data_parser import AtomFeatures
from .tools import load_word2vec, d2D, get_month_day, split2PT
from .Radam import *


class LabelSmoothingLoss(nn.Module):
    r'''CrossEntropyLoss with Label Smoothing.
    Args:
        classes (int): number of classes.
        smoothing (float): label smoothing value.
        dim (int): the dimension to sum.
    Shape:
        - pre: :math:`(N, C)` where `C = 2`, predicted scores.
        - target: :math:`(N)`, ground truth label
    Returns:
        loss
    '''
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class Trainer:
    r'''Train and evaluate the model.
    Args:
        cfg: config
        dataset_packed: dataloader of train dataset
        device_params (torch device): Device to use.
    Functions:
        self.train will return loss_total, U_m_P
        self.test will return AUC, Precision, Recall, PRC
    '''
    def __init__(self, model, cfg, trainfirst=False) -> None:

        self.model = model
        self.Loss = LabelSmoothingLoss(classes=2, smoothing=0.1) if cfg.ifsmoothing else nn.CrossEntropyLoss() 
        
        self.del_threshold = cfg.del_threshold
        weight_p, bias_p = [], []
        if cfg.state_dict_path is None and trainfirst:
           
            
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        self.del_threshold = 1 #cfg.del_threshold
        self.quantile = cfg.quantile
        self.optimizer_inner = RAdam(
            [{'params': weight_p, 'weight_decay': cfg.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=cfg.lr)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)
        self.ifpu = cfg.ifpu
        self.batch_size = cfg.BATCH_SIZE
        self.del_threshold_rem = []
    def train(self, dataset_packed, device_params=torch.device('cuda:0')):
        self.model.train()
        loss_total = 0
        U_m_P = []
        S = np.array([])
        for i, data_pack in enumerate(dataset_packed):
            compound, adjs, protein, label ,compound_num,protein_num = d2D(data_pack, device_params)
            
            output = self.model(protein, protein_num, compound,adjs, compound_num)
            loss = self.Loss(output, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.detach().cpu().numpy()
            if self.ifpu:
                with torch.no_grad():
                    correct_labels = label.detach().to('cpu').data.numpy()
                    ys = F.softmax(output, 1).to('cpu').data.numpy()
                    for j in range(len(correct_labels)):
                        if correct_labels[j] < 0.1 and ys[j][1] > self.del_threshold:
                            U_m_P.append(i*self.batch_size+j)
                    predicted_scores = ys[correct_labels >0.5, 1]
                    S = np.append(S, predicted_scores)
            message = 'Train Step {}/{}, loss: {:.5f}, loss_total: {:.5f}'
            self.info_message(message, i, len(dataset_packed), loss, loss_total, end="\r")
        if self.ifpu:
            self.del_threshold = np.quantile(S, self.quantile)
            self.del_threshold_rem.append(self.del_threshold)
        return loss_total, U_m_P
    
    def test(self, dataset, device_params=torch.device('cuda:0')):
        self.model.eval()
        T, Y, S = [], [], []
        with torch.no_grad():
            for _, data_pack in enumerate(dataset):
                compound,adjs,protein, label ,compound_num,protein_num = d2D(data_pack, device_params)
                predicted_interaction = self.model(protein, protein_num, compound,adjs, compound_num)
                
                correct_labels = label.to('cpu').data.numpy()
                ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
                predicted_labels = np.argmax(ys, axis=1)
                predicted_scores = ys[:, 1]
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
        AUC = roc_auc_score(T, S)
        Precision = precision_score(T, Y)
        Recall = recall_score(T, Y)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        return AUC, Precision, Recall, PRC

    def save_AUCs(self, AUCs, filename):
        """
        save log
        """
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        """
        save model
        """
        torch.save(model.state_dict(), filename)


    def info_message(self, message, *args, end="\n") -> None:
        print(message.format(*args), end=end)
            


class Predictor:
    r"""Predict the label of dataset
    Args:
        dataset (PreReader): dataset.
        device_params (torch device): Device to use.

    Returns:
        np array [dataset length]: predicted labels,
        np array [dataset length]: predicted scores.
    """

    def __init__(self, model) -> None:
        self.model = model
    def predict(self, dataset, device_params=torch.device('cuda:0')):
        self.model.eval()
        Y, S = [], []
        with torch.no_grad():
            for i, data_pack in enumerate(dataset):
                compound,adjs,protein, compound_num,protein_num = d2D(data_pack, device_params)
                predicted_interaction = self.model(protein, protein_num, compound,adjs, compound_num)

                ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
                predicted_labels = np.argmax(ys, axis=1)
                predicted_scores = ys[:, 1]
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
        return Y, S

class PredictReader(torch.utils.data.Dataset, AtomFeatures):
    r'''Read the dataset for prediction
    Args:
        data (pd.DataFrame): Predict dataset.
        pro2seq (dict): proteinid/name(data['seq']) to sequence mapping.(if not provided, it will be data['seq']).
        word2vec_path (str): path to word2vec file.
    Note:
        1. data must have only two columns: 'smiles' and 'seq'
        2. 'seq' can be either id/name or protein sequence, if seq is id/name, you must provide pro2seq, otherwise it shuld be None.
    Returns:
        Generator: atom feature, adjacency matrix, embedding protein.
        =================================================================================
        ==If self.weong_w2d is empty, the input order is the same as the output order, ==
        ==otherwise you must delete all the input data in self.weong_w2d.              ==
        ================================================================================= 
    '''
    atom_dim = 46
    def __init__(self, data, pro2seq=None,
                 word2vec_path="./model/model_save/seq_Embedding_plus.model") -> None:
        super().__init__()
        self.data = data
        self.pro2seq = pro2seq
        self.weong_w2d = pd.DataFrame(columns=['smiles', 'seq'])
        self.seq_emb_model = load_word2vec(self.checkpath(word2vec_path))

    def checkpath(self, path):
        """
        Check if the path exists
        """
        if os.path.exists(path): return path
        raise Exception(f"Wrong path: {path}, this path does not exist")
        
    # @lru_cache
    @lru_cache(maxsize=None)
    def sequence_embedding(self, seq): 
        """
        protein sequence embedding
        """
        sequence = self.pro2seq[seq] if self.pro2seq is not None else seq
        protein_embedding = get_protein_embedding(self.seq_emb_model, seq_to_kmers(sequence))
        protein = torch.FloatTensor(protein_embedding)
        return protein
        
    # @lru_cache
    @lru_cache(maxsize=None)
    def smiles_embedding(self, smiles):
        """
        smiles embedding
        """
        atom_feature, adj = self.mol_features(smiles)
        atom_feature = torch.FloatTensor(atom_feature)
        adj = torch.FloatTensor(adj)
        return atom_feature, adj

    def data_embedding(self, data):
        assert isinstance(data, pd.core.series.Series), "error type"
        assert len(data) == 2, "error dataframe length, did you drop the index?"
        smiles, sequence = data
        atom_feature, adj = self.smiles_embedding(smiles)
        
        protein = self.sequence_embedding(sequence)
        return (atom_feature, adj, protein)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            res = self.data_embedding(self.data.iloc[idx])
        except:
            self.weong_w2d = self.weong_w2d.append(self.data.iloc[idx])
            res = self.data_embedding(self.data.iloc[0])
        return res

class TestReader(PredictReader):
    r'''Read the dataset for testing
    Args:
        data (pd.DataFrame): Test dataset.
        positive_data (pd.DataFrame): positive dataset.
        unlabeled_data (pd.DataFrame): unlabeled dataset.
        pro2seq (dict): proteinid/name(data['seq']) to sequence mapping.(if not provided, it will be data['seq']).
        word2vec_path (str): path to word2vec file.
    Note:
        1. you can use data or (positive_data and unlabeled_data)
        2. data must have only two columns: 'smiles', 'seq', 'label'
        3. 'seq' can be either id/name or protein sequence, if seq is id/name, you must provide pro2seq, otherwise it shuld be None.
        4. 'label' can be either 0 or 1
    Returns:
        Generator: atom feature, adjacency matrix, embedding protein.

    '''
    def __init__(self, data=None,positive_data=None, unlabel_data=None, pro2seq=None,
                 word2vec_path="./model/model_save/seq_Embedding_plus.model",
                ) -> None:
        if data is None:
            assert positive_data is not None and unlabel_data is not None,'wrong input'
        if data is not None: 
            assert positive_data is None and unlabel_data is None, 'wrong input'
        if data is not None:
            self.positive, self.unlabel = split2PT(data)
        else:
            self.positive = positive_data
            self.unlabel = unlabel_data
        self.data = data
        self.pro2seq = pro2seq
        
        self.P_length = len(self.positive)
        self.U_length = len(self.unlabel)

        self.seq_emb_model = load_word2vec(self.checkpath(word2vec_path))

    def data_embedding(self, data):
        assert isinstance(data, pd.core.series.Series), "error type"
        assert len(data) == 3, "error dataframe length, did you drop the index?"
        smiles, sequence, interaction = data
        atom_feature, adj = self.smiles_embedding(smiles)
        protein = self.sequence_embedding(sequence)
        label = np.array(interaction, dtype=np.float32)
        label = torch.LongTensor(label)
        return (atom_feature, adj, protein, label)
        
    def __len__(self):
        return self.P_length+self.U_length

    def __getitem__(self, index):
        if self.data is not None:
            return self.data_embedding(self.data.iloc[index])
        if index < self.P_length:
            return self.data_embedding(self.positive.iloc[index])
        return self.data_embedding(self.unlabel.iloc[index-self.P_length])



class TrainReader(TestReader):
    """
    Only for train
    Need split positive and unlabel data
    """
    
    def __init__(self, data=None,positive_data=None, unlabel_data=None,pro2seq=None,
                 U_m_P_savepath="./data/U_m_P/",
                 word2vec_path="./model/model_save/seq_Embedding_plus.model",
                 ifpu=True, 
                ) -> None:
        """"
        U_m_P_savepath: to save data that may be positive in unlabel data 
        """
        if data is None:
            assert positive_data is not None and unlabel_data is not None,'wrong input'
        if data is not None: 
            assert positive_data is None and unlabel_data is None, 'wrong input'
        if data is not None:
            self.data = data
            self.positive, self.unlabel = split2PT(data)
        else:
            self.positive = positive_data
            self.unlabel = unlabel_data
        
        self.choice_rem = self.positive.copy()
        self.epoch_rem = []
        
        
        self.pro2seq = pro2seq
        

        self.P_length = len(self.positive)
        self.U_length = len(self.unlabel)
        self.U_may_P_count = 0
        
        self.seq_emb_model = load_word2vec(self.checkpath(word2vec_path))
        
        self.ifpu = ifpu
        if self.ifpu:
            self.umps = self.checkpath(U_m_P_savepath)
            self.loader_indexes = np.zeros(self.P_length*2, dtype=int)
            self.loader_cnt = 0
        """
        reset_T(): Very important, to get index of unlabel data
        """ 
        self.reset_T()
    def reset_T(self):
        if self.ifpu:
            self.U_length = len(self.unlabel)
            self.unlabels_index = np.random.choice(np.arange(self.U_length), replace=False, size=self.P_length)
            self.choice_rem = pd.concat([self.choice_rem, self.unlabel.iloc[self.unlabels_index]], ignore_index=True).drop_duplicates()
            self.epoch_rem = self.unlabels_index
            self.loader_indexes = np.zeros(self.P_length*2, dtype=int)
    def del_U(self, epoch=None, epoch_may_pos_indexes=None):
        """
        you can use this function as reset_T without epoch_may_pos_indexes
        for delete unlabel data and save data that may be a positive sample
        """
        if self.ifpu and epoch_may_pos_indexes is not None:
            self.U_may_P_count += len(epoch_may_pos_indexes)
            assert self.U_length-len(epoch_may_pos_indexes) >= self.P_length, "not enough unlabel data"
            del_indexes = self.loader_indexes[epoch_may_pos_indexes]
            self.unlabel.iloc[del_indexes].to_csv(f"{self.umps}{epoch}_{len(epoch_may_pos_indexes)}_{get_month_day()}.csv", index=False)
            self.unlabel = self.unlabel.drop(del_indexes).reset_index(drop=True)
        self.reset_T()
    def __len__(self):
        return self.P_length*2 if self.ifpu else self.P_length+self.U_length
    def __getitem__(self, index):
        if index < self.P_length:
            return self.data_embedding(self.positive.iloc[index])
        if self.ifpu:
            selfindex = self.unlabels_index[index-self.P_length]
            self.loader_indexes[index] = selfindex
        else:
            selfindex = index-self.P_length
        return self.data_embedding(self.unlabel.iloc[selfindex])
    


    








