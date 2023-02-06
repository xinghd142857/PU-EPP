# -*- coding: utf-8 -*-
# Author: xinghd


import torch
import torch.nn.functional as F
from torch import nn
import math

class FFN(nn.Module):
    """
    Two linear layers, Feed Foward
    """
    def __init__(self, ffn_input_num, ffn_hidden_num, ffn_outputs_num, **kwargs):
        """
        ffn_input_num: hidden_dim
        ffn_hidden_num: hidden_dim
        ffn_outputs_num: hidden_dim
        """
        super().__init__()
        self.linear1 = nn.Linear(ffn_input_num, ffn_hidden_num)
        self.activation = nn.LeakyReLU()
        self.linear2 = nn.Linear(ffn_hidden_num, ffn_outputs_num)
        if 'drop_rate' in kwargs:
            self.dropout = nn.Dropout(kwargs['drop_rate'])
        else:
            self.dropout = nn.Dropout(0.1)
    def forward(self, X):
        return self.linear2(self.dropout(self.activation(self.linear1(X))))

class AddNorm(nn.Module):
    ''' dropout -> add X -> normalization'''
    def __init__(self, norm_shape, drop_rate, **kwargs):
        """
        norm_shape: layernorm parameter (64)
        """
        super().__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.ln = nn.LayerNorm(norm_shape)
    def forward(self, X, y):
        return self.ln(self.dropout(y)+X)

class PositionalEncoding(nn.Module):
    """
    Positional Encoding
    """
    def __init__(self, hidden_dim, drop_rate, max_len=1500):
        """
        max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(drop_rate)
        # Create a sufficient P
        self.P = torch.zeros((1, max_len, hidden_dim))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, hidden_dim, 2, dtype=torch.float32) / hidden_dim)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    def forward(self, X):
#         print(X.shape)
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


def sequence_mask(X, valid_lengths, value=-1e10):
    """
    valid length value = 1
    invalid length value = 0
    """
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None,:] < valid_lengths[:,None]
    X[~mask] = value
    return X


def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
def masked_softmax(X, valid_lengths=None):
    """
    Make invalid numbers very small
    """
    if valid_lengths is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lengths.dim() == 1:
            valid_lengths = torch.repeat_interleave(valid_lengths, shape[1])
        else:
            valid_lengths = valid_lengths.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lengths,value=-1e10)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    """
    Softmax((q*k**T/dk**0.5)*v)
    """
    def __init__(self, drop_rate, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(drop_rate)
    def forward(self, queries, keys, values, valid_lengths=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lengths)
        return torch.bmm(self.dropout(self.attention_weights), values)

def transpose_qkv(X, num_heads):
    """
    for MultiHeadAttention(n heads to 1 large head)
    """
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])
class MultiHeadAttention(nn.Module):
    """
    n heads attention
    """
    def __init__(self, query_size, key_size, value_size, num_hiddens,
        num_heads, drop_rate, bias=False, **kwargs):
        """
        num_heads: number of heads
        query_size == key_size == value_size == hidden_dim
        Different names only represent different meanings, and they have the same value
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(drop_rate)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    def forward(self, queries, keys, values, valid_lengths=None):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lengths is not None:
            valid_lengths = torch.repeat_interleave(valid_lengths, repeats=self.num_heads, dim=0)
        
        output = self.attention(queries, keys, values, valid_lengths)
        
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

class EncoderBlock(nn.Module):
    """
    Muti-Head Attention -> AddNorm -> FFN -> AddNorm
    """
    def __init__(self, query_size, key_size, value_size, 
                 ffn_input_num, ffn_hidden_num, 
                 norm_shape,drop_rate, num_heads, num_hiddens,
                 bias=False, **kwargs):
        """
        query_size == key_size == value_size == hidden_dim
        Different names only represent different meanings, and they have the same value
        """
        super().__init__()
        self.attention = MultiHeadAttention(key_size, query_size, value_size, 
                                            num_hiddens, num_heads, drop_rate,bias)
        self.addnorm1 = AddNorm(norm_shape, drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        self.ffn = FFN(ffn_input_num, ffn_hidden_num, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, drop_rate)
    def forward(self, X, valid_lengths=None):
        tmp = self.attention(X,X,X, valid_lengths)
        Y = self.addnorm1(X, tmp)
        
        return self.addnorm2(Y, self.ffn(Y))

class Encoder(nn.Module):
    ''' input X: protein([batch_size, protein_length, protein_dim(100)]) '''
    def __init__(self, protein_dim, hidden_dim,
                 norm_shape, drop_rate=0.1, num_heads=8, 
                 num_layers=12, **kwargs):
        """
        protein_dim: embedding protein length (default: 100)
        hidden_dim: hidden layers dimension (default: 128)
        norm_shape: for AddNorm(LayerNorm) (default: 64)
        num_heads: number of attendtion heads(default: 12)
        num_layers: number of encoderblocks(default: 8)
        """
        super().__init__(**kwargs)
        self.num_hiddens = hidden_dim
        self.fc = nn.Linear(protein_dim, hidden_dim)
#         self.pos_encoding = PositionalEncoding(hidden_dim, drop_rate)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 EncoderBlock(hidden_dim, hidden_dim, hidden_dim, 
                                              hidden_dim, hidden_dim, 
                                              norm_shape,drop_rate, num_heads, hidden_dim,
                                              ))
            
    def forward(self, X, valid_lengths=None):
        self._attention_weights = [None] * len(self.blks)
        X = self.fc(X)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lengths)
            self._attention_weights[i] = blk.attention.attention.attention_weights
        return X

    @property
    def attention_weights(self):
        """
        For visualization
        """
        return self._attention_weights

class DecoderBlock(nn.Module):
    """
    Muti-Head Attention(self attention) -> AddNorm -> 
    Muti-Head Attention(encoder-decoder attention) -> AddNorm -> FFN-> AddNorm
    """
    def __init__(self, query_size, key_size, value_size, ffn_input_num, ffn_hidden_num, 
                 norm_shape, drop_rate, num_heads, num_hiddens, i, **kwargs):
        """
        i(0 - num_layers: int): ith layer
        query_size == key_size == value_size == hidden_dim
        Different names only represent different meanings, and they have the same value
        """
        super().__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            query_size, key_size, value_size, num_hiddens, num_heads, drop_rate)
        self.addnorm1 = AddNorm(norm_shape, drop_rate)
        self.attention2 = MultiHeadAttention(
            query_size, key_size, value_size, num_hiddens, num_heads, drop_rate)
        self.addnorm2 = AddNorm(norm_shape, drop_rate)
        self.ffn = FFN(ffn_input_num, ffn_hidden_num,num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, drop_rate)
    def forward(self, X, state, dec_valid_lens):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
            state[2][self.i] = key_values

        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # encoder-decoder attention
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class Decoder(nn.Module):
    '''
    Decoder with GCN
    input X: compound([batch_size, compound_length, atom_dim(46)]) & encoder output
    '''
    def __init__(self, atom_dim, hidden_dim,
                 norm_shape, drop_rate=0.1, num_heads=8, 
                 num_layers=12, **kwargs):
        """
        atom_dim: embedding atom length(default: 46)
        hidden_dim: hidden layers dimension (default: 64)
        norm_shape: for AddNorm(LayerNorm) (default: 64)
        num_heads: number of attendtion heads(default: 8)
        num_layers: number of encoderblocks(default: 6)
        """
        super().__init__(**kwargs)
        self.num_hiddens = hidden_dim
        self.num_layers = num_layers
        self.fc = nn.Linear(atom_dim, hidden_dim)

        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 DecoderBlock(hidden_dim, hidden_dim, hidden_dim, hidden_dim,
                                              hidden_dim, norm_shape,drop_rate, num_heads, 
                                              hidden_dim, i))
        self.drop_final = nn.Dropout(0.2) # special drop_rate
        self.dense1 = nn.Linear(hidden_dim, 256)
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 2)
        self.gcndrop1 = nn.Dropout(0.1)
        self.gcndrop2 = nn.Dropout(0.2)
        
    def init_state(self, enc_outputs, enc_valid_lens=None, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
    def gcn_init_weight(self, atom_dim):
        self.weight_1 = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.weight_2 = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))

        torch.nn.init.xavier_uniform_(self.weight_1)
        torch.nn.init.xavier_uniform_(self.weight_2)

    def gcn(self, compound, adj):
        """
        Graph Convolutional Neural Network
        compound: input feature
        adj: 
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                 dtype=np.int32).reshape(edges_unordered.shape)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(labels.shape[0], labels.shape[0]),
                                dtype=np.float32)

            # build symmetric adjacency matrix
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

            features = normalize(features)
            adj = normalize(adj + sp.eye(adj.shape[0]))
        """
        support = torch.matmul(compound, self.weight_1)
        output = torch.bmm(adj, support)

        support = torch.matmul(output, self.weight_2)
        output = torch.bmm(adj, support)

        return output
    def forward(self, X, adjs, state, dec_valid_lengths=None):
        X = self.gcn(X, adjs)
        X = self.fc(X)
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state, dec_valid_lengths)
            # self attention
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # encoder-decoder attention
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights

        norm = F.softmax(sequence_mask(torch.norm(X, dim=2), dec_valid_lengths), dim=1)
        X_sum = torch.sum(X * norm[:, :, None], dim=1)

        label = self.drop_final(F.leaky_relu(self.dense1(X_sum)))
        label = self.drop_final(F.leaky_relu(self.dense2(label)))
        return self.dense3(label)
    @property
    def attention_weights(self):
        """
        For visualization
        """
        return self._attention_weights

class ModelCat(nn.Module):
    """
    concat Encoder Decoder
    protein: embedding protein(encoder X)([batch_size, protein_length, protein_dim])
    protein_nums: the protein length of each protein
    compounds: embedding compounds(decoder X)([batch_size, compound_length, atom_dim])
    adjs: gcn adj
    compound_nums: the compound length of each compound
    """
    def __init__(self, encoder, decoder, atom_dim=46):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.atom_dim = atom_dim

        self.decoder.gcn_init_weight(self.atom_dim)

    def forward(self, protein, protein_nums, compounds, adjs, compound_nums):

        enc_outputs = self.encoder(protein, protein_nums)
        dec_state = self.decoder.init_state(enc_outputs, protein_nums)
        output = self.decoder(compounds, adjs, dec_state, compound_nums)
        return output