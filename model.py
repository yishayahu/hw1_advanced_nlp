import torch
from torch import nn
import torch.nn.init as init
import math
import numpy as np
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
class WSDModel(nn.Module):

    def __init__(self, V, Y, D=300, dropout_prob=0.2, use_padding=False,use_positional_encoding=False,causal=False):
        super(WSDModel, self).__init__()
        self.use_padding = use_padding

        self.D = D
        self.pad_id = 0
        self.E_v = nn.Embedding(V, D, padding_idx=self.pad_id)
        self.E_y = nn.Embedding(Y, D, padding_idx=self.pad_id)
        init.kaiming_uniform_(self.E_v.weight[1:], a=math.sqrt(5))
        init.kaiming_uniform_(self.E_y.weight[1:], a=math.sqrt(5))

        self.W_A = nn.Parameter(torch.Tensor(D, D))
        self.W_O = nn.Parameter(torch.Tensor(D, D))
        init.kaiming_uniform_(self.W_A, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_O, a=math.sqrt(5))

        self.dropout_layer = nn.Dropout(p=dropout_prob)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm([self.D])
        self.pos_encoder = None
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(V, dropout_prob)
        self.causal= causal

    def attention(self, X, Q, mask):
        """
        Computes the contextualized representation of query Q, given context X, using the attention model.

        :param X:
            Context matrix of shape [B, N, D]
        :param Q:
            Query matrix of shape [B, k, D], where k equals 1 (in single word attention) or N (self attention)
        :param mask:
            Boolean mask of size [B, N] indicating padding indices in the context X.

        :return:
            Contextualized query and attention matrix / vector
        """
        Q_c = None
        A = None        
        

        # Have a look at the difference between torch.matmul() and torch.bmm().
        A = torch.matmul(Q, self.W_A)
        A = torch.matmul(A, X.transpose(-2, -1))



        if self.use_padding:

            A[A==self.pad_id] = -10000
            if self.causal:
                A = A + torch.tensor(np.triu(np.full(A.shape, -10000), k=1),device=A.device)
        A = self.softmax(A)
        Q_c = torch.matmul(A, X)
        Q_c = torch.matmul(Q_c, self.W_O)

        return Q_c, A.squeeze()

    def forward(self, M_s, v_q=None,):
        """
        :param M_s:
            [B, N] dimensional matrix containing token integer ids
        :param v_q:
            [B] dimensional vector containing query word indices within the sentences represented by M_s.
            This argument is only passed in single word attention mode.

        :return: logits and attention tensors.
        """

        X = self.dropout_layer(self.E_v(M_s))   # [B, N, D]
        
        Q = None
        if v_q is not None:

            Q = torch.gather(M_s, 1, v_q.unsqueeze(1))
            Q = self.E_v(Q)
            # Look up the gather() and expand() methods in PyTorch.

        else:
            # TODO Part 3: Your Code Here.
            Q = self.E_v(M_s)
        if self.pos_encoder:
            Q = self.pos_encoder(Q)
            

        mask = M_s.ne(self.pad_id)
        Q_c, A = self.attention(X, Q, mask)
        H = self.layer_norm(Q_c + Q)
        
        E_y = self.dropout_layer(self.E_y.weight)
        y_logits = (H @ E_y.T).squeeze()
        return y_logits, A
