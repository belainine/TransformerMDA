''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

def normaliseD(inputa):
    
    size=inputa.size()
    
    donum=torch.sqrt(torch.sum(inputa*inputa,3)).view(-1)
    inputa=inputa.view(-1,size[-1]).transpose(0,1)
    inputa=torch.div(inputa,donum).transpose(0,1).view(size)
    return inputa
def normaliseE(inputa):
    
    size=inputa.size()
    
    donum=torch.sqrt(torch.sum(inputa*inputa,4)).view(-1)
    inputa=inputa.view(-1,size[-1]).transpose(0,1)
    inputa=torch.div(inputa,donum).transpose(0,1).view(size)
    return inputa
class DotProductAttentionD(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, BSM=True):
        super().__init__()
        self.temperature = temperature
        self.BSM=BSM
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        
        attn = torch.matmul(q if self.BSM==True else q/ self.temperature
                            , k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
    
def softmax2D(attn):
    
    attnt=attn.transpose(1,3)
    sizereshape2d=[i for i in attnt.size()[:3]]+[-1]
    attnt1d=F.softmax(attnt.contiguous().view(*sizereshape2d), dim=-1).view(attnt.size())
    attnt1d=attnt1d.transpose(1,3)
    return attnt1d
class DotProductAttentionE(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, BSM=True):
        super().__init__()
        self.temperature = temperature
        self.BSM=BSM
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q if self.BSM==True else q/ self.temperature
                            , k.transpose(3, 4))

        if mask is not None:
            #attn = attn.masked_fill(mask.transpose(1,4) == 0, -1e9)
            #attn = attn.masked_fill(mask.transpose(1,2) == 0, -1e9)

            mask=mask.transpose(0,2)
            mask=mask.transpose(1,3)
            attn=attn.transpose(0,2)
            attn=attn.transpose(1,3)

            attn = attn.masked_fill(mask == 0, -1e9)
            attn=attn.transpose(0,2)
            attn=attn.transpose(1,3)
        attn = self.dropout(softmax2D(attn))
        #attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
class MultiHeadAttentionBSM3D(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_q, d_qi, dropout=0.1):
        super().__init__()
        
        self.n_head = n_head
        self.d_q = d_model//n_head
        self.d_qi = d_model//n_head
        self.d_model = d_model
        self.w_qs = nn.Linear(d_model, n_head * d_qi, bias=False)

        self.w_qis = nn.Linear(d_model, n_head * d_qi, bias=False)
        self.fc = nn.Linear(n_head * d_qi, d_model, bias=False)

        self.attention = DotProductAttentionE(temperature=d_q ** 0.5,BSM=True)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, qi, mask=None):
        
        d_q, d_qi, n_head = self.d_q, self.d_qi, self.n_head
        sz_b, len_q, len_qi = q.size(0), q.size(2), qi.size(2)

        residual = q
        q = self.layer_norm(q)
        #qi = self.layer_norm2(qi)
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q)
        q=q.view(sz_b,-1, len_q, n_head, d_q)
        qi = self.w_qis(qi)
        qi=qi.view(sz_b,-1, len_qi, n_head, d_qi)
        
        #q = self.w_qs(q).view(sz_b, len_q, n_head, d_q)
        #qi = self.w_qis(qi).view(sz_b, len_qi, n_head, d_qi)
        
        qi_n=normaliseE(qi).transpose(2,3)
        #qi_n=qi.transpose(2,3)
        # Transpose for attention dot product: b x n x lq x dv
        q = normaliseE(q).transpose(2,3)
        #q = q.transpose(2,3)
        if mask is not None:
            
            mask = mask.unsqueeze(2)   # For head axis broadcasting.
            
        q, attn = self.attention(q, qi_n, qi_n, mask=mask)
        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        
        q = q.transpose(2, 3).contiguous().view(sz_b,-1,  len_q, self.d_model)
        #q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        
        q = self.dropout(self.fc(q))
        q += residual

        return q, attn
class MultiHeadAttentionBSM2D(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_q, d_qi, dropout=0.1):
        super().__init__()
        
        self.n_head = n_head
        self.d_q = d_model//n_head
        self.d_qi = d_model//n_head
        self.d_model = d_model
        self.w_qs = nn.Linear(d_model, n_head * d_qi, bias=False)

        self.w_qis = nn.Linear(d_model, n_head * d_qi, bias=False)
        self.fc = nn.Linear(n_head * d_qi, d_model, bias=False)

        self.attention = DotProductAttentionD(temperature=d_q ** 0.5,BSM=True)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, qi, mask=None):
        
        d_q, d_qi, n_head = self.d_q, self.d_qi, self.n_head
        sz_b, len_q, len_qi = q.size(0), q.size(1), qi.size(1)

        residual = q
        q = self.layer_norm(q)
        #qi = self.layer_norm2(qi)
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_q)
        qi = self.w_qis(qi)
        
        qi=qi.view(sz_b, len_qi, n_head, d_qi)
        
        #q = self.w_qs(q).view(sz_b, len_q, n_head, d_q)
        #qi = self.w_qis(qi).view(sz_b, len_qi, n_head, d_qi)
        
        qi_n=normaliseD(qi).transpose(1,2)
        # Transpose for attention dot product: b x n x lq x dv
        q = normaliseD(q).transpose(1,2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, qi_n, qi_n, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        
        q = q.transpose(1, 2).contiguous().view(sz_b, -1, self.d_model)
        #q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        
        q = self.dropout(self.fc(q))
        q += residual

        return q, attn
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = DotProductAttentionD(temperature=d_k ** 0.5, BSM=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        #q = q.transpose(1, 2).contiguous().view(sz_b, -1, self.d_model)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        return q, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.layer_norm(x) 
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
      
        x += residual

        return x
class PositionwiseFeedForward3D(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x=x.transpose(1, 2)
        shape_x=x.size()
        x=x.reshape(shape_x[0], shape_x[1], -1)
        x = self.layer_norm(x)

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x=x.reshape(*shape_x)
        x=x.transpose(1, 2)
        x += residual

        return x
class ScaleFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_out, dropout=0.1):
        super().__init__()
        
        self.w_1 = nn.Linear(d_in, d_out) # position-wise

        self.layer_norm = nn.LayerNorm(d_out, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        
        x = self.layer_norm(x)
        x=x.transpose(1, 2)
        x=x.reshape(x.size()[0], x.size()[1], -1)
        
        x = self.w_1(x)
        x = self.dropout(x)
        

        return x