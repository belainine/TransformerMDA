''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer
from transformer.SubLayers import MultiHeadAttention, MultiHeadAttentionBSM2D, MultiHeadAttentionBSM3D, PositionwiseFeedForward,PositionwiseFeedForward3D,ScaleFeedForward

__author__ = "Yu-Hsiang Huang"


def get_pad_mask(seq, pad_idx):

    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    
    subsequent_mask = (1 - torch.triu(
        torch.ones((1 , len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask
class PositionalEncoding_Sphere3D(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding_Sphere3D, self).__init__()

        # Not a parameter
        sinusoid_table_v,sinusoid_table_h=self._get_sinusoid_encoding_table(n_position, d_hid)
        self.register_buffer('pos_table_v', sinusoid_table_v)
        self.register_buffer('pos_table_h', sinusoid_table_h)

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 3 * (hid_j // 3) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table_v = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table_h = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table_v[:, 0::3] = np.sin(sinusoid_table_v[:, 0::3])  # dim 3i
        sinusoid_table_v[:, 1::3] = np.cos(sinusoid_table_v[:, 1::3])  # dim 3i+1
        sinusoid_table_v[:, 2::3] = np.power(sinusoid_table_v[:, 2::3],0) # dim 3i+2
        
        
        sinusoid_table_h[:, 0::3] = np.cos(sinusoid_table_h[:, 0::3])  # dim 3i
        sinusoid_table_h[:, 1::3] = np.sin(sinusoid_table_h[:, 1::3])  # dim 3i+1
        sinusoid_table_h[:, 2::3] = np.cos(sinusoid_table_h[:, 2::3])  # dim 3i+2
        return torch.FloatTensor(sinusoid_table_v).unsqueeze(0),torch.FloatTensor(sinusoid_table_h).unsqueeze(0)

    def forward(self, x, isEncoder=True):
        #return x + self.pos_table[:, :x.size(1)].clone().detach()
        #    x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(2 if isEncoder else 1)
        batch_size=x.size(2 if isEncoder else 1)
        pos_table_v,pos_table_h = self.pos_table_v[:,:seq_len].clone().detach(),self.pos_table_h[:,:x.size(1)].clone().detach()
        if x.is_cuda:
            pos_table_v.cuda()
            pos_table_h.cuda()
        pos_table_h=pos_table_h.transpose(0,1).repeat(1,seq_len,1)
        pos_table_v=pos_table_v.repeat(x.size(1),1,1)
        
        cube=pos_table_v*pos_table_h#/np.sqrt(x.size(3)/3)#[:x.size(1)]
        
        x = x + cube
        return x
class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        sinusoid_table_v,sinusoid_table_h=self._get_sinusoid_encoding_table(n_position, d_hid)
        self.register_buffer('pos_table_v', sinusoid_table_v)
        self.register_buffer('pos_table_h', sinusoid_table_h)

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table_v = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table_h = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table_v[:, 0::2] = np.sin(sinusoid_table_v[:, 0::2])  # dim 2i
        sinusoid_table_v[:, 1::2] = np.cos(sinusoid_table_v[:, 1::2])  # dim 2i+1
        
        sinusoid_table_h[:, 0::2] = np.cos(sinusoid_table_h[:, 0::2])  # dim 2i
        sinusoid_table_h[:, 1::2] = np.sin(sinusoid_table_h[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table_v).unsqueeze(0),torch.FloatTensor(sinusoid_table_h).unsqueeze(0)

    def forward(self, x, isEncoder=True):
        #return x + self.pos_table[:, :x.size(1)].clone().detach()
        #    x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(2 if isEncoder else 1)
        batch_size=x.size(2 if isEncoder else 1)
        pos_table_v,pos_table_h = self.pos_table_v[:,:seq_len].clone().detach(),self.pos_table_h[:,:x.size(1)].clone().detach()
        if x.is_cuda:
            pos_table_v.cuda()
            pos_table_h.cuda()
        if isEncoder==True:
            x=x.transpose(1,2)+pos_table_h
            x = x.transpose(1,2)
        x = x + pos_table_v
        return x

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200,Attr_BSM=True):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding_Sphere3D(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout,BSM=Attr_BSM)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        
        enc_output = self.dropout(self.position_enc(self.src_word_emb(src_seq)))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab,utt_length, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1,Attr_BSM=True):

        super().__init__()
        self.scaleF=ScaleFeedForward(d_model*utt_length,d_model, dropout=dropout)
        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model,utt_length, d_inner*utt_length, n_head, d_k, d_v, dropout=dropout,BSM=Attr_BSM)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        utt_length=enc_output.size()[1]
        
        dec_slf_attn_list, dec_enc_attn_list = [], []
        
        # -- Forward
        
        dec_output = self.dropout(self.position_enc(self.trg_word_emb(trg_seq), isEncoder=False))
        dec_output_3d=dec_output.repeat(utt_length,1,1,1).transpose(0,1)
        for dec_layer in self.layer_stack:
            dec_output_3d, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output_3d, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn] if return_attns else []
                dec_enc_attn_list += [dec_enc_attn] if return_attns else []
        
        dec_output_3d = self.layer_norm(dec_output_3d)
        
        dec_output_3d = self.scaleF(dec_output_3d)
        if return_attns:
            return dec_output_3d, dec_slf_attn_list, dec_enc_attn_list
        
        return dec_output_3d,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab,utt_length, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True, BSM=True):

        super().__init__()
        
        
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        
        d_k=d_v=d_model//n_head
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout,Attr_BSM=BSM)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab,utt_length=utt_length, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout,Attr_BSM=BSM)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        self.x_logit_scale = 1.
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq):
        
        
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        #utt_length=src_seq.size()[1]
        #trg_seq_3d=trg_seq.repeat(utt_length,1,1).transpose(0,1)     
        
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
        
        
        enc_output,  enc_slf_attn_list = self.encoder(src_seq, src_mask, return_attns=True)
     
        src_mask_decoder= get_pad_mask(src_seq, self.src_pad_idx)

        dec_output_3d, _, att = self.decoder(trg_seq, trg_mask, enc_output, src_mask_decoder, return_attns=True)
        
        seq_logit = self.trg_word_prj(dec_output_3d) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))
