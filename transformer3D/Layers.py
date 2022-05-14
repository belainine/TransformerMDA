''' Define the Layers '''
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformer.SubLayers import MultiHeadAttention, MultiHeadAttentionBSM2D, MultiHeadAttentionBSM3D, PositionwiseFeedForward,PositionwiseFeedForward3D,ScaleFeedForward


__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1,BSM=True):
        super(EncoderLayer, self).__init__()
        if BSM==True:
            self.slf_attn = MultiHeadAttentionBSM3D(n_head, d_model, d_k, d_v, dropout=dropout)
        else:
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model,utt_length, d_inner, n_head, d_k, d_v, dropout=0.1,BSM=True):
        super(DecoderLayer, self).__init__()
        if BSM==True:
            self.slf_attn = MultiHeadAttentionBSM3D(n_head, d_model, d_k, d_v, dropout=dropout)
            self.enc_attn = MultiHeadAttentionBSM3D(n_head, d_model, d_k, d_v, dropout=dropout)
        else:
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
            self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.utt_length=utt_length
        
    def forward(
            self, dec_input_3d, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        
        utt_length=self.utt_length
        

        dec_output_3d, dec_slf_attn = self.slf_attn(
            dec_input_3d, dec_input_3d, dec_input_3d, mask=slf_attn_mask.unsqueeze(1))

        dec_output_3d, dec_enc_attn = self.enc_attn(
            dec_output_3d, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output_3d)
        
        #dec_output = dec_output_3d.sum(1)
        return dec_output, dec_slf_attn, dec_enc_attn
