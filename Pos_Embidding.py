# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:34:22 2020

@author: belainine
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
#plt.switch_backend('agg')
#import matplotlib.ticker as ticker

hidden_units = 100 # Dimension of embedding
vocab_size = 10 # Maximum sentence length
# Matrix of [[1, ..., 99], [1, ..., 99], ...]
i = np.tile(np.expand_dims(range(hidden_units), 0), [vocab_size, 1])
# Matrix of [[1, ..., 1], [2, ..., 2], ...]
pos = np.tile(np.expand_dims(range(vocab_size), 1), [1, hidden_units])
# Apply the intermediate funcitons
pos = np.multiply(pos, 1/10000.0)
i = np.multiply(i, 2.0/hidden_units)
matrix = np.power(pos, i)

n_position=100
d_pos_vec=100
position_enc = np.array([
    [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
    if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

position_enc_h = np.array([
    [pos / np.power(10000, ((2*j)%50)/d_pos_vec) for j in range(d_pos_vec)]
    if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])
position_enc = np.array([
    [pos / np.power(10000, ((2*j))/d_pos_vec) for j in range(d_pos_vec)]
    if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])
position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
# Apply the sine function to the even colums


position_enc_h[1:, 0::2] = np.sin(position_enc_h[1:, 0::2]) # dim 2i
position_enc_h[1:, 1::2] = np.cos(position_enc_h[1:, 1::2]) # dim 2i+1
matrix[:, 1::2] = np.sin(matrix[:, 1::2]) # even
# Apply the cosine function to the odd columns
matrix[:, ::2] = np.cos(matrix[:, ::2]) # odd
# Plot
position_enc_v=position_enc
position_enc_h=position_enc_h
end_indxs=[[k for k in range(20)] for i in range(int(d_pos_vec/20))]
plt.rcParams["figure.figsize"] = (10,10)
fig = plt.figure(figsize=(7, 9))
fig, (ax,ax1) = plt.subplots(nrows=2)
im1 = ax.imshow(position_enc_h[0:40], cmap='ocean', aspect='auto')
#im = ax.imshow(position_enc_h[0:40], cmap=cm.coolwarm, aspect='auto')
ax.set_xlabel('Max sequence length',fontsize=15)
ax.set_ylabel('Embedding dimension',fontsize=15)
ax.set_title(r'Horizontal Position',fontsize=15)
fig.colorbar(im1, shrink=1,  ax=ax)

#------------------------------------
im = ax1.imshow(position_enc_v[0:40], cmap='ocean', aspect='auto')
#im = ax1.imshow(position_enc_v[0:40], cmap=cm.coolwarm, aspect='auto')
ax1.set_xlabel('Max sequence length',fontsize=15)
ax1.set_ylabel('Embedding dimension',fontsize=15)
ax1.set_title(r'Vertical Position',fontsize=15)
fig.colorbar(im, shrink=1,  ax=ax1)
# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
fig.savefig("test.png")
plt.show()