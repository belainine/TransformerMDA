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

n_position=30
n_position=30
d_pos_vec=256
d_pos_vec_v=256
def get_position_angle_vec(position,d_pos_vec_v):
            return [position / np.power(10000, 3 * (hid_j // 3) / d_pos_vec_v) for hid_j in range(d_pos_vec_v)]
position_enc = np.array([
    get_position_angle_vec(pos,d_pos_vec_v)#[pos / np.power(10000, 3*i/d_pos_vec_v) for i in range(d_pos_vec_v)]
    if pos != 0 else np.zeros(d_pos_vec_v) for pos in range(n_position)])

position_enc_h = np.array([
    get_position_angle_vec(pos,d_pos_vec)#[pos / np.power(10000, ((3*(j+1)))/d_pos_vec) for j in range(d_pos_vec)]
    if pos != 0 else np.zeros(d_pos_vec) for pos in range(d_pos_vec)])
position_enc = np.array([
    [pos / np.power(10000, ((3*(j+1)))/d_pos_vec_v) for j in range(d_pos_vec_v)]
    if pos != 0 else np.zeros(d_pos_vec_v) for pos in range(n_position)])
position_enc[:, 0::3] = np.sin(position_enc[:, 0::3]) # dim 3i
position_enc[:, 1::3] = np.cos(position_enc[:, 1::3]) # dim 3i+1
position_enc[:, 2::3] = np.power(position_enc[:, 2::3],0) # dim 3i+2
# Apply the sine function to the even colums


position_enc_h[:, 0::3] = np.cos(position_enc_h[:, 0::3]) # dim 3i
position_enc_h[:, 1::3] = np.sin(position_enc_h[:, 1::3]) # dim 3i+1
position_enc_h[:, 2::3] = np.cos(position_enc_h[:, 2::3]) # dim 3i+1

# Plot
position_enc_v=position_enc
position_enc_h=position_enc_h
end_indxs=[[k for k in range(10)] for i in range(int(d_pos_vec/20))]
plt.rcParams["figure.figsize"] = (8,6)
fig = plt.figure(figsize=(7, 9))
fig, (ax,ax1) = plt.subplots(nrows=2)#plt.subplots(nrows=2)
position_enc_h = np.array([position_enc_h[i] for i in range(int(d_pos_vec))])
im1 = ax.imshow(position_enc_h[0:10], cmap='ocean', aspect='auto')
#im = ax.imshow(position_enc_h[0:40], cmap=cm.coolwarm, aspect='auto')
ax.set_ylabel('Position d\'énoncé',fontsize=12)
ax.set_xlabel('Dimension de plongement',fontsize=12)
ax.set_title(r'Position horizontale',fontsize=12)
fig.colorbar(im1, shrink=1,  ax=ax)

#------------------------------------
#position_enc_v = position_enc_v 
im = ax1.imshow(position_enc_v[0:n_position], cmap='ocean', aspect='auto')
#im = ax1.imshow(position_enc_v[0:40], cmap=cm.coolwarm, aspect='auto')
ax1.set_ylabel('Position de mot',fontsize=12)
ax1.set_xlabel('Dimension de plongement',fontsize=12)
ax1.set_title(r'Position verticale',fontsize=12)
fig.colorbar(im, shrink=1,  ax=ax1)
# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
fig.savefig("test.png")
plt.show()
plt.rcParams["figure.figsize"] = (38,5)

fig, (ax) = plt.subplots(nrows=2,ncols=5)
#------------------------------------
for i in range(10):
    position_enc_cube = position_enc_v * position_enc_h[i]/np.sqrt(256/3)
    k,j=i//5,i%5
    im = ax[k,j].imshow(position_enc_cube[0:n_position], cmap='ocean', aspect='auto')
    #im = ax1.imshow(position_enc_v[0:40], cmap=cm.coolwarm, aspect='auto')
    ax[k,j].set_ylabel('\nPosition de mot + Position d\'énoncé '+str(i),fontsize=8)
    ax[k,j].set_xlabel('Dimension de plongement',fontsize=8)
    #ax[k,j].set_title(r'Word position + utterance position '+str(i),fontsize=10)
    fig.colorbar(im, shrink=1,  ax=ax[k,j])
# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
fig.savefig("test.png")
plt.show()

fig, (ax) = plt.subplots(nrows=2,ncols=5)
#------------------------------------
for i in range(10):
    position_enc_cube = position_enc_v * position_enc_h[i]/np.sqrt(256/3)
    k,j=i//5,i%5
    im = ax[k,j].imshow(np.dot(position_enc_cube,position_enc_cube.T), cmap='ocean', aspect='auto')
    #im = ax1.imshow(position_enc_v[0:40], cmap=cm.coolwarm, aspect='auto')
    ax[k,j].set_ylabel('\nPosition de mot + Position d\'énoncé '+str(i),fontsize=8)
    ax[k,j].set_xlabel('Dimension de plongement',fontsize=8)
    #ax[k,j].set_title(r'Word position + utterance position '+str(i),fontsize=10)
    fig.colorbar(im, shrink=1,  ax=ax[k,j])
# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
fig.savefig("test.png")
plt.show()
