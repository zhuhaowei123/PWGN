#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import json
# import librosa
import torch
import soundfile as sf
import warnings


# In[ ]:


def l1_l2_loss(y_true, y_pred, l1_weight, l2_weight):
    loss = 0

    if l1_weight != 0:
        loss += l1_weight*torch.nn.L1Loss()(y_true, y_pred)

    if l2_weight != 0:
        loss += l2_weight * torch.nn.MSELoss()(y_true, y_pred)

    return loss


# In[ ]:


def compute_receptive_field_length(stacks, dilations, filter_length, target_field_length):

    half_filter_length = (filter_length-1)/2 #（3-1）/2=1
    length = 0
    for d in dilations:
        length += d*half_filter_length #2的0次到10次的累加
    length = 2*length #2的1次到11次的累加
    length = stacks * length #3倍
    length += target_field_length #+1
    return length


# In[ ]:


def one_hot_encode(x, num_values=256):
    if isinstance(x, int):
        x = np.array([x])
    if isinstance(x, list):
        x = np.array(x)
    return np.eye(num_values, dtype='uint8')[x.astype('uint8')]


# In[ ]:


def one_hot_decode(x):
    return np.argmax(x, axis=-1)


# In[ ]:


def binary_encode(x, max_value):
    if isinstance(x, int):
        x = np.array([x])
    if isinstance(x, list):
        x = np.array(x)
    width = np.ceil(np.log2(max_value)).astype(int) #ceil计算大于等于改值的最小整数
    return (((x[:, None] & (1 << np.arange(width)))) > 0).astype(int)


# In[ ]:


def get_condition_input_encode_func(representation):
    if representation == 'binary':
        return binary_encode
    else:
        return one_hot_encode


# In[ ]:


def ensure_keys_in_dict(keys, dictionary):
    if all (key in dictionary for key in keys):
        return True
    return False


# In[ ]:


def get_subdict_from_dict(keys, dictionary):
    return dict((k, dictionary[k]) for k in keys if k in dictionary)





# In[ ]:


def rms(x):
    return np.sqrt(np.mean(np.square(x), axis=-1))


# In[ ]:


def normalize(x):
    max_peak = np.max(np.abs(x))
    return x / max_peak








def dir_contains_files(path):
    file_list = os.listdir(path)
    if not len(file_list)==0:
        return True
    else: 
        return False

def snr_db(rms_amplitude_A, rms_amplitude_B):
    return 20.0*np.log10(rms_amplitude_A/rms_amplitude_B)




