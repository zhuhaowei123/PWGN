#!/usr/bin/env python
# coding: utf-8

# In[ ]:




from __future__ import division
import os
import data.PWGN.util as util
import tqdm
import numpy as np
import torch
import scipy.io as scio


# In[ ]:


def denoise_sample(predict_config, inputs, condition_input, batch_size, output_filename_prefix, output_folder_path):
    model = predict_config.get_trained_model()

    # if len(inputs['noisy']) < model.receptive_field_length:
    #     raise ValueError('Input is not long enough to be used with this model.')
    # num_output_samples = inputs['noisy'].shape[0] - (model.receptive_field_length - 1)
    length = len(inputs['noisy'])
    num_output_samples = inputs['noisy'].shape[0] - (length - 1)
    num_fragments = int(np.ceil(num_output_samples / length)) #np.ceil 计算大于等于该值的最小整数
    num_batches = int(np.ceil(num_fragments / batch_size))

    denoised_output = []
    noise_output = []
    # distance = []
    # num_pad_values = 0
    fragment_i = 0
    for batch_i in tqdm.tqdm(range(0, num_batches)):

        if batch_i == num_batches-1: #If its the last batch'
            batch_size = num_fragments - batch_i*batch_size

        # condition_batch = np.array([condition_input, ] * batch_size, dtype='uint8')
        input_batch = np.zeros((batch_size, length))

        #Assemble batch
        for batch_fragment_i in range(0, batch_size):

            if fragment_i + length > num_output_samples:
                remainder = inputs['noisy'][fragment_i:]
                # current_fragment = np.zeros((model.input_length,))
                current_fragment = np.zeros((length,))
                current_fragment[:remainder.shape[0]] = np.squeeze(remainder)
                # num_pad_values = model.input_length - remainder.shape[0]
            else:
                current_fragment = inputs['noisy'][fragment_i:fragment_i + length]

            input_batch[batch_fragment_i, :] = current_fragment
            fragment_i += length

        denoised_output_fragments = predict_config.denoise_batch({'data_input': torch.from_numpy(input_batch)})
        # 'condition_input': torch.from_numpy(condition_batch)
        # print(b)
        denoised_output_fragments = list(denoised_output_fragments)
        # distance = denoised_output_fragments[2]
        if type(denoised_output_fragments) is list:
            noise_output_fragment = denoised_output_fragments[1]
            denoised_output_fragment = denoised_output_fragments[0]


        # denoised_output_fragment = denoised_output_fragment[:, model.target_padding: model.target_padding + length]
        denoised_output_fragment = denoised_output_fragment.reshape(-1).tolist()

        if noise_output_fragment is not None:
            noise_output_fragment = noise_output_fragment[:, model.target_padding: model.target_padding + length]
            noise_output_fragment = noise_output_fragment.reshape(-1).tolist()

        if type(denoised_output_fragments) is float:
            denoised_output_fragment = [denoised_output_fragment]
        if type(noise_output_fragment) is float:
            noise_output_fragment = [noise_output_fragment]

        denoised_output = denoised_output + denoised_output_fragment
        noise_output = noise_output + noise_output_fragment

    denoised_output = np.array(denoised_output)

    valid_noisy_signal = inputs['noisy']

    if inputs['clean'] is not None:
        inputs['noise'] = inputs['noisy'] - inputs['clean']


        valid_clean_signal = inputs['clean']

        noise_in_denoised_output = denoised_output - valid_clean_signal


        rms_noise_out = util.rms(noise_in_denoised_output)
        rms_noise_in = util.rms(inputs['noise'])
        print('rms_noise_out', rms_noise_out, 'rms_noise_in', rms_noise_in)

        output_clean_filename = output_filename_prefix + 'clean.mat'
        output_clean_filepath = os.path.join(output_folder_path, output_clean_filename)

        scio.savemat(output_clean_filepath, {'clean': inputs['clean']})
        output_denoised_filename = output_filename_prefix + 'denoised.mat'
        output_noisy_filename = output_filename_prefix + 'noisy.mat'
    else:
        output_denoised_filename = output_filename_prefix + 'denoised.mat'
        output_noisy_filename = output_filename_prefix + 'noisy.mat'


    output_denoised_filepath = os.path.join(output_folder_path, output_denoised_filename)
    output_noisy_filepath = os.path.join(output_folder_path, output_noisy_filename)
    scio.savemat(output_denoised_filepath , {'denoised':denoised_output})
    scio.savemat(output_noisy_filepath, {'noisy': valid_noisy_signal})
# In[ ]:




