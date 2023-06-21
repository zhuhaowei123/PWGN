#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import data.PWGN.util as util
import os
import numpy as np
import torch
import scipy.io


# In[ ]:


class NSDTSEADataset():

    def __init__(self, config, model):

        self.model = model
        self.path = config['dataset']['path']
        self.file_paths = {'train': {'clean': [], 'noisy': []}, 'test': {'clean': [], 'noisy': []}}
        self.sequences = {'train': {'clean': [], 'noisy': []}, 'test': {'clean': [], 'noisy': []}}
        self.voice_indices = {'train': [], 'test': []}
        self.speakers = {'train': [], 'test': []}
        self.speaker_mapping = {}
        self.batch_size = config['training']['batch_size']
        self.noise_only_percent = config['dataset']['noise_only_percent']
        self.in_memory_percentage = config['dataset']['in_memory_percentage']
        self.num_sequences_in_memory = 0
        self.condition_encode_function = util.get_condition_input_encode_func(config['model']['condition_encoding'])

    def load_dataset(self):

        print('Loading NSDTSEA dataset...')

        for Set in ['train', 'test']:
            for condition in ['clean', 'noisy']:
                current_directory = os.path.join(self.path, condition + '_' + Set + 'set')

                sequences, file_paths, speakers = self.load_directory(current_directory, condition)
                self.file_paths[Set][condition] = file_paths
                self.speakers[Set] = speakers
                self.sequences[Set][condition] = sequences



        return self

    def load_directory(self, directory_path, condition):

        filenames = [filename for filename in sorted(os.listdir(directory_path)) if filename.endswith('.mat')]

        speakers = []
        file_paths = []
        # speech_onset_offset_indices = []
        # regain_factors = []
        sequences = []
        for filename in filenames:

            speaker_name = filename[0:5]
            speakers.append(speaker_name)
            filepath_clean = directory_path + "/" + filename
            filepath_noisy = directory_path.replace("clean", "data") + "/" + filename.replace("clean", "data")


            if condition == 'clean':
                # sequence = util.load_wav(filepath, self.sample_rate)
                sequence = scipy.io.loadmat(filepath_clean)['result']
                sequences.append(sequence)
                self.num_sequences_in_memory += 1

            else:
                if self.in_memory_percentage == 1 or np.random.uniform(0, 1) <= (self.in_memory_percentage-0.5)*2:
                    sequence = scipy.io.loadmat(filepath_noisy)['result']
                    sequences.append(sequence)
                    self.num_sequences_in_memory += 1
                else:
                    sequences.append([-1])

            if speaker_name not in self.speaker_mapping:
                self.speaker_mapping[speaker_name] = len(self.speaker_mapping) + 1

            file_paths.append(filepath_clean)

        return sequences, file_paths, speakers #speech_onset_offset_indices,regain_factors

    def get_num_sequences_in_dataset(self):
        return len(self.sequences['train']['clean']) + len(self.sequences['train']['noisy']) + len(self.sequences['test']['clean']) + len(self.sequences['test']['noisy'])

    def retrieve_sequence(self, Set, condition, sequence_num):    #sequence_num表示第几条数据

        if len(self.sequences[Set][condition][sequence_num]) == 1:
            sequence = self.sequences[Set][condition][sequence_num]

            if (float(self.num_sequences_in_memory) / self.get_num_sequences_in_dataset()) < self.in_memory_percentage:
                self.sequences[Set][condition][sequence_num] = sequence
                self.num_sequences_in_memory += 1
        else:
            sequence = self.sequences[Set][condition][sequence_num]

        return np.array(sequence)

    def get_random_batch_generator(self, Set):

        if Set not in ['train', 'test']:
            raise ValueError("Argument SET must be either 'train' or 'test'")

        while True:

            sample_indices = np.random.randint(0, len(self.sequences[Set]['clean']), self.batch_size)
            batch_inputs = []
            batch_outputs_1 = []
            batch_outputs_2 = []
            for i, sample_i in enumerate(sample_indices):  #循环10次（batch_size=10）



                speech = self.retrieve_sequence(Set, 'clean', sample_i)
                noisy = self.retrieve_sequence(Set, 'noisy', sample_i)
                noise = noisy - speech
                Input = np.squeeze(speech + noise)
                output_speech = np.squeeze(speech)
                output_noise = np.squeeze(noise) #修改
                if self.noise_only_percent > 0:
                    if np.random.uniform(0, 1) <= self.noise_only_percent:
                        Input = output_noise #Noise only
                        output_speech = np.array([0] * self.model.input_length) #Silence

                batch_inputs.append(Input)
                batch_outputs_1.append(output_speech)
                batch_outputs_2.append(output_noise)

            batch_inputs = np.array(batch_inputs, dtype='float32')
            batch_outputs_1 = np.array(batch_outputs_1, dtype='float32')
            batch_outputs_2 = np.array(batch_outputs_2, dtype='float32')
            batch_outputs_1 = batch_outputs_1[:, self.model.get_padded_target_field_indices()]
            batch_outputs_2 = batch_outputs_2[:, self.model.get_padded_target_field_indices()]

            batch = {'data_input': batch_inputs}, {'data_output_1': batch_outputs_1, 'data_output_2': batch_outputs_2}
            #'data_output_1' 是干净的（即目标）波形,一组10个，（10*789）

            yield batch



# In[ ]:


class denoising_dataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator
    def __iter__(self):
        return self.generator

