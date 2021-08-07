# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import os
import random
import librosa
import numpy as np
import pandas as pd
import librosa.display

from typing import Any
from torch.utils.data import Dataset

class AICovidVNDataset(Dataset):
    def __init__(
        self,
        root: str='./data/aivncovid-19/',
        mode: str='train',
        eval_type: str='random',
        transform: Any=None,
        window_size: int=1,
        hop_length: int=512,
        sample_rate: int=48000,
        n_fft: int=2048, 
        masking: bool=False,
        pitch_shift: bool=False,
        breathcough=False
    ) -> None:
        super(AICovidVNDataset, self).__init__()

        self.root = root
        self.mode = mode
        self.window_size = window_size * sample_rate
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.transform = transform
        self.eval_type = eval_type
        self.masking = masking
        self.pitch_shift = pitch_shift
        self.breathcough = breathcough
        self.data_df = pd.read_csv(os.path.join(root, f'{mode}.csv'))
        print(f"Root data dir: {os.path.abspath(self.root)}")

    def __len__(self):
        return len(self.data_df.index)

    def custom_transform(self, signal):
        """
        create log spectrograph of signal
        """
        stft = librosa.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        
        if self.masking:
            log_spectrogram = self.spec_augment(log_spectrogram)
        
        if self.transform:
            log_spectrogram = self.transform(log_spectrogram)

        return log_spectrogram

    def spec_augment(
        self,
        spec: np.ndarray=None,
        num_mask: int=2,
        freq_masking_max_percentage: float=0.15,
        time_masking_max_percentage: float=0.3
    ):
        spec = spec.copy()
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)

            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = 0

            time_percentage = random.uniform(0.0, time_masking_max_percentage)

            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = 0

        return spec

    def pad(self, signal):
        sample_signal = np.zeros((self.window_size,))
        sample_signal[:signal.shape[0], ] = signal

        return sample_signal

    def __getitem__(self, index):
        # TODO: Get path of chosen index
        audio_path = self.data_df['path'].iloc[index]
        label = self.data_df['label'].iloc[index]
        chunks = self.load_process(audio_path)

        return chunks, label

    def load_process(self, audio_path):
        # TODO: Load the data
        signal, sample_rate = librosa.load(audio_path, sr=self.sample_rate)

        # TODO: Performs pitch shift
        if self.pitch_shift:
            step = np.random.uniform(-6, 6)
            signal = librosa.effects.pitch_shift(signal, sample_rate, step)

        # For `train`, sample random window size from audiofile
        if self.mode.lower() == 'train' or self.eval_type != 'maj_vote':
            # TODO: Apply padding if necessary, else samples window size
            if signal.shape[0] <= self.window_size:
                sample_signal = self.pad(signal)
            else:
                if self.eval_type == 'random':
                    rand_indx = np.random.randint(0, signal.shape[0] - self.window_size)
                else:
                    rand_indx = 0
                sample_signal = signal[rand_indx: rand_indx + self.window_size]
            
            # TODO: Performs transformations
            sample_signal = self.custom_transform(sample_signal)

            return sample_signal
            
        # For `eval/ test`, chunk audiofile into chunks of size wsz and process and return all
        else:
            chunks = np.array_split(signal, int(np.ceil(signal.shape[0] / self.window_size)))
            
            def process_chunk(chunk):
                if chunk.shape[0] <= self.window_size:
                    sample_signal = self.pad(chunk)
                
                chunk = self.custom_transform(sample_signal)

                return chunk
            
            chunks = [process_chunk(chunk) for chunk in chunks]

            return chunks
