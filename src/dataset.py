import torch as th
import torchaudio
import numpy as np
from sklearn.preprocessing import StandardScaler

class AudioMelDataset:
    def __init__(self, sampling_rate=24000, n_fft=1024, hop_length=256, n_mels=100):
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.scaler = StandardScaler()
        self.mel_mean = None
        self.mel_std = None
        self.mel_transformer =  torchaudio.transforms.MelSpectrogram(
            sample_rate= self.sampling_rate,
            n_fft= self.n_fft,
            hop_length= self.hop_length,
            n_mels= self.n_mels
        )
    
    def get_mel_spectogram(self, waveform: th.tensor, device:th.device) -> th.tensor:
        '''
        Utility function to get logarithmic mel_spectogram from any waveform based on the parameters defined above.
        final shape of mel_spectogram -> [T, n_mels]
        '''
        transformer = self.mel_transformer.to(device)
        waveform = waveform.to(device)
        mel_spec = transformer(waveform)        ## here shape will be [1,n_mels, T]
        mel_spec = mel_spec.squeeze(0).transpose(0,1)       ## new shape -> [T, n_mels]
        return th.log(mel_spec + 1e-5)
    
    def prepare_training_data(self, num_files = 500, root_dir = "."):
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        dataset = torchaudio.datasets.LJSPEECH(root=root_dir, download=True)
        all_mels = []
        limit = min(num_files, len(dataset))
        for i in range(limit):
            wave_form, sr, _, _ = dataset[i]
            if sr!= self.sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate).to(device)       ## resample to match the freq expected by mel_spectogram convertor
                wave_form = resampler(wave_form.to(device))
            
            with th.no_grad():
                mel_i = self.get_mel_spectogram(waveform=wave_form, device=device)
                all_mels.append(mel_i.cpu())
            
        training_data = th.cat(all_mels, dim=0).numpy()         ## shape of training data -> [t, n_mels] where t = sum(T)
        training_data_scaled = self.scaler.fit_transform(training_data)

        self.mel_mean = th.tensor(self.scaler.mean_ ,  dtype=th.float32).to(device)             ## shape [n_mels]
        self.mel_std = th.tensor(self.scaler.scale_ ,  dtype=th.float32).to(device)

        return training_data_scaled

