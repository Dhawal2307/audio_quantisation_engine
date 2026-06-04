import torch as th
import numpy as np
from src.model import ResidualVectorQuantizer
from src.evaluate import run_performance_benchmarks, calculate_mse, calculate_snr
from src.triton_backend  import triton_decode
from src.dataset import AudioMelDataset
import torchaudio
import time

def main():

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print("Device found is ", device)

    dataset_obj = AudioMelDataset()
    quantizer = ResidualVectorQuantizer()       
    training_data = dataset_obj.prepare_training_data()                         ## this is our training data
    quantizer.fit(training_data_scaled=training_data, device =device)           ## shape of quantizer.all_cb -> [8, 256, 100]
    stacked_codebooks = quantizer.codebooks.float()                      ## Stack the all_codebooks to get one matrix


    ## Now we will pick a audio datapoint and run evaluations on it
    ljsSpeechdatset = torchaudio.datasets.LJSPEECH(root= "./", download=True)
    waveform ,sr , _, _ = ljsSpeechdatset[23]            ## Pick an arbitraty data point

    if sr != dataset_obj.sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq= sr, new_freq=dataset_obj.sampling_rate)
        waveform = resampler(waveform)

    ## extract the log mel-spectogram for this sample
    log_mel_original = dataset_obj.get_mel_spectogram(waveform=waveform, device=device)         ## shape is [t, n_mels]
    n_frames_i = log_mel_original.shape[0]                                                      ## store the number of frames for this audio sample
    log_mel_original = log_mel_original.cpu().numpy()                                           ## change this to np.ndarray to scale 
    scaled_mel = dataset_obj.scaler.transform(log_mel_original)

    ## Compress this scaled log-mel_spectorgram
    cb_indices_i = quantizer.get_indices(mel_scaled=scaled_mel)                                 ## shape here is [8,n_frames_i]
    packed_indices_i = quantizer.pack_indices(cb_indices_i)                                     ## shape here is [n_frames_i]

    ## Decode these indices and unpack the mel_spectogram. 
    print("Decoding the packed audio indices and performing benchmark tests")


    avg_latency_sec = run_performance_benchmarks(
        packed_streams=packed_indices_i, 
        stacked_codebooks=stacked_codebooks, 
        device=device, 
        iterations=50
    )
    decode_latency_ms = avg_latency_sec * 1000

    packed_indices_i_tensor = th.from_numpy(packed_indices_i)
    decoded_scaled_mel = triton_decode(packed_indices_i_tensor, stacked_codebooks, device= device).cpu().numpy()

    inverse_transformed_decoded_mel = dataset_obj.scaler.inverse_transform(decoded_scaled_mel)

    mse_score = calculate_mse(log_mel_original, inverse_transformed_decoded_mel)
    snr_score = calculate_snr(log_mel_original, inverse_transformed_decoded_mel)

    seconds_per_frame = dataset_obj.hop_length / dataset_obj.sampling_rate
    total_audio_duration = n_frames_i * seconds_per_frame
    rtf_score = avg_latency_sec / total_audio_duration

    print("\n" + "="*40)
    print("Final Metrics")
    print("="*40)

    print(f"  Audio Duration      : {total_audio_duration:.2f} seconds")
    print(f"  Decoding Latency    : {decode_latency_ms:.2f} ms")
    print(f"  Real-Time Factor    : {rtf_score:.5f} sec/sample")
    print(f"  Signal-to-Noise     : {snr_score:.2f} dB SNR")
    print(f"  Mean Squared Error  : {mse_score:.4f} MSE")
    print("="*40)

if __name__ == "__main__":
    main()


   










