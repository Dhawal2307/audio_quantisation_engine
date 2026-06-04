import torch as th
import time 
import numpy as np
from src.triton_backend import triton_decode

def calculate_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    noise = original - reconstructed
    signal_power = np.mean(original**2)
    noise_power = np.mean(noise**2)
    
    if noise_power == 0:
        return float('inf')
    snr= float(10*np.log10(signal_power/noise_power))
    return snr

def calculate_mse(original:np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.mean((original-reconstructed)**2))

def run_performance_benchmarks(packed_streams: np.ndarray, stacked_codebooks: th.Tensor, device: th.device, iterations:int = 50)-> float:
    packed_tensors = th.from_numpy(packed_streams)

    ## Hardware warmup loop to achieve steady state execution
    for _ in range(5):
        _ = triton_decode(packed_tensors, stacked_codebooks, device)
    
    ## synchronize the kernels to ensure our clocks starts exactly at 0
    if device.type == 'cuda':
        th.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(iterations):
        _ = triton_decode(packed_tensors, stacked_codebooks, device)
    
    if device.type == 'cuda':
        th.cuda.synchronize()
    end_time = time.perf_counter()

    triton_avg_latency = (end_time - start_time)/ iterations
    print(f"Triton Kernel Average Latency: {triton_avg_latency * 1000:.4f} ms")
    return triton_avg_latency

