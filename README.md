# Audio Quantisation with RVQ + Triton Decoder

This project implements a residual vector quantization (RVQ) pipeline for audio mel features and a custom Triton kernel for fast GPU-side decoding.

The end-to-end flow is:

1. Load audio from LJSpeech.
2. Convert waveform to log-mel spectrogram.
3. Normalize features.
4. Train multi-layer RVQ codebooks.
5. Encode frames to packed 64-bit token stream.
6. Decode packed tokens with a Triton GPU kernel.
7. Evaluate reconstruction quality using MSE and SNR.

## Project Structure

```text
audio_quantisation/
├── requirements.txt
└── src/
    ├── dataset.py         # Audio loading, mel extraction, and scaling
    ├── model.py           # ResidualVectorQuantizer and index packing
    ├── triton_backend.py  # Custom Triton decode kernel + launcher
    ├── evaluate.py        # MSE/SNR and benchmark helpers
    └── main.py            # End-to-end training + inference demo
```

## How It Works

### 1) Feature Extraction

Input waveform is converted to a log-mel matrix with shape $[T, N_{mels}]$.

### 2) Residual Vector Quantization

The model uses 8 RVQ layers by default.
At layer $i$, k-means is trained on residuals from layer $i-1$.

### 3) Bit Packing

Each frame gets one cluster index per RVQ layer.
These indices are packed into a single 64-bit value:

$$
packed = \sum_{i=0}^{L-1}(index_i \ll 8i)
$$

where $L$ is the number of RVQ layers.

### 4) Triton Decode

The Triton kernel unpacks indices and accumulates codebook vectors in parallel on GPU, reconstructing scaled mel features.

### 5) Evaluation

The reconstructed mel is inverse-scaled and compared with original mel using:

1. Mean Squared Error (MSE)
2. Signal-to-Noise Ratio (SNR)
3. Decode latency and real-time factor (RTF)

## Requirements

Install dependencies from:

```bash
pip install -r requirements.txt
```

Typical runtime requirements:

1. Python 3.10+
2. PyTorch + torchaudio
3. scikit-learn
4. Triton-compatible environment for GPU decode path

## Run

From the `audio_quantisation` directory:

```bash
python -m src.main
```

This script:

1. Prepares training data from LJSpeech.
2. Trains RVQ codebooks.
3. Encodes one sample into packed indices.
4. Decodes with Triton kernel.
5. Prints latency, RTF, SNR, and MSE.

## Example Output

```text
========================================
Final Metrics
========================================
  Audio Duration      : 6.24 seconds
  Decoding Latency    : 0.14 ms
  Real-Time Factor    : 0.00002 sec/sample
  Signal-to-Noise     : 13.24 dB SNR
  Mean Squared Error  : 0.0412 MSE
========================================
```

## Notes

1. LJSpeech data is downloaded automatically by torchaudio when first run.
2. The Triton backend is intended for CUDA-capable GPU environments.
3. Hyperparameters (layers, clusters, mel bins) can be adjusted in source modules.