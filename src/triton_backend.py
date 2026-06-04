import torch as th
import os
import triton
import triton.language as tl

@triton.jit                                 ## This decorator compiles this function in an optimised GPU kernel in runtime
def rvq_decode_kernel(
    packed_ptr,                             ## Pointer to the array of uint64 packed_indices numbers [n_frames]
    codebook_ptr,                           ## Pointer to the 8 layered codebook. [8, n_clusters, n_mels]
    output_ptr,                             ## Pointer to the output mel_spectogram matrix [n_frames, n_mels]
    n_frames,                               ## no of frames or time stamps per mel
    BLOCK_SIZE : tl.constexpr,              ## Run time constant. Defined the memory block size each instance of this function will grab from VRAM to VRAM
    n_mels: int,
    n_layers:int,
    codebook_size:int
):
    ## Identify the pid for the kernel instance
    pid = tl.program_id(0)

    ## Identify the memory offset for this kernel
    offsets = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)             ## This will give each instance a window of memory to extract from VRAM
    mask = offsets < n_frames                                       ## If n_frames isnt divisible by BLOCK_SIZE, the last block will over flow the n_frames. This mask will help prevent this
    packed_val = tl.load(packed_ptr + offsets, mask = mask)        ## This helps the instance fetch the memory window from VRAM to SRAM. This happens simultaneously for all instances

    for m in range(n_mels):
        accumulator = tl.full([BLOCK_SIZE], 0.0, dtype=tl.float32)
        for i in range(n_layers):
            idx = (packed_val >> (i * 8)) & 0xFF
            cb_offset = (i * codebook_size * n_mels) + (idx * n_mels) + m
            val = tl.load(codebook_ptr + cb_offset, mask=mask)
            accumulator += val
        output_write_ptr = output_ptr + (offsets * n_mels + m)
        tl.store(output_write_ptr, accumulator, mask=mask)


def triton_decode(packed_data: th.Tensor, stacked_codebooks: th.Tensor, device: th.device) -> th.Tensor:
    """
    Host-side launcher for the Triton decoding kernel.
    Handles grid sizing, memory allocation, and type checking.
    """
        
    # Triton kernels require explicit 64-bit integers for packed token manipulation
    packed_tensor = packed_data.to(device).view(th.int64)
    n_frames = packed_tensor.shape[0]
    n_mels = stacked_codebooks.shape[2]
    n_layers = stacked_codebooks.shape[0]
    codebook_size = stacked_codebooks.shape[1]
    
    # Pre-allocate the destination tensor on the GPU
    output = th.zeros((n_frames, n_mels), device=device, dtype=th.float32)
    
    # Executive execution configuration
    BLOCK_SIZE = 128
    
    # Grid function: calculates how many parallel GPU blocks are required to cover all frames.
    # triton.cdiv performs ceiling division (e.g., if n_frames = 129, cdiv(129, 128) = 2 blocks).
    grid = lambda meta: (triton.cdiv(n_frames, meta['BLOCK_SIZE']),)
    
    # Launch the kernel asynchronously onto the GPU pipeline
    rvq_decode_kernel[grid](
        packed_tensor,
        stacked_codebooks,
        output,
        n_frames,
        BLOCK_SIZE=BLOCK_SIZE, # Passed directly as a compile-time tl.constexpr
        n_mels = n_mels,
        n_layers =  n_layers,
        codebook_size = codebook_size
    )
    return output