import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.nn.functional import conv2d

def generate_fib_pairs(max_val):
    fibs = [0, 1]
    while True:
        next_fib = fibs[-1] + fibs[-2]
        if next_fib > max_val:
            break
        fibs.append(next_fib)
        
    pairs = set()
    for i in range(len(fibs)-1):
        pairs.add((fibs[i], fibs[i+1]))
        pairs.add((fibs[i+1], fibs[i]))
    return pairs

def generate_hybrid_fib_mask(N):
    mask = torch.zeros((N, N))
    pairs = generate_fib_pairs(N)
    
    # 1. The Sparse Edge Scatter (The O(1) Fibonacci Lattice)
    for r_origin, c_origin in pairs:
        for r_path, c_path in pairs:
            r = r_origin + r_path
            c = c_origin + c_path
            if r < N and c < N:
                mask[r, c] = 1.0
                mask[r, N - 1 - c] = 1.0
                mask[N - 1 - r, c] = 1.0
                mask[N - 1 - r, N - 1 - c] = 1.0
                
    # Dilation to make the abstract points function as a physical anti-aliasing web
    kernel_size = 17
    kernel = torch.ones(1, 1, kernel_size, kernel_size)
    mask = conv2d(mask.view(1, 1, N, N), kernel, padding=kernel_size//2).squeeze()
    mask = (mask > 0).float()
    
    # 2. The mathematically 'Cheap' Dense Core (The Best of Both Worlds)
    # We drop a solid 100% dense circle into the absolute center to capture 
    # massive amounts of signal energy. (Radius = N/3)
    Y, X = torch.meshgrid(torch.linspace(-1, 1, N), torch.linspace(-1, 1, N), indexing='ij')
    core = (X**2 + Y**2) < (0.33 ** 2) 
    mask[core] = 1.0
    
    return mask

def measure_leakage(tensor: torch.Tensor) -> float:
    mag = torch.abs(tensor)
    peak = mag.max().item()
    total = mag.sum().item()
    return peak / total if total != 0 else 0.0

def main():
    N = 1024
    device = 'cpu'
    
    hybrid_mask = generate_hybrid_fib_mask(N).to(device)
    
    # Baseline
    blackman_1d = torch.blackman_window(N, device=device)
    blackman_2d = torch.outer(blackman_1d, blackman_1d)
    
    fft_hybrid = torch.fft.fftshift(torch.fft.fft2(hybrid_mask))
    fft_black = torch.fft.fftshift(torch.fft.fft2(blackman_2d))
    
    e_hybrid = (hybrid_mask.sum() / (N*N)) * 100
    l_hybrid = measure_leakage(fft_hybrid)
    
    e_black = (blackman_2d.sum() / (N*N)) * 100
    l_black = measure_leakage(fft_black)
    
    print("--- Hybrid Core/Fibonacci Architecture Test ---")
    print(f"BASELINE Blackman    | Energy: {e_black:05.1f}% | Leakage: {l_black:.6f}")
    print(f"Hybrid Fib-Core      | Energy: {e_hybrid:05.1f}% | Leakage: {l_hybrid:.6f}")
    
    # Visualization Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Hybrid Architecture: Cheap Dense Core + O(1) Fibonacci Scatter Shield", fontsize=16)
    
    axes[0].imshow(hybrid_mask.cpu(), cmap='magma'); axes[0].set_title(f"Hybrid Base Mask\nEnergy: {e_hybrid:.1f}%")
    
    center = N // 2
    span = 75
    sl = slice(center - span, center + span)
    
    axes[1].imshow(torch.log1p(torch.abs(fft_hybrid[sl, sl])).cpu(), cmap='viridis')
    axes[1].set_title(f"Frequency Leakage Bloom\nLeakage: {l_hybrid:.6f}")
    
    for ax in axes.flatten():
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('hybrid_fib_test.png', dpi=200, bbox_inches='tight')
    print("Saved hybrid_fib_test.png")

if __name__ == '__main__':
    main()
