import torch
import torch.utils.benchmark as benchmark

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

def run_dense_fib(N, pairs):
    # Initialize the blank array
    mask = torch.zeros((N, N))
    
    # 289 iterations of pure integer drops
    for r_origin, c_origin in pairs:
        for r_path, c_path in pairs:
            r = r_origin + r_path
            c = c_origin + c_path
            if r < N and c < N:
                mask[r, c] = 1.0
                mask[r, N - 1 - c] = 1.0
                mask[N - 1 - r, c] = 1.0
                mask[N - 1 - r, N - 1 - c] = 1.0
    return mask

def run_native_blackman(N):
    b = torch.blackman_window(N)
    return torch.outer(b, b)

def main():
    N = 1024 # 1,048,576 pixels
    
    # We precompute the sequence pairs up to 1024 (there are only 17 of them)
    pairs = generate_fib_pairs(N)
    
    t_blackman = benchmark.Timer(
        stmt='run_native_blackman(N)', 
        setup='from __main__ import run_native_blackman', 
        globals={'N': N}
    )
    
    t_fib = benchmark.Timer(
        stmt='run_dense_fib(N, pairs)', 
        setup='from __main__ import run_dense_fib', 
        globals={'N': N, 'pairs': pairs}
    )

    print("Formal Hardware Load Test (1 Million Pixels/1024x1024)\n")
    
    print("1. PyTorch Native C++ ATen Blackman:")
    print(t_blackman.timeit(100))

    print("\n2. Pure Python Fibonacci Integer Offset:")
    print(t_fib.timeit(100))

if __name__ == '__main__':
    main()
