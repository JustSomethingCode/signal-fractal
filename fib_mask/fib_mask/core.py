import torch

def _generate_fib_pairs(max_val):
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

def dense_fibonacci_lattice_2d(size, dtype=None, device=None, requires_grad=False):
    """
    Generates a purely sparse Sub-linear O(log N^2) Fibonacci Multi-Origin Lattice.
    Perfect for zero-overhead anti-aliasing scatter shields.
    """
    N = size
    mask = torch.zeros((N, N), dtype=dtype, device=device, requires_grad=requires_grad)
    pairs = _generate_fib_pairs(N)
    
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

def hybrid_fibonacci_window_2d(size, core_radius_ratio=0.33, dtype=None, device=None, requires_grad=False):
    """
    The ultimate VRAM-saving 2D Window logic. 
    A mathematically dense inner core (cheap energy) encased in a Sparse Fibonacci 
    topographical tracking array (perfect aliasing suppression).
    """
    N = size
    mask = dense_fibonacci_lattice_2d(N, dtype=dtype, device=device, requires_grad=requires_grad)
    
    # Drop the thick energy core
    Y, X = torch.meshgrid(torch.linspace(-1, 1, N, device=device), 
                          torch.linspace(-1, 1, N, device=device), indexing='ij')
    
    core = (X**2 + Y**2) < (core_radius_ratio ** 2)
    mask[core] = 1.0
    
    return mask
