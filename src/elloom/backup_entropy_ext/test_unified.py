import time
import numpy as np
import fast_lattice_entropy as fle

print("="*70)
print("UNIFIED SIMD + ENTROPY PERFORMANCE TEST")
print("="*70)

sizes = [(100, 100, 20000), (500, 500, 500000)]

for size_x, size_y, n_qubits in sizes:
    lattice = fle.FastLatticeEntropy([size_x, size_y], 2)
    
    # Determine number of iterations based on size
    n_iterations = max(10, 10000 // n_qubits)
    
    # WARMUP (critical for accurate timing!)
    for _ in range(5):
        _ = lattice.get_all_qubits_pure()
        _ = lattice.get_all_qubits_with_noise()
    
    # Test 1: Pure (no entropy) - AVERAGED
    times_pure = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        arr_pure = lattice.get_all_qubits_pure()
        end = time.perf_counter()
        times_pure.append((end - start) * 1000)
    
    time_pure = np.median(times_pure)  # Use median (robust to outliers)
    
    # Test 2: With entropy - SEPARATED (returns tuple)
    times_noise = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        lattice_arr, noise_arr = lattice.get_all_qubits_with_noise()
        end = time.perf_counter()
        times_noise.append((end - start) * 1000)
    
    time_noise = np.median(times_noise)
    
    # Test 3: Noise generation ONLY (to measure RNG cost separately)
    times_noise_only = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        noise_only = lattice.generate_noise_only(n_qubits)
        end = time.perf_counter()
        times_noise_only.append((end - start) * 1000)
    
    time_noise_only = np.median(times_noise_only)
    
    overhead = ((time_noise - time_pure) / time_pure) * 100
    
    print(f"\n{size_x}×{size_y} ({n_qubits} qubits, {n_iterations} iterations):")
    print(f"  Pure SIMD:        {time_pure:.3f} ms (±{np.std(times_pure):.3f})")
    print(f"  With entropy:     {time_noise:.3f} ms (±{np.std(times_noise):.3f})")
    print(f"  Noise only:       {time_noise_only:.3f} ms (±{np.std(times_noise_only):.3f})")
    print(f"  Total overhead:   {overhead:+.1f}%")
    
    # Calculate actual RNG overhead
    rng_overhead = ((time_noise_only) / time_pure) * 100
    print(f"  RNG overhead:     {rng_overhead:.1f}%")
    
    if overhead < 30:
        print(f"  ✅ LOW OVERHEAD!")
    elif overhead < 100:
        print(f"  ✓ ACCEPTABLE")
    else:
        print(f"  ⚠ High overhead")

print("\n" + "="*70)
print("BREAKDOWN:")
print("  'Pure SIMD' = just lattice generation (integers)")
print("  'With entropy' = lattice + noise generation (separated)")
print("  'Noise only' = just thermal noise generation")
print("  'RNG overhead' = cost of noise generation vs lattice")
print("="*70)
