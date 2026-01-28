import sys
sys.path.insert(0, 'src/entropy_core')
import entropy_core as ec
import numpy as np
import time

# Fix the import path FIRST (this is what was failing)
sys.path.insert(0, 'src/entropy_core')  # Points directly to .pyd directory

def benchmark_comprehensive():
    sizes = [
        (100, 100),
        (200, 200),
        (500, 500),
        (700, 700),
        (1000, 1000),
        (2000, 2000),
    ]
    unit_cell_size = 1  # 1 qubit per site
    n_iterations = 10
    
    print("=" * 70)
    print("COMPREHENSIVE LATTICE + ENTROPY PERFORMANCE TEST")
    print("=" * 70)
    
    results = []
    
    for rows, cols in sizes:
        total = rows * cols * unit_cell_size
        shape = [rows, cols]
        
        print(f"\nTesting {rows}×{cols}...", end=" ", flush=True)
        
        try:
            entropy = ec.FastLatticeEntropy(shape, unit_cell_size)
            
            # Warmup
            for _ in range(3):
                _ = entropy.get_all_qubits_pure()
                _ = entropy.get_all_qubits_with_noise()
                _ = entropy.generate_noise_only(total)
            
            # Benchmark pure lattice
            lat_times = []
            for _ in range(n_iterations):
                t1 = time.perf_counter()
                lattice = entropy.get_all_qubits_pure()
                t2 = time.perf_counter()
                lat_times.append((t2 - t1) * 1000)
            
            lat_mean = np.mean(lat_times)
            lat_std = np.std(lat_times)
            
            # Benchmark lattice + noise
            combined_times = []
            for _ in range(n_iterations):
                t1 = time.perf_counter()
                noisy = entropy.get_all_qubits_with_noise()
                t2 = time.perf_counter()
                combined_times.append((t2 - t1) * 1000)
            
            combined_mean = np.mean(combined_times)
            combined_std = np.std(combined_times)
            
            # Benchmark noise only
            noise_times = []
            for _ in range(n_iterations):
                t1 = time.perf_counter()
                noise = entropy.generate_noise_only(total)
                t2 = time.perf_counter()
                noise_times.append((t2 - t1) * 1000)
            
            noise_mean = np.mean(noise_times)
            noise_std = np.std(noise_times)
            
            overhead_pct = (noise_mean / lat_mean) * 100 if lat_mean > 0 else 0
            ns_per_gaussian = (noise_mean * 1e6) / total if total > 0 else 0
            
            status = "✓ EXCELLENT" if overhead_pct < 50 else "✓ GOOD" if overhead_pct < 150 else "✓ OK"
            
            results.append({
                'size': f"{rows}×{cols}",
                'total': total,
                'lat_mean': lat_mean,
                'noise_mean': noise_mean,
                'combined_mean': combined_mean,
                'overhead': overhead_pct,
                'ns_per_sample': ns_per_gaussian,
                'status': status
            })
            
            print(f"✓ Done")
            print(f"  Pure lattice:     {lat_mean:.3f} ms (±{lat_std:.3f})")
            print(f"  Noise only:       {noise_mean:.3f} ms (±{noise_std:.3f})")
            print(f"  Full pipeline:    {combined_mean:.3f} ms (±{combined_std:.3f})")
            print(f"  Overhead %:       {overhead_pct:.1f}%")
            print(f"  ns/Gaussian:      {ns_per_gaussian:.2f} ns")
            print(f"  Status:           {status}")
            
        except Exception as e:
            print(f"✗ FAILED: {e}")
            print(f"  Skipping larger sizes...")
            break
    
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY TABLE")
        print("=" * 70)
        print(f"{'Size':<12} {'Qubits':<12} {'Lattice':<10} {'Noise':<10} {'Overhead':<10} {'Status':<15}")
        print("-" * 70)
        
        for r in results:
            qubits_fmt = f"{r['total']:,}"
            print(f"{r['size']:<12} {qubits_fmt:<12} {r['lat_mean']:>7.2f} ms {r['noise_mean']:>7.2f} ms "
                  f"{r['overhead']:>7.1f}%  {r['status']:<15}")
        
        print("=" * 70)
        print("\n🎉 SUCCESS METRICS:")
        print(f"  ✓ Tested up to {results[-1]['size']} ({results[-1]['total']:,} qubits)")
        print(f"  ✓ Best overhead: {min(r['overhead'] for r in results):.1f}%")
        print(f"  ✓ Avg speed: {np.mean([r['ns_per_sample'] for r in results]):.2f} ns/Gaussian")
        
        # Show scaling
        if len(results) >= 2:
            print(f"\n📊 THROUGHPUT:")
            for r in results:
                throughput = r['total'] / r['combined_mean']
                print(f"  {r['size']:<12} {throughput:>8.0f} qubits/ms")
        
        print("=" * 70)


if __name__ == "__main__":
    benchmark_comprehensive()
