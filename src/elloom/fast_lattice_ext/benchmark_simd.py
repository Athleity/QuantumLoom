

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../fast_lattice_ext'))

# Import modules
try:
    import fast_lattice
    BASIC_AVAILABLE = True
except ImportError as e:
    print(f"❌ Basic C++ module not available: {e}")
    BASIC_AVAILABLE = False

try:
    from fast_lattice_simd import FastLatticeSIMD
    SIMD_AVAILABLE = True
except ImportError as e:
    print(f"❌ SIMD module not available: {e}")
    print("   Build it with: python setup_simd.py build_ext --inplace")
    SIMD_AVAILABLE = False


def benchmark_lattice_generation(sizes, num_iterations=100):
    """Benchmark lattice generation for different sizes"""
    
    results = {
        'sizes': [],
        'total_qubits': [],
        'basic_cpp_time': [],
        'simd_time': [],
        'speedup': []
    }
    
    print("\n" + "=" * 70)
    print("BENCHMARK: Lattice Generation")
    print("=" * 70)
    print(f"Iterations per size: {num_iterations}")
    print()
    
    for size in sizes:
        total_qubits = size[0] * size[1] * 2  # 2 basis vectors
        print(f"Testing {size[0]}×{size[1]} lattice ({total_qubits:,} qubits)...")
        
        results['sizes'].append(f"{size[0]}×{size[1]}")
        results['total_qubits'].append(total_qubits)
        
        # Benchmark Basic C++
        if BASIC_AVAILABLE:
            basic_lattice = fast_lattice.FastLattice(list(size), 2)
            
            start = time.perf_counter()
            for _ in range(num_iterations):
                qubits = basic_lattice.get_all_qubits_array()
            elapsed = (time.perf_counter() - start) / num_iterations * 1000
            
            results['basic_cpp_time'].append(elapsed)
            print(f"  Basic C++: {elapsed:.3f} ms")
        else:
            results['basic_cpp_time'].append(float('nan'))
            print(f"  Basic C++: Not available")
        
        # Benchmark SIMD
        if SIMD_AVAILABLE:
            simd_lattice = FastLatticeSIMD(list(size), 2)
            
            start = time.perf_counter()
            for _ in range(num_iterations):
                qubits = simd_lattice.get_all_qubits_array()
            elapsed = (time.perf_counter() - start) / num_iterations * 1000
            
            results['simd_time'].append(elapsed)
            print(f"  SIMD:      {elapsed:.3f} ms")
            
            # Calculate speedup
            if BASIC_AVAILABLE:
                speedup = results['basic_cpp_time'][-1] / elapsed
                results['speedup'].append(speedup)
                print(f"  Speedup:   {speedup:.2f}x")
            else:
                results['speedup'].append(float('nan'))
        else:
            results['simd_time'].append(float('nan'))
            results['speedup'].append(float('nan'))
            print(f"  SIMD:      Not available")
        
        print()
    
    return results


def benchmark_batch_indexing(lattice_size, batch_sizes, num_iterations=1000):
    """Benchmark batch index calculation"""
    
    print("\n" + "=" * 70)
    print("BENCHMARK: Batch Indexing")
    print("=" * 70)
    print(f"Lattice size: {lattice_size[0]}×{lattice_size[1]}")
    print(f"Iterations per batch: {num_iterations}")
    print()
    
    results = {
        'batch_sizes': [],
        'basic_loop_time': [],
        'simd_batch_time': [],
        'speedup': []
    }
    
    if not SIMD_AVAILABLE:
        print("❌ SIMD not available - cannot run batch indexing benchmark")
        return results
    
    basic_lattice = fast_lattice.FastLattice(list(lattice_size), 2)
    simd_lattice = FastLatticeSIMD(list(lattice_size), 2)
    
    for batch_size in batch_sizes:
        print(f"Batch size: {batch_size:,}")
        results['batch_sizes'].append(batch_size)
        
        # Generate random coordinates
        x_coords = np.random.randint(0, lattice_size[0], batch_size, dtype=np.int32)
        y_coords = np.random.randint(0, lattice_size[1], batch_size, dtype=np.int32)
        b_coords = np.random.randint(0, 2, batch_size, dtype=np.int32)
        
        # Basic C++ (loop over each coordinate)
        start = time.perf_counter()
        for _ in range(num_iterations):
            indices = np.zeros(batch_size, dtype=np.uint64)
            for i in range(batch_size):
                indices[i] = basic_lattice.get_flat_index_2d(
                    int(x_coords[i]), int(y_coords[i]), int(b_coords[i])
                )
        elapsed = (time.perf_counter() - start) / num_iterations * 1000
        results['basic_loop_time'].append(elapsed)
        print(f"  Basic (loop):  {elapsed:.3f} ms")
        
        # SIMD batch processing
        start = time.perf_counter()
        for _ in range(num_iterations):
            indices = simd_lattice.get_flat_indices_batch_2d(
                x_coords, y_coords, b_coords
            )
        elapsed = (time.perf_counter() - start) / num_iterations * 1000
        results['simd_batch_time'].append(elapsed)
        print(f"  SIMD (batch):  {elapsed:.3f} ms")
        
        # Speedup
        speedup = results['basic_loop_time'][-1] / elapsed
        results['speedup'].append(speedup)
        print(f"  Speedup:       {speedup:.2f}x")
        print()
    
    return results


def benchmark_neighbor_lookup(lattice_size, num_lookups=10000, radius=1):
    """Benchmark neighbor lookup operations"""
    
    print("\n" + "=" * 70)
    print("BENCHMARK: Neighbor Lookup")
    print("=" * 70)
    print(f"Lattice size: {lattice_size[0]}×{lattice_size[1]}")
    print(f"Number of lookups: {num_lookups:,}")
    print(f"Neighbor radius: {radius}")
    print()
    
    if not SIMD_AVAILABLE:
        print("❌ SIMD not available - skipping neighbor lookup benchmark")
        return {}
    
    simd_lattice = FastLatticeSIMD(list(lattice_size), 2)
    
    # Generate random positions
    x_coords = np.random.randint(radius, lattice_size[0] - radius, num_lookups)
    y_coords = np.random.randint(radius, lattice_size[1] - radius, num_lookups)
    
    # Benchmark
    start = time.perf_counter()
    total_neighbors = 0
    for i in range(num_lookups):
        neighbors = simd_lattice.get_neighbors_2d(
            int(x_coords[i]), int(y_coords[i]), 0, radius
        )
        total_neighbors += len(neighbors)
    elapsed = time.perf_counter() - start
    
    avg_time_us = (elapsed / num_lookups) * 1e6
    print(f"  Total time:       {elapsed:.3f} s")
    print(f"  Avg per lookup:   {avg_time_us:.2f} μs")
    print(f"  Avg neighbors:    {total_neighbors / num_lookups:.1f}")
    print(f"  Throughput:       {num_lookups / elapsed:.0f} lookups/sec")
    
    return {
        'total_time': elapsed,
        'avg_time_us': avg_time_us,
        'throughput': num_lookups / elapsed
    }


def plot_results(generation_results, batch_results):
    """Create visualization of benchmark results"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Lattice Generation
    ax1 = axes[0]
    x = np.arange(len(generation_results['sizes']))
    width = 0.35
    
    if any(not np.isnan(t) for t in generation_results['basic_cpp_time']):
        ax1.bar(x - width/2, generation_results['basic_cpp_time'], 
                width, label='Basic C++', color='blue', alpha=0.7)
    
    if any(not np.isnan(t) for t in generation_results['simd_time']):
        ax1.bar(x + width/2, generation_results['simd_time'], 
                width, label='SIMD', color='green', alpha=0.7)
    
    ax1.set_xlabel('Lattice Size')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Lattice Generation Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(generation_results['sizes'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Batch Indexing
    if batch_results and len(batch_results['batch_sizes']) > 0:
        ax2 = axes[1]
        
        ax2.plot(batch_results['batch_sizes'], batch_results['basic_loop_time'], 
                'o-', label='Basic (loop)', linewidth=2, markersize=8, color='blue')
        ax2.plot(batch_results['batch_sizes'], batch_results['simd_batch_time'], 
                's-', label='SIMD (batch)', linewidth=2, markersize=8, color='green')
        
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Time (ms)')
        ax2.set_title('Batch Indexing Performance')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(__file__).parent.parent.parent.parent / 'benchmarks' / 'results' / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'simd_benchmark.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_path}")
    
    return fig


def main():
    """Run all benchmarks"""
    
    print("=" * 70)
    print("SIMD vs Basic C++ Performance Comparison")
    print("=" * 70)
    print()
    print(f"Basic C++ available: {BASIC_AVAILABLE}")
    print(f"SIMD available:      {SIMD_AVAILABLE}")
    print()
    
    if not BASIC_AVAILABLE and not SIMD_AVAILABLE:
        print("❌ No modules available to benchmark!")
        print("\nBuild instructions:")
        print("  Basic C++: cd ../fast_lattice_ext && python setup.py build_ext --inplace")
        print("  SIMD:      cd . && python setup_simd.py build_ext --inplace")
        return
    
    # Test configurations
    lattice_sizes = [
        [10, 10],
        [50, 50],
        [100, 100],
        [200, 200],
        [500, 500]
    ]
    
    batch_sizes = [10, 100, 1000, 10000, 100000]
    
    # Run benchmarks
    generation_results = benchmark_lattice_generation(lattice_sizes, num_iterations=100)
    
    batch_results = {}
    if SIMD_AVAILABLE:
        batch_results = benchmark_batch_indexing([100, 100], batch_sizes, num_iterations=1000)
        neighbor_results = benchmark_neighbor_lookup([100, 100], num_lookups=10000, radius=2)
    
    # Create visualizations
    if generation_results or batch_results:
        plot_results(generation_results, batch_results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if generation_results and not all(np.isnan(generation_results['speedup'])):
        avg_speedup = np.nanmean(generation_results['speedup'])
        max_speedup = np.nanmax(generation_results['speedup'])
        print(f"Lattice Generation:")
        print(f"  Average SIMD speedup: {avg_speedup:.2f}x")
        print(f"  Maximum SIMD speedup: {max_speedup:.2f}x")
    
    if batch_results and batch_results.get('speedup'):
        avg_batch_speedup = np.mean(batch_results['speedup'])
        max_batch_speedup = np.max(batch_results['speedup'])
        print(f"\nBatch Indexing:")
        print(f"  Average SIMD speedup: {avg_batch_speedup:.2f}x")
        print(f"  Maximum SIMD speedup: {max_batch_speedup:.2f}x")
    
    print("\n✅ Benchmark complete!")


if __name__ == '__main__':
    main()