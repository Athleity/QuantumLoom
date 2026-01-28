"""
Complete Benchmark Suite - Compares ALL implementations
Location: benchmarks/benchmark_all.py

Benchmarks:
1. Original Python (from el-loom)
2. Basic C++ FastLattice
3. SIMD-optimized
4. GPU/CUDA (if available)
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src' / 'elloom' / 'fast_lattice_ext'))
sys.path.insert(0, str(project_root / 'src' / 'elloom' / 'advanced_lattice'))
sys.path.insert(0, str(project_root / 'src' / 'elloom'))

# Check what's available
print("Checking available modules...")
print("-" * 70)

PYTHON_AVAILABLE = False
try:
    from elloom.lattice import Lattice
    PYTHON_AVAILABLE = True
    print("✓ Original Python implementation")
except ImportError as e:
    print(f"✗ Original Python: {e}")

BASIC_CPP_AVAILABLE = False
try:
    import fast_lattice
    BASIC_CPP_AVAILABLE = True
    print("✓ Basic C++ FastLattice")
except ImportError as e:
    print(f"✗ Basic C++: {e}")

SIMD_AVAILABLE = False
try:
    from fast_lattice_simd import FastLatticeSIMD
    SIMD_AVAILABLE = True
    print("✓ SIMD-optimized")
except ImportError as e:
    print(f"✗ SIMD: {e}")

CUDA_AVAILABLE = False
try:
    from fast_lattice_cuda import CUDALattice
    CUDA_AVAILABLE = True
    print("✓ GPU/CUDA")
except ImportError as e:
    print(f"✗ CUDA: {e}")

print("-" * 70)
print()


def benchmark_python_original(size, unit_cell_size, num_iterations):
    """Benchmark original Python implementation"""
    if not PYTHON_AVAILABLE:
        return float('nan')
    
    try:
        # Create lattice
        lattice = Lattice.square_2d(lattice_size=tuple(size))
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            qubits = lattice.all_qubits()
        elapsed = (time.perf_counter() - start) / num_iterations * 1000
        
        return elapsed
    except Exception as e:
        print(f"  Error in Python benchmark: {e}")
        return float('nan')


def benchmark_basic_cpp(size, unit_cell_size, num_iterations):
    """Benchmark basic C++ implementation"""
    if not BASIC_CPP_AVAILABLE:
        return float('nan')
    
    lattice = fast_lattice.FastLattice(size, unit_cell_size)
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        qubits = lattice.get_all_qubits_array()
    elapsed = (time.perf_counter() - start) / num_iterations * 1000
    
    return elapsed


def benchmark_simd(size, unit_cell_size, num_iterations):
    """Benchmark SIMD implementation"""
    if not SIMD_AVAILABLE:
        return float('nan')
    
    lattice = FastLatticeSIMD(size, unit_cell_size)
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        qubits = lattice.get_all_qubits_array()
    elapsed = (time.perf_counter() - start) / num_iterations * 1000
    
    return elapsed


def benchmark_cuda(size, unit_cell_size, num_iterations):
    """Benchmark CUDA implementation"""
    if not CUDA_AVAILABLE:
        return float('nan')
    
    lattice = CUDALattice(size, unit_cell_size)
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        qubits = lattice.generate_all_qubits_gpu()
    elapsed = (time.perf_counter() - start) / num_iterations * 1000
    
    return elapsed


def run_comprehensive_benchmark():
    """Run comprehensive benchmark across all implementations"""
    
    print("=" * 70)
    print("COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("=" * 70)
    print()
    
    # Test configurations
    configs = [
        {'size': [10, 10], 'unit_cell': 2, 'iters': 1000, 'name': '10×10'},
        {'size': [50, 50], 'unit_cell': 2, 'iters': 100, 'name': '50×50'},
        {'size': [100, 100], 'unit_cell': 2, 'iters': 100, 'name': '100×100'},
        {'size': [200, 200], 'unit_cell': 2, 'iters': 50, 'name': '200×200'},
        {'size': [500, 500], 'unit_cell': 2, 'iters': 10, 'name': '500×500'},
    ]
    
    results = []
    
    for config in configs:
        size = config['size']
        unit_cell = config['unit_cell']
        iters = config['iters']
        name = config['name']
        total_qubits = size[0] * size[1] * unit_cell
        
        print(f"\nTesting {name} ({total_qubits:,} qubits, {iters} iterations)")
        print("-" * 70)
        
        result = {
            'size': name,
            'total_qubits': total_qubits,
        }
        
        # Python
        if PYTHON_AVAILABLE:
            time_py = benchmark_python_original(size, unit_cell, iters)
            result['python'] = time_py
            print(f"  Python:    {time_py:8.3f} ms")
        else:
            result['python'] = float('nan')
            print(f"  Python:    Not available")
        
        # Basic C++
        if BASIC_CPP_AVAILABLE:
            time_cpp = benchmark_basic_cpp(size, unit_cell, iters)
            result['cpp'] = time_cpp
            print(f"  C++:       {time_cpp:8.3f} ms", end='')
            if PYTHON_AVAILABLE and not np.isnan(time_py):
                speedup = time_py / time_cpp
                result['cpp_speedup'] = speedup
                print(f"  (speedup: {speedup:5.1f}x)")
            else:
                result['cpp_speedup'] = float('nan')
                print()
        else:
            result['cpp'] = float('nan')
            result['cpp_speedup'] = float('nan')
            print(f"  C++:       Not available")
        
        # SIMD
        if SIMD_AVAILABLE:
            time_simd = benchmark_simd(size, unit_cell, iters)
            result['simd'] = time_simd
            print(f"  SIMD:      {time_simd:8.3f} ms", end='')
            if PYTHON_AVAILABLE and not np.isnan(time_py):
                speedup = time_py / time_simd
                result['simd_speedup'] = speedup
                print(f"  (speedup: {speedup:5.1f}x)")
            else:
                result['simd_speedup'] = float('nan')
                print()
        else:
            result['simd'] = float('nan')
            result['simd_speedup'] = float('nan')
            print(f"  SIMD:      Not available")
        
        # CUDA
        if CUDA_AVAILABLE:
            time_cuda = benchmark_cuda(size, unit_cell, iters)
            result['cuda'] = time_cuda
            print(f"  CUDA:      {time_cuda:8.3f} ms", end='')
            if PYTHON_AVAILABLE and not np.isnan(time_py):
                speedup = time_py / time_cuda
                result['cuda_speedup'] = speedup
                print(f"  (speedup: {speedup:5.1f}x)")
            else:
                result['cuda_speedup'] = float('nan')
                print()
        else:
            result['cuda'] = float('nan')
            result['cuda_speedup'] = float('nan')
            print(f"  CUDA:      Not available")
        
        results.append(result)
    
    return pd.DataFrame(results)


def plot_benchmark_results(df):
    """Create visualization of benchmark results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Execution Time Comparison
    ax1 = axes[0, 0]
    x = np.arange(len(df))
    width = 0.2
    
    cols_to_plot = []
    labels = []
    colors = []
    
    if 'python' in df.columns and not df['python'].isna().all():
        cols_to_plot.append('python')
        labels.append('Python')
        colors.append('red')
    
    if 'cpp' in df.columns and not df['cpp'].isna().all():
        cols_to_plot.append('cpp')
        labels.append('C++')
        colors.append('blue')
    
    if 'simd' in df.columns and not df['simd'].isna().all():
        cols_to_plot.append('simd')
        labels.append('SIMD')
        colors.append('green')
    
    if 'cuda' in df.columns and not df['cuda'].isna().all():
        cols_to_plot.append('cuda')
        labels.append('CUDA')
        colors.append('purple')
    
    for i, (col, label, color) in enumerate(zip(cols_to_plot, labels, colors)):
        offset = (i - len(cols_to_plot)/2 + 0.5) * width
        ax1.bar(x + offset, df[col], width, label=label, color=color, alpha=0.7)
    
    ax1.set_xlabel('Lattice Size')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['size'])
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Speedup vs Python
    ax2 = axes[0, 1]
    speedup_cols = [c for c in df.columns if c.endswith('_speedup')]
    
    for col in speedup_cols:
        impl_name = col.replace('_speedup', '').upper()
        ax2.plot(df['size'], df[col], 'o-', label=impl_name, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Lattice Size')
    ax2.set_ylabel('Speedup vs Python')
    ax2.set_title('Performance Speedup')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Baseline (Python)')
    
    # Plot 3: Throughput (qubits/second)
    ax3 = axes[1, 0]
    
    for col in cols_to_plot:
        throughput = (df['total_qubits'] / df[col]) * 1000  # qubits per second
        label = col.upper() if col != 'python' else 'Python'
        ax3.plot(df['size'], throughput, 'o-', label=label, linewidth=2, markersize=8)
    
    ax3.set_xlabel('Lattice Size')
    ax3.set_ylabel('Throughput (qubits/second)')
    ax3.set_title('Processing Throughput')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Scaling Analysis
    ax4 = axes[1, 1]
    
    for col in cols_to_plot:
        label = col.upper() if col != 'python' else 'Python'
        ax4.loglog(df['total_qubits'], df[col], 'o-', label=label, linewidth=2, markersize=8)
    
    ax4.set_xlabel('Total Qubits')
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('Scaling Behavior (log-log)')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent / 'results' / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'comprehensive_benchmark.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_path}")
    
    return fig


def save_results(df):
    """Save results to CSV and JSON"""
    
    output_dir = Path(__file__).parent / 'results' / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV
    csv_path = output_dir / 'benchmark_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"📄 CSV saved to: {csv_path}")
    
    # JSON
    json_path = output_dir / 'benchmark_results.json'
    df.to_json(json_path, orient='records', indent=2)
    print(f"📄 JSON saved to: {json_path}")


def main():
    """Run complete benchmark suite"""
    
    if not any([PYTHON_AVAILABLE, BASIC_CPP_AVAILABLE, SIMD_AVAILABLE, CUDA_AVAILABLE]):
        print("\n❌ No implementations available to benchmark!")
        print("\nBuild instructions:")
        print("  1. Basic C++: cd src/elloom/fast_lattice_ext && python setup.py build_ext --inplace")
        print("  2. SIMD:      cd src/elloom/advanced_lattice && python setup_simd.py build_ext --inplace")
        print("  3. CUDA:      cd src/elloom/advanced_lattice && python setup_cuda.py build_ext --inplace")
        return
    
    # Run benchmarks
    df = run_comprehensive_benchmark()
    
    # Display summary
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(df.to_string(index=False))
    
    # Calculate averages
    print("\n" + "=" * 70)
    print("AVERAGE SPEEDUPS")
    print("=" * 70)
    
    for col in df.columns:
        if col.endswith('_speedup'):
            impl = col.replace('_speedup', '').upper()
            avg = df[col].mean()
            max_val = df[col].max()
            if not np.isnan(avg):
                print(f"{impl:8s}: Average = {avg:5.1f}x, Maximum = {max_val:5.1f}x")
    
    # Save results
    print()
    save_results(df)
    
    # Plot
    plot_benchmark_results(df)
    
    print("\n✅ Complete benchmark finished!")


if __name__ == '__main__':
    main()