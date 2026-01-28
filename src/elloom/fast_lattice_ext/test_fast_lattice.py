"""
Example usage and benchmarks for FastLattice C++ module.

Build instructions:
    pip install pybind11 numpy
    python setup.py build_ext --inplace

This will create fast_lattice.so (or .pyd on Windows)
"""

import numpy as np
import time
from typing import List, Tuple

# Import the C++ module
import fast_lattice


def benchmark_original_vs_fast():
    """Compare original Python implementation with C++ FastLattice"""
    
    # Test parameters
    sizes = [
        ([10], 2, "1D (10 cells, 2 basis)"),
        ([20, 20], 2, "2D (20x20, 2 basis)"),
        ([50, 50], 3, "2D (50x50, 3 basis)"),
        ([20, 20, 20], 1, "3D (20x20x20, 1 basis)"),
        ([100, 100], 2, "2D Large (100x100, 2 basis)"),
    ]
    
    print("=" * 70)
    print("BENCHMARK: Original Python vs C++ FastLattice")
    print("=" * 70)
    
    for size, unit_cell_size, description in sizes:
        print(f"\n{description}")
        print("-" * 70)
        
        # Create C++ lattice
        fast_lat = fast_lattice.FastLattice(size, unit_cell_size)
        total_qubits = fast_lat.total_qubits
        
        print(f"Total qubits: {total_qubits:,}")
        
        # Benchmark: Generate all qubits
        start = time.perf_counter()
        qubits_cpp = fast_lat.get_all_qubits_array()
        time_cpp_array = time.perf_counter() - start
        
        # Benchmark: Iterator (if reasonable size)
        if total_qubits < 100000:
            start = time.perf_counter()
            count = sum(1 for _ in fast_lat)
            time_cpp_iter = time.perf_counter() - start
            print(f"  C++ Array:    {time_cpp_array*1000:.2f} ms")
            print(f"  C++ Iterator: {time_cpp_iter*1000:.2f} ms")
        else:
            print(f"  C++ Array:    {time_cpp_array*1000:.2f} ms")
            print(f"  C++ Iterator: (skipped - too large)")
        
        # Simulate original approach (simplified)
        start = time.perf_counter()
        n_dims = len(size)
        grids = np.meshgrid(*[range(s) for s in size], indexing='ij')
        unit_cells = list(zip(*[grid.flatten() for grid in grids]))
        
        if unit_cell_size == 1:
            all_qubits = unit_cells
        else:
            bases = range(unit_cell_size)
            all_qubits = [
                tuple_ + (b,) 
                for tuple_ in unit_cells 
                for b in bases
            ]
        time_python = time.perf_counter() - start
        
        print(f"  Python orig:  {time_python*1000:.2f} ms")
        print(f"  Speedup:      {time_python/time_cpp_array:.1f}x")


def example_usage():
    """Demonstrate FastLattice features"""
    
    print("\n" + "=" * 70)
    print("EXAMPLE USAGE")
    print("=" * 70)
    
    # Create a 2D lattice (square_2d equivalent)
    print("\n1. Creating a 2D square lattice (10x10 with 2 basis)")
    lattice = fast_lattice.FastLattice([10, 10], 2)
    
    print(f"   Dimensions: {lattice.n_dimensions}")
    print(f"   Unit cell size: {lattice.unit_cell_size}")
    print(f"   Total qubits: {lattice.total_qubits}")
    print(f"   Size: {lattice.size}")
    
    # Direct indexing
    print("\n2. Fast indexing examples:")
    coords = [5, 7, 1]  # x=5, y=7, basis=1
    flat_idx = lattice.get_flat_index(coords)
    print(f"   Coords {coords} -> Flat index: {flat_idx}")
    
    # Using specialized 2D method
    flat_idx_2d = lattice.get_flat_index_2d(5, 7, 1)
    print(f"   Direct 2D indexing: {flat_idx_2d}")
    print(f"   Match: {flat_idx == flat_idx_2d}")
    
    # Iteration
    print("\n3. Iterating over first 10 qubits:")
    for i, coords in enumerate(lattice):
        if i >= 10:
            break
        flat_idx = lattice.get_flat_index(coords)
        print(f"   Qubit {i}: {coords} -> index {flat_idx}")
    
    # Get all as numpy array
    print("\n4. Getting all qubits as numpy array:")
    all_qubits = lattice.get_all_qubits_array()
    print(f"   Shape: {all_qubits.shape}")
    print(f"   First 5 qubits:\n{all_qubits[:5]}")
    
    # 3D example
    print("\n5. Creating a 3D cubic lattice (5x5x5)")
    lattice_3d = fast_lattice.FastLattice([5, 5, 5], 1)
    print(f"   Total qubits: {lattice_3d.total_qubits}")
    
    coords_3d = [2, 3, 4, 0]  # x=2, y=3, z=4, basis=0
    flat_idx_3d = lattice_3d.get_flat_index_3d(2, 3, 4, 0)
    print(f"   Coords {coords_3d} -> Flat index: {flat_idx_3d}")
    
    # Get flat indices directly
    print("\n6. Getting all flat indices:")
    flat_indices = lattice_3d.get_all_flat_indices()
    print(f"   Shape: {flat_indices.shape}")
    print(f"   First 10: {flat_indices[:10]}")


def memory_layout_demo():
    """Show how memory layout works"""
    
    print("\n" + "=" * 70)
    print("MEMORY LAYOUT DEMONSTRATION")
    print("=" * 70)
    
    # Small 2D lattice for visualization
    lattice = fast_lattice.FastLattice([3, 3], 2)
    
    print("\n2D Lattice: 3x3 cells, 2 basis vectors")
    print("Total qubits: 18 (3 * 3 * 2)")
    print("\nMemory layout (flat_index | [x, y, b]):")
    
    all_qubits = lattice.get_all_qubits_array()
    for i in range(len(lattice)):
        coords = all_qubits[i]
        flat_idx = lattice.get_flat_index_2d(coords[0], coords[1], coords[2])
        print(f"  {flat_idx:3d} | {coords}")
    
    print("\nNote: Basis index varies fastest (innermost loop)")
    print("      This ensures qubits in the same unit cell are adjacent in memory")


def integration_example():
    """Show how to integrate with existing Entropica code"""
    
    print("\n" + "=" * 70)
    print("INTEGRATION WITH ENTROPICA LATTICE")
    print("=" * 70)
    
    print("""
The FastLattice can be used as a drop-in replacement for expensive operations:

class Lattice:
    # ... existing Entropica code ...
    
    def __init__(self, basis_vectors, lattice_vectors, size=None, lattice_type=...):
        # ... existing initialization ...
        
        # Add FastLattice for performance-critical operations
        if size is not None:
            self._fast_lattice = fast_lattice.FastLattice(
                list(size), 
                len(basis_vectors)
            )
        else:
            self._fast_lattice = None
    
    def all_qubits(self, size=None, force_including_basis=False):
        # Use C++ version when possible
        if self._fast_lattice is not None and size is None:
            qubit_array = self._fast_lattice.get_all_qubits_array()
            return [tuple(q) for q in qubit_array]
        
        # Fall back to original implementation
        unit_cells = self.all_unit_cells(size)
        # ... rest of original code ...
    
    def get_qubit_index(self, coords):
        '''Get flat memory index for a qubit'''
        if self._fast_lattice is not None:
            return self._fast_lattice.get_flat_index(coords)
        # Fallback calculation
        # ...

Benefits:
  - 10-100x speedup for large lattices
  - Zero Python object allocation during iteration
  - Cache-friendly memory layout
  - Backwards compatible (only used when size is known)
    """)


if __name__ == "__main__":
    try:
        example_usage()
        memory_layout_demo()
        benchmark_original_vs_fast()
        integration_example()
        
    except ImportError as e:
        print("\n" + "!" * 70)
        print("ERROR: Could not import fast_lattice module")
        print("!" * 70)
        print("\nPlease build the module first:")
        print("  pip install pybind11 numpy")
        print("  python setup.py build_ext --inplace")
        print("\nError details:", str(e))