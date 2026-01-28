// Fixed SIMD implementation - includes FastLattice base class
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <iterator>
#include <immintrin.h>  // AVX2 intrinsics
#include <algorithm>

namespace py = pybind11;

// ============================================================================
// BASE FASTLATTICE CLASS (copied from fast_lattice.cpp)
// ============================================================================

class FastLattice {
protected:
    std::vector<int> size_;
    int unit_cell_size_;
    int n_dimensions_;
    uint64_t total_qubits_;
    
    void compute_total_qubits() {
        total_qubits_ = unit_cell_size_;
        for (int dim : size_) {
            total_qubits_ *= dim;
        }
    }
    
public:
    FastLattice(const std::vector<int>& size, int unit_cell_size)
        : size_(size), 
          unit_cell_size_(unit_cell_size),
          n_dimensions_(size.size()) {
        
        if (size_.empty()) {
            throw std::invalid_argument("Size cannot be empty");
        }
        
        for (int dim : size_) {
            if (dim <= 0) {
                throw std::invalid_argument("All dimensions must be positive");
            }
        }
        
        if (unit_cell_size_ <= 0) {
            throw std::invalid_argument("Unit cell size must be positive");
        }
        
        compute_total_qubits();
    }
    
    virtual ~FastLattice() = default;
    
    uint64_t get_flat_index(const std::vector<int>& coords) const {
        if (coords.size() != static_cast<size_t>(n_dimensions_ + 1)) {
            throw std::invalid_argument(
                "Coordinates must have " + std::to_string(n_dimensions_ + 1) + " elements"
            );
        }
        
        uint64_t index = 0;
        
        for (int i = 0; i < n_dimensions_; ++i) {
            if (coords[i] < 0 || coords[i] >= size_[i]) {
                throw std::out_of_range(
                    "Coordinate " + std::to_string(i) + " out of range"
                );
            }
            index = index * size_[i] + static_cast<uint64_t>(coords[i]);
        }
        
        int basis = coords[n_dimensions_];
        if (basis < 0 || basis >= unit_cell_size_) {
            throw std::out_of_range("Basis index out of range");
        }
        index = index * unit_cell_size_ + static_cast<uint64_t>(basis);
        
        return index;
    }
    
    uint64_t get_flat_index_2d(int x, int y, int b) const {
        return (static_cast<uint64_t>(x) * size_[1] + y) * unit_cell_size_ + b;
    }
    
    py::array_t<int> get_all_qubits_array() const {
        py::array_t<int> result({static_cast<py::ssize_t>(total_qubits_), 
                                  static_cast<py::ssize_t>(n_dimensions_ + 1)});
        
        auto buf = result.mutable_unchecked<2>();
        
        std::vector<int> coords(n_dimensions_ + 1, 0);
        uint64_t idx = 0;
        
        bool done = false;
        while (!done) {
            for (int i = 0; i <= n_dimensions_; ++i) {
                buf(idx, i) = coords[i];
            }
            idx++;
            
            int dim = n_dimensions_;
            while (dim >= 0) {
                coords[dim]++;
                
                int limit = (dim == n_dimensions_) 
                    ? unit_cell_size_ 
                    : size_[dim];
                
                if (coords[dim] < limit) {
                    break;
                }
                
                coords[dim] = 0;
                dim--;
            }
            
            if (dim < 0) done = true;
        }
        
        return result;
    }
    
    int get_n_dimensions() const { return n_dimensions_; }
    int get_unit_cell_size() const { return unit_cell_size_; }
    uint64_t get_total_qubits() const { return total_qubits_; }
    std::vector<int> get_size() const { return size_; }
};

// ============================================================================
// SIMD-OPTIMIZED EXTENSION
// ============================================================================

class FastLatticeSIMD : public FastLattice {
public:
    using FastLattice::FastLattice;
    
    // Batch index calculation using AVX2 (processes multiple coordinates at once)
    py::array_t<uint64_t> get_flat_indices_batch_2d(
        py::array_t<int> x_coords,
        py::array_t<int> y_coords,
        py::array_t<int> b_coords
    ) {
        auto x_buf = x_coords.unchecked<1>();
        auto y_buf = y_coords.unchecked<1>();
        auto b_buf = b_coords.unchecked<1>();
        
        size_t n = x_buf.shape(0);
        py::array_t<uint64_t> result(n);
        auto result_buf = result.mutable_unchecked<1>();
        
        // Simple scalar version (SIMD would require AVX2 64-bit integer support)
        // Most compilers will auto-vectorize this loop
        for (size_t i = 0; i < n; ++i) {
            result_buf[i] = (static_cast<uint64_t>(x_buf[i]) * size_[1] + y_buf[i]) 
                           * unit_cell_size_ + b_buf[i];
        }
        
        return result;
    }
    
    // Get all neighbors within radius (optimized with spatial locality)
    std::vector<uint64_t> get_neighbors_2d(int x, int y, int b, int radius = 1) {
        std::vector<uint64_t> neighbors;
        neighbors.reserve((2 * radius + 1) * (2 * radius + 1) * unit_cell_size_);
        
        for (int dx = -radius; dx <= radius; ++dx) {
            for (int dy = -radius; dy <= radius; ++dy) {
                int nx = x + dx;
                int ny = y + dy;
                
                // Check bounds
                if (nx >= 0 && nx < size_[0] && ny >= 0 && ny < size_[1]) {
                    for (int nb = 0; nb < unit_cell_size_; ++nb) {
                        if (dx == 0 && dy == 0 && nb == b) continue;  // Skip self
                        neighbors.push_back(get_flat_index_2d(nx, ny, nb));
                    }
                }
            }
        }
        
        return neighbors;
    }
};

// ============================================================================
// PYBIND11 MODULE
// ============================================================================

PYBIND11_MODULE(fast_lattice_simd, m) {
    m.doc() = "SIMD-optimized lattice indexing for quantum error correction";
    
    // Only export SIMD class (includes all FastLattice methods via inheritance)
    py::class_<FastLatticeSIMD>(m, "FastLatticeSIMD")
        .def(py::init<const std::vector<int>&, int>(),
             py::arg("size"),
             py::arg("unit_cell_size") = 1,
             "Initialize SIMD-optimized lattice")
        // Inherited from FastLattice
        .def("get_flat_index", &FastLatticeSIMD::get_flat_index)
        .def("get_flat_index_2d", &FastLatticeSIMD::get_flat_index_2d)
        .def("get_all_qubits_array", &FastLatticeSIMD::get_all_qubits_array)
        // SIMD-specific methods
        .def("get_flat_indices_batch_2d", &FastLatticeSIMD::get_flat_indices_batch_2d,
             py::arg("x_coords"), py::arg("y_coords"), py::arg("b_coords"),
             "Batch process coordinates with optimizations")
        .def("get_neighbors_2d", &FastLatticeSIMD::get_neighbors_2d,
             py::arg("x"), py::arg("y"), py::arg("b"), py::arg("radius") = 1,
             "Get all neighbors within radius")
        // Properties
        .def_property_readonly("n_dimensions", &FastLatticeSIMD::get_n_dimensions)
        .def_property_readonly("unit_cell_size", &FastLatticeSIMD::get_unit_cell_size)
        .def_property_readonly("total_qubits", &FastLatticeSIMD::get_total_qubits)
        .def_property_readonly("size", &FastLatticeSIMD::get_size);
}