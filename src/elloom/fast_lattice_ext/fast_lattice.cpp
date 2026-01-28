#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <iterator>

namespace py = pybind11;

class FastLattice {
private:
    std::vector<int> size_;           // Lattice dimensions
    int unit_cell_size_;              // Number of basis vectors
    int n_dimensions_;                // Number of lattice vectors
    std::vector<uint64_t> strides_;   // Precomputed strides for indexing
    uint64_t total_qubits_;           // Total number of qubits
    
    void compute_strides() {
        strides_.resize(n_dimensions_ + 1);
        strides_[n_dimensions_] = 1;  // Basis stride
        
        // Compute strides from right to left
        for (int i = n_dimensions_ - 1; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * size_[i];
        }
        
        // Total qubits = product of all dimensions * unit_cell_size
        total_qubits_ = strides_[0] * unit_cell_size_;
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
        
        compute_strides();
    }
    
    // Fast flat index calculation using bit operations where possible
    uint64_t get_flat_index(const std::vector<int>& coords) const {
        if (coords.size() != static_cast<size_t>(n_dimensions_ + 1)) {
            throw std::invalid_argument(
                "Coordinates must have " + std::to_string(n_dimensions_ + 1) + " elements"
            );
        }
        
        uint64_t index = 0;
        
        // Process lattice coordinates from left to right
        for (int i = 0; i < n_dimensions_; ++i) {
            if (coords[i] < 0 || coords[i] >= size_[i]) {
                throw std::out_of_range(
                    "Coordinate " + std::to_string(i) + " out of range"
                );
            }
            index = index * size_[i] + static_cast<uint64_t>(coords[i]);
        }
        
        // Multiply by unit cell size and add basis index
        int basis = coords[n_dimensions_];
        if (basis < 0 || basis >= unit_cell_size_) {
            throw std::out_of_range("Basis index out of range");
        }
        index = index * unit_cell_size_ + static_cast<uint64_t>(basis);
        
        return index;
    }
    
    // Overloaded versions for common dimensions
    uint64_t get_flat_index_1d(int x, int b) const {
        return static_cast<uint64_t>(x) * unit_cell_size_ + b;
    }
    
    uint64_t get_flat_index_2d(int x, int y, int b) const {
        return (static_cast<uint64_t>(x) * size_[1] + y) * unit_cell_size_ + b;
    }
    
    uint64_t get_flat_index_3d(int x, int y, int z, int b) const {
        return ((static_cast<uint64_t>(x) * size_[1] + y) * size_[2] + z) * unit_cell_size_ + b;
    }
    
    // Iterator class for efficient traversal
    class QubitIterator {
    private:
        const FastLattice* lattice_;
        std::vector<int> current_coords_;
        bool at_end_;
        
        void increment() {
            // Increment from rightmost dimension
            int dim = lattice_->n_dimensions_;
            
            while (dim >= 0) {
                current_coords_[dim]++;
                
                int limit = (dim == lattice_->n_dimensions_) 
                    ? lattice_->unit_cell_size_ 
                    : lattice_->size_[dim];
                
                if (current_coords_[dim] < limit) {
                    return;  // No carry needed
                }
                
                current_coords_[dim] = 0;
                dim--;
            }
            
            // If we've carried past all dimensions, we're done
            at_end_ = true;
        }
        
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::vector<int>;
        using difference_type = std::ptrdiff_t;
        using pointer = const std::vector<int>*;
        using reference = const std::vector<int>&;
        
        QubitIterator(const FastLattice* lattice, bool at_end = false)
            : lattice_(lattice), 
              current_coords_(lattice->n_dimensions_ + 1, 0),
              at_end_(at_end) {}
        
        reference operator*() const { return current_coords_; }
        pointer operator->() const { return &current_coords_; }
        
        QubitIterator& operator++() {
            increment();
            return *this;
        }
        
        QubitIterator operator++(int) {
            QubitIterator tmp = *this;
            increment();
            return tmp;
        }
        
        bool operator==(const QubitIterator& other) const {
            return at_end_ == other.at_end_ && 
                   (at_end_ || current_coords_ == other.current_coords_);
        }
        
        bool operator!=(const QubitIterator& other) const {
            return !(*this == other);
        }
    };
    
    QubitIterator begin() const { return QubitIterator(this, false); }
    QubitIterator end() const { return QubitIterator(this, true); }
    
    // Generate all qubits as numpy array (still faster than Python)
    py::array_t<int> get_all_qubits_array() const {
        py::array_t<int> result({static_cast<py::ssize_t>(total_qubits_), 
                                  static_cast<py::ssize_t>(n_dimensions_ + 1)});
        
        auto buf = result.mutable_unchecked<2>();
        
        std::vector<int> coords(n_dimensions_ + 1, 0);
        uint64_t idx = 0;
        
        // Manual iteration without creating iterator objects
        bool done = false;
        while (!done) {
            // Copy current coordinates
            for (int i = 0; i <= n_dimensions_; ++i) {
                buf(idx, i) = coords[i];
            }
            idx++;
            
            // Increment coordinates
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
    
    // Generate flat indices for all qubits
    py::array_t<uint64_t> get_all_flat_indices() const {
        py::array_t<uint64_t> result(total_qubits_);
        auto buf = result.mutable_unchecked<1>();
        
        for (uint64_t i = 0; i < total_qubits_; ++i) {
            buf(i) = i;
        }
        
        return result;
    }
    
    // Getters
    int get_n_dimensions() const { return n_dimensions_; }
    int get_unit_cell_size() const { return unit_cell_size_; }
    uint64_t get_total_qubits() const { return total_qubits_; }
    std::vector<int> get_size() const { return size_; }
};

PYBIND11_MODULE(fast_lattice, m) {
    m.doc() = "High-performance lattice indexing for quantum error correction";
    
    py::class_<FastLattice>(m, "FastLattice")
        .def(py::init<const std::vector<int>&, int>(),
             py::arg("size"),
             py::arg("unit_cell_size") = 1,
             "Initialize lattice with size and unit cell size")
        
        .def("get_flat_index", &FastLattice::get_flat_index,
             py::arg("coords"),
             "Get flat memory index from lattice coordinates")
        
        .def("get_flat_index_1d", &FastLattice::get_flat_index_1d,
             py::arg("x"), py::arg("b") = 0,
             "Fast 1D indexing")
        
        .def("get_flat_index_2d", &FastLattice::get_flat_index_2d,
             py::arg("x"), py::arg("y"), py::arg("b") = 0,
             "Fast 2D indexing")
        
        .def("get_flat_index_3d", &FastLattice::get_flat_index_3d,
             py::arg("x"), py::arg("y"), py::arg("z"), py::arg("b") = 0,
             "Fast 3D indexing")
        
        .def("get_all_qubits_array", &FastLattice::get_all_qubits_array,
             "Get all qubit coordinates as numpy array")
        
        .def("get_all_flat_indices", &FastLattice::get_all_flat_indices,
             "Get all flat indices as numpy array")
        
        .def("__iter__", [](const FastLattice& lattice) {
            return py::make_iterator(lattice.begin(), lattice.end());
        }, py::keep_alive<0, 1>())
        
        .def("__len__", &FastLattice::get_total_qubits)
        
        .def_property_readonly("n_dimensions", &FastLattice::get_n_dimensions)
        .def_property_readonly("unit_cell_size", &FastLattice::get_unit_cell_size)
        .def_property_readonly("total_qubits", &FastLattice::get_total_qubits)
        .def_property_readonly("size", &FastLattice::get_size);
}