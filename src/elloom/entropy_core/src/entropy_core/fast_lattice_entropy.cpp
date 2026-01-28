#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include "entropy_source.hpp"

namespace py = pybind11;
using namespace entropy;

class FastLatticeEntropy {
private:
    std::vector<int> size_;
    int unit_cell_size_;
    uint64_t total_sites_;
    uint64_t total_qubits_;
    int n_dimensions_;
    std::unique_ptr<ThermalNoise> thermal_noise_;
    double current_temperature_;

public:
    FastLatticeEntropy(std::vector<int> size, int unit_cell_size)
        : size_(size), unit_cell_size_(unit_cell_size) {

        n_dimensions_ = static_cast<int>(size_.size());

        // Calculate sites first
        total_sites_ = 1;
        for (int s : size_) {
            total_sites_ *= s;
        }
        
        // Total qubits = sites × basis qubits per site
        total_qubits_ = total_sites_ * unit_cell_size_;

        current_temperature_ = EntropySource::get_cpu_temperature();
        thermal_noise_ = std::make_unique<ThermalNoise>(current_temperature_);
    }

    py::tuple get_all_qubits_with_noise(double noise_strength = 1.0) {
        // Generate integer lattice
        py::array_t<int> lattice({static_cast<py::ssize_t>(total_qubits_),
                                   static_cast<py::ssize_t>(n_dimensions_ + 1)});
        auto lat_buf = lattice.request();
        int* lat_ptr = static_cast<int*>(lat_buf.ptr);

        uint64_t idx = 0;

        if (n_dimensions_ == 2) {
            for (int x = 0; x < size_[0]; ++x) {
                for (int y = 0; y < size_[1]; ++y) {
                    for (int b = 0; b < unit_cell_size_; ++b) {
                        lat_ptr[idx * 3 + 0] = x;
                        lat_ptr[idx * 3 + 1] = y;
                        lat_ptr[idx * 3 + 2] = b;
                        idx++;
                    }
                }
            }
        } else if (n_dimensions_ == 3) {
            for (int x = 0; x < size_[0]; ++x) {
                for (int y = 0; y < size_[1]; ++y) {
                    for (int z = 0; z < size_[2]; ++z) {
                        for (int b = 0; b < unit_cell_size_; ++b) {
                            lat_ptr[idx * 4 + 0] = x;
                            lat_ptr[idx * 4 + 1] = y;
                            lat_ptr[idx * 4 + 2] = z;
                            lat_ptr[idx * 4 + 3] = b;
                            idx++;
                        }
                    }
                }
            }
        }

        // Generate noise
        py::array_t<double> noise(total_qubits_);
        auto noise_buf = noise.request();
        double* noise_ptr = static_cast<double*>(noise_buf.ptr);

        thermal_noise_->generate_thermal_noise_inplace(noise_ptr, static_cast<int>(total_qubits_));

        // Scale noise by strength
        if (noise_strength != 1.0) {
            for (uint64_t i = 0; i < total_qubits_; ++i) {
                noise_ptr[i] *= noise_strength;
            }
        }

        return py::make_tuple(lattice, noise);
    }

    py::array_t<uint64_t> get_flat_indices_with_entropy_2d(
        py::array_t<int> x_coords,
        py::array_t<int> y_coords,
        py::array_t<int> b_coords
    ) {
        auto x_buf = x_coords.request();
        auto y_buf = y_coords.request();
        auto b_buf = b_coords.request();

        if (x_buf.size != y_buf.size || x_buf.size != b_buf.size) {
            throw std::runtime_error("Coordinate arrays must have same size");
        }

        size_t n = static_cast<size_t>(x_buf.size);
        int* x_ptr = static_cast<int*>(x_buf.ptr);
        int* y_ptr = static_cast<int*>(y_buf.ptr);
        int* b_ptr = static_cast<int*>(b_buf.ptr);

        py::array_t<uint64_t> result(n);
        auto result_buf = result.request();
        uint64_t* result_ptr = static_cast<uint64_t*>(result_buf.ptr);

        int size_y = size_[1];

        for (size_t i = 0; i < n; ++i) {
            result_ptr[i] = (x_ptr[i] * size_y + y_ptr[i]) * unit_cell_size_ + b_ptr[i];
        }

        return result;
    }

    py::array_t<int> get_all_qubits_pure() {
        py::array_t<int> result({static_cast<py::ssize_t>(total_qubits_),
                                  static_cast<py::ssize_t>(n_dimensions_ + 1)});
        auto buf = result.request();
        int* ptr = static_cast<int*>(buf.ptr);

        uint64_t idx = 0;

        if (n_dimensions_ == 2) {
            for (int x = 0; x < size_[0]; ++x) {
                for (int y = 0; y < size_[1]; ++y) {
                    for (int b = 0; b < unit_cell_size_; ++b) {
                        ptr[idx * 3 + 0] = x;
                        ptr[idx * 3 + 1] = y;
                        ptr[idx * 3 + 2] = b;
                        idx++;
                    }
                }
            }
        } else if (n_dimensions_ == 3) {
            for (int x = 0; x < size_[0]; ++x) {
                for (int y = 0; y < size_[1]; ++y) {
                    for (int z = 0; z < size_[2]; ++z) {
                        for (int b = 0; b < unit_cell_size_; ++b) {
                            ptr[idx * 4 + 0] = x;
                            ptr[idx * 4 + 1] = y;
                            ptr[idx * 4 + 2] = z;
                            ptr[idx * 4 + 3] = b;
                            idx++;
                        }
                    }
                }
            }
        }

        return result;
    }

    py::array_t<double> generate_noise_only(int n_samples) {
        py::array_t<double> noise(n_samples);
        auto buf = noise.request();
        double* ptr = static_cast<double*>(buf.ptr);

        thermal_noise_->generate_thermal_noise_inplace(ptr, n_samples);

        return noise;
    }

    double get_system_temperature() const {
        return EntropySource::get_cpu_temperature();
    }

    void update_temperature() {
        current_temperature_ = EntropySource::get_cpu_temperature();
        thermal_noise_ = std::make_unique<ThermalNoise>(current_temperature_);
    }
    
    py::dict get_lattice_info() const {
        py::dict info;
        info["dimensions"] = n_dimensions_;
        info["size"] = size_;
        info["unit_cell_size"] = unit_cell_size_;
        info["total_sites"] = total_sites_;
        info["total_qubits"] = total_qubits_;
        info["temperature_K"] = current_temperature_;
        return info;
    }
};

// Change module name to entropy_core
PYBIND11_MODULE(entropy_core, m) {
    m.doc() = "High-performance quantum lattice with integrated entropy";

    py::class_<FastLatticeEntropy>(m, "FastLatticeEntropy")
        .def(py::init<std::vector<int>, int>())
        .def("get_all_qubits_with_noise", &FastLatticeEntropy::get_all_qubits_with_noise,
             py::arg("noise_strength") = 1.0)
        .def("get_all_qubits_pure", &FastLatticeEntropy::get_all_qubits_pure)
        .def("generate_noise_only", &FastLatticeEntropy::generate_noise_only)
        .def("get_flat_indices_with_entropy_2d",
             &FastLatticeEntropy::get_flat_indices_with_entropy_2d)
        .def("get_system_temperature", &FastLatticeEntropy::get_system_temperature)
        .def("update_temperature", &FastLatticeEntropy::update_temperature)
        .def("get_lattice_info", &FastLatticeEntropy::get_lattice_info);
}
