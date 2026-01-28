#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "entropy_source.hpp"

namespace py = pybind11;
using namespace entropy;

class EntropyAwareLattice {
private:
    std::vector<int> size_;
    int unit_cell_size_;
    uint64_t total_qubits_;
    std::unique_ptr<ThermalNoise> thermal_noise_;
    double current_temperature_;
    
public:
    EntropyAwareLattice(std::vector<int> size, int unit_cell_size)
        : size_(size), unit_cell_size_(unit_cell_size) {
        
        total_qubits_ = unit_cell_size_;
        for (int s : size_) {
            total_qubits_ *= s;
        }
        
        // Initialize with current CPU temperature
        current_temperature_ = EntropySource::get_cpu_temperature();
        thermal_noise_ = std::make_unique<ThermalNoise>(current_temperature_);
    }
    
    // Get current system temperature
    double get_system_temperature() const {
        return EntropySource::get_cpu_temperature();
    }
    
    // Sample system entropy
    py::array_t<double> sample_system_entropy(int n_samples) {
        auto entropy_data = EntropySource::get_mixed_entropy(n_samples);
        
        py::array_t<double> result(entropy_data.size());
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);
        
        std::copy(entropy_data.begin(), entropy_data.end(), ptr);
        return result;
    }
    
    // Generate thermal noise
    py::array_t<double> generate_thermal_noise(int n_samples) {
        auto noise = thermal_noise_->generate_thermal_noise(n_samples);
        
        py::array_t<double> result(noise.size());
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);
        
        std::copy(noise.begin(), noise.end(), ptr);
        return result;
    }
    
    // Update temperature (adaptive mode)
    void update_temperature() {
        current_temperature_ = EntropySource::get_cpu_temperature();
        thermal_noise_ = std::make_unique<ThermalNoise>(current_temperature_);
    }
    
    // Get entropy metrics
    py::dict get_entropy_metrics() {
        py::dict metrics;
        
        metrics["current_temp_K"] = current_temperature_;
        metrics["current_temp_C"] = current_temperature_ - 273.15;
        metrics["thermal_variance"] = thermal_noise_->get_thermal_variance();
        
        // Sample timing jitter statistics
        auto jitter = EntropySource::sample_timing_jitter(1000);
        double avg_jitter = 0.0;
        for (auto j : jitter) {
            avg_jitter += j;
        }
        avg_jitter /= jitter.size();
        metrics["avg_timing_jitter_ns"] = avg_jitter;
        
        return metrics;
    }
    
    std::string get_system_info() const {
        return EntropySource::get_system_info();
    }
};

// Python bindings
PYBIND11_MODULE(entropy_lattice, m) {
    m.doc() = "Entropy-aware lattice for quantum error correction";
    
    py::class_<EntropyAwareLattice>(m, "EntropyAwareLattice")
        .def(py::init<std::vector<int>, int>(),
             py::arg("size"),
             py::arg("unit_cell_size"))
        .def("get_system_temperature", &EntropyAwareLattice::get_system_temperature)
        .def("sample_system_entropy", &EntropyAwareLattice::sample_system_entropy)
        .def("generate_thermal_noise", &EntropyAwareLattice::generate_thermal_noise)
        .def("update_temperature", &EntropyAwareLattice::update_temperature)
        .def("get_entropy_metrics", &EntropyAwareLattice::get_entropy_metrics)
        .def("get_system_info", &EntropyAwareLattice::get_system_info);
}
