#include <pybind11/pybind11.h>
#include "entropy_source.hpp"
#include <chrono>
#include <vector>
#include <iostream>

namespace py = pybind11;
using namespace entropy;

void profile_rng() {
    using namespace std::chrono;
    
    ThermalNoise thermal(310.0);
    int n = 500000;
    
    // Test 1: Pure RNG calls
    auto start = high_resolution_clock::now();
    Xoshiro256StarStar rng(12345);
    std::vector<uint64_t> rng_out;
    rng_out.reserve(n * 2);
    for (int i = 0; i < n * 2; ++i) {
        rng_out.push_back(rng());
    }
    auto end = high_resolution_clock::now();
    double rng_time = duration_cast<microseconds>(end - start).count() / 1000.0;
    py::print("1. Pure RNG (1M calls):", rng_time, "ms");
    
    // Test 2: RNG + conversion
    start = high_resolution_clock::now();
    const double scale = 1.0 / static_cast<double>(UINT64_MAX);
    std::vector<double> uniforms;
    uniforms.reserve(n * 2);
    for (int i = 0; i < n * 2; ++i) {
        uniforms.push_back(static_cast<double>(rng()) * scale);
    }
    end = high_resolution_clock::now();
    double convert_time = duration_cast<microseconds>(end - start).count() / 1000.0;
    py::print("2. RNG + conversion (1M):", convert_time, "ms");
    
    // Test 3: Just the do-while loop (no vector)
    start = high_resolution_clock::now();
    int count = 0;
    const double scale2 = 1.0 / static_cast<double>(UINT64_MAX);
    for (int i = 0; i < n; ++i) {
        double u, v, x, y, q;
        do {
            u = static_cast<double>(rng()) * scale2;
            v = 1.7156 * (static_cast<double>(rng()) * scale2 - 0.5);
            x = u - 0.449871;
            y = std::abs(v) + 0.386595;
            q = x*x + y*(0.19600*y - 0.25472*x);
            if (q < 0.27597) break;
            count++;
        } while (q > 0.27846 || v*v > -4.0 * u * u * std::log(u));
    }
    end = high_resolution_clock::now();
    double loop_time = duration_cast<microseconds>(end - start).count() / 1000.0;
    py::print("3. Gaussian loop only (500K):", loop_time, "ms");
    py::print("   Rejections:", count);
    
    // Test 4: Full function
    start = high_resolution_clock::now();
    auto noise = thermal.generate_thermal_noise(n);
    end = high_resolution_clock::now();
    double full_time = duration_cast<microseconds>(end - start).count() / 1000.0;
    py::print("4. Full generate_thermal_noise (500K):", full_time, "ms");
    
    py::print("\n=== BREAKDOWN ===");
    py::print("Base RNG cost:", rng_time / 2, "ms (for 500K)");
    py::print("Conversion:", (convert_time - rng_time) / 2, "ms");
    py::print("Gaussian math:", loop_time - convert_time / 2, "ms");
    py::print("Vector overhead:", full_time - loop_time, "ms");
    py::print("Total:", full_time, "ms");
}

PYBIND11_MODULE(profile_rng, m) {
    m.def("profile", &profile_rng);
}
