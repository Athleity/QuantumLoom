#include "entropy_source.hpp"
#include <chrono>
#include <iostream>
#include <vector>

int main() {
    using namespace std::chrono;
    using namespace entropy;
    
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
    std::cout << "1. Pure RNG (1M calls): " << rng_time << " ms\n";
    
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
    std::cout << "2. RNG + conversion (1M): " << convert_time << " ms\n";
    
    // Test 3: Full Gaussian
    start = high_resolution_clock::now();
    auto noise = thermal.generate_thermal_noise(n);
    end = high_resolution_clock::now();
    double gaussian_time = duration_cast<microseconds>(end - start).count() / 1000.0;
    std::cout << "3. Full Gaussian (500K): " << gaussian_time << " ms\n";
    
    std::cout << "\nBREAKDOWN:\n";
    std::cout << "  Base RNG cost: " << rng_time << " ms\n";
    std::cout << "  Conversion overhead: " << (convert_time - rng_time) << " ms\n";
    std::cout << "  Gaussian transform: " << (gaussian_time - convert_time/2) << " ms\n";
    std::cout << "  Total measured: " << gaussian_time << " ms\n";
    
    return 0;
}
