#pragma once
#include <vector>
#include <cstdint>
#include <chrono>
#include <string>
#include <cmath>

namespace entropy {

class Xoshiro256StarStar {
private:
    uint64_t s[4];
    
    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
    
public:
    using result_type = uint64_t;
    
    explicit Xoshiro256StarStar(uint64_t seed = 0) {
        s[0] = seed;
        s[1] = seed + 0x9E3779B97F4A7C15ULL;
        s[2] = seed + 0xBF58476D1CE4E5B9ULL;
        s[3] = seed + 0x94D049BB133111EBULL;
        
        for (int i = 0; i < 20; ++i) {
            (*this)();
        }
    }
    
    inline uint64_t operator()() {
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        
        return result;
    }
    
    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return UINT64_MAX; }
};

class EntropySource {
public:
    static double get_cpu_temperature();
    static std::vector<uint64_t> sample_timing_jitter(int n_samples);
    static std::vector<double> get_mixed_entropy(int n_samples);
    static std::string get_system_info();
    
private:
    static double read_windows_temperature();
    static double read_linux_temperature();
};

class ThermalNoise {
private:
    double temperature_K_;
    const double k_B = 1.380649e-23;
    Xoshiro256StarStar rng_;
    
public:
    ThermalNoise(double temperature_K);
    
    // Generate into pre-allocated buffer (FAST!)
    void generate_thermal_noise_inplace(double* output, int n_samples);
    
    // Original interface (for compatibility)
    std::vector<double> generate_thermal_noise(int n_samples);
    
    double get_thermal_variance() const;
    static double variance_to_temperature(double variance);
};

} // namespace entropy
