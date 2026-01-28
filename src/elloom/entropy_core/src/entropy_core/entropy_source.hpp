#pragma once
#include <vector>
#include <cstdint>
#include <chrono>
#include <string>
#include <cmath>
#include <memory>
#include <algorithm>

namespace entropy {

// ============================================================================
// SplitMix64 RNG - Ultra-fast PRNG
// ============================================================================
class SplitMix64 {
private:
    uint64_t state_;

public:
    using result_type = uint64_t;

    explicit SplitMix64(uint64_t seed = 1) : state_(seed) {
        if (state_ == 0) state_ = 1;
    }

    inline uint64_t operator()() {
        uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    inline double next_double() {
        return (operator()() >> 11) * 0x1.0p-53;
    }

    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return UINT64_MAX; }
};

// ============================================================================
// FastZiggurat - Optimized Ziggurat algorithm for Gaussian generation
// ============================================================================
class FastZiggurat {
private:
    static constexpr int N = 128;
    static constexpr double R = 3.442619855899;

    alignas(64) double ktab[N];
    alignas(64) double wtab[N];
    alignas(64) double ftab[N];

    SplitMix64 rng_;

    void init_tables() {
        double m = 2147483648.0;
        double dn = R, tn = dn;
        double vn = 9.91256303526217e-3;
        double q = vn / std::exp(-0.5 * dn * dn);

        ktab[0] = (dn / q) * m;
        ktab[1] = 0;

        wtab[0] = q / m;
        wtab[N-1] = dn / m;

        ftab[0] = 1.0;
        ftab[N-1] = std::exp(-0.5 * dn * dn);

        for (int i = N - 1; i >= 1; i--) {
            dn = std::sqrt(-2.0 * std::log(vn / dn + std::exp(-0.5 * dn * dn)));
            ktab[i+1] = (dn / tn) * m;
            tn = dn;
            ftab[i] = std::exp(-0.5 * dn * dn);
            wtab[i] = dn / m;
        }
    }

    inline double generate() {
        while (true) {
            uint64_t u = rng_();
            int i = u & 0x7F;
            int32_t j = (int32_t)(u >> 32);
            double x = j * wtab[i];

            uint32_t abs_j = (j < 0) ? -j : j;
            if (abs_j < ktab[i]) return x;

            if (i == 0) {
                double xx, yy;
                do {
                    xx = -std::log(rng_.next_double() + 1e-10) / R;
                    yy = -std::log(rng_.next_double() + 1e-10);
                } while (yy + yy < xx * xx);
                return (j < 0) ? (R + xx) : -(R + xx);
            }

            if (ftab[i+1] + rng_.next_double() * (ftab[i] - ftab[i+1]) <
                std::exp(-0.5 * x * x)) {
                return x;
            }
        }
    }

public:
    FastZiggurat(uint64_t seed) : rng_(seed) {
        init_tables();
    }

    inline void generate_batch(double* out, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            out[i] = generate();
        }
    }
};

// ============================================================================
// NoisePool - Pre-generation pool for amortized cost
// ============================================================================
class NoisePool {
private:
    std::vector<double> pool_;
    size_t position_;
    size_t pool_size_;
    double stddev_;
    std::unique_ptr<FastZiggurat> generator_;

    void refill() {
        generator_->generate_batch(pool_.data(), pool_size_);
        position_ = 0;
    }

public:
    NoisePool(double stddev, size_t pool_size = 4194304)  // 4M default
        : pool_(pool_size),
          position_(pool_size),  // Force initial refill
          pool_size_(pool_size),
          stddev_(stddev),
          generator_(std::make_unique<FastZiggurat>(
              std::chrono::high_resolution_clock::now().time_since_epoch().count())) {
    }

    inline void get_noise(double* out, size_t n) {
        size_t copied = 0;

        while (copied < n) {
            if (position_ >= pool_size_) {
                refill();
            }

            size_t available = pool_size_ - position_;
            size_t to_copy = std::min(available, n - copied);

            double scale = stddev_;
            for (size_t i = 0; i < to_copy; ++i) {
                out[copied + i] = pool_[position_ + i] * scale;
            }

            position_ += to_copy;
            copied += to_copy;
        }
    }

    inline double get_single() {
        if (position_ >= pool_size_) {
            refill();
        }
        return pool_[position_++] * stddev_;
    }
    
    void warmup() {
        if (position_ >= pool_size_) {
            refill();
        }
    }
};

// ============================================================================
// EntropySource - System entropy collection
// ============================================================================
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

// ============================================================================
// ThermalNoise - Thermal noise generator with pooling
// ============================================================================
class ThermalNoise {
private:
    double temperature_K_;
    const double k_B = 1.380649e-23;
    std::unique_ptr<NoisePool> pool_;

public:
    ThermalNoise(double temperature_K)
        : temperature_K_(temperature_K) {
        double variance = k_B * temperature_K_;
        double stddev = std::sqrt(variance);
        pool_ = std::make_unique<NoisePool>(stddev, 4194304);  // 4M pool
        pool_->warmup();
    }

    inline void generate_thermal_noise_inplace(double* output, int n_samples) {
        pool_->get_noise(output, static_cast<size_t>(n_samples));
    }

    std::vector<double> generate_thermal_noise(int n_samples) {
        std::vector<double> noise(n_samples);
        generate_thermal_noise_inplace(noise.data(), n_samples);
        return noise;
    }

    double get_thermal_variance() const {
        return k_B * temperature_K_;
    }

    static double variance_to_temperature(double variance) {
        const double k_B = 1.380649e-23;
        return variance / k_B;
    }
};

} // namespace entropy
