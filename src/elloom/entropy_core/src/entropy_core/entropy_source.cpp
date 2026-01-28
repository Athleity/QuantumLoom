#include "entropy_source.hpp"
#include <sstream>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#else
#include <cpuid.h>
#include <unistd.h>
#endif

namespace entropy {

double EntropySource::get_cpu_temperature() {
#ifdef _WIN32
    return read_windows_temperature();
#else
    return read_linux_temperature();
#endif
}

double EntropySource::read_windows_temperature() {
#ifdef _WIN32
    SYSTEM_POWER_STATUS powerStatus;
    if (GetSystemPowerStatus(&powerStatus)) {
        return 298.15 + (powerStatus.ACLineStatus ? 5.0 : 0.0);
    }
#endif
    return 298.15;
}

double EntropySource::read_linux_temperature() {
    std::ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");
    if (temp_file.is_open()) {
        int temp_millidegrees;
        temp_file >> temp_millidegrees;
        return 273.15 + (temp_millidegrees / 1000.0);
    }
    return 298.15;
}

std::vector<uint64_t> EntropySource::sample_timing_jitter(int n_samples) {
    std::vector<uint64_t> samples(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        volatile int x = 0;
        for (int j = 0; j < 10; ++j) x += j;
        auto t2 = std::chrono::high_resolution_clock::now();
        samples[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
    return samples;
}

std::vector<double> EntropySource::get_mixed_entropy(int n_samples) {
    auto jitter = sample_timing_jitter(n_samples);
    std::vector<double> mixed(n_samples);
    double temp = get_cpu_temperature();
    for (int i = 0; i < n_samples; ++i) {
        mixed[i] = static_cast<double>(jitter[i]) * temp;
    }
    return mixed;
}

std::string EntropySource::get_system_info() {
    std::ostringstream info;
    info << "CPU Temperature: " << get_cpu_temperature() << " K\n";
    auto jitter = sample_timing_jitter(10);
    info << "Timing jitter samples: ";
    for (auto j : jitter) info << j << " ";
    return info.str();
}

} // namespace entropy
