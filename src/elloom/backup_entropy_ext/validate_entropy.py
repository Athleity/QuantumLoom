"""Validate entropy quality with statistical tests"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import entropy_lattice as el

print("="*70)
print("ENTROPY QUALITY VALIDATION")
print("="*70)

# Create lattice
lattice = el.EntropyAwareLattice([100, 100], 2)

# Test 1: Distribution Analysis
print("\n1. DISTRIBUTION ANALYSIS")
print("-"*70)

n_samples = 100000
noise = lattice.generate_thermal_noise(n_samples)

print(f"Samples: {n_samples}")
print(f"Mean: {noise.mean():.6e} (should be ~0)")
print(f"Std:  {noise.std():.6e}")
print(f"Skewness: {stats.skew(noise):.6f} (should be ~0)")
print(f"Kurtosis: {stats.kurtosis(noise):.6f} (should be ~0)")

# Normality test
statistic, p_value = stats.normaltest(noise)
print(f"\nNormality test (D'Agostino-Pearson):")
print(f"  Statistic: {statistic:.4f}")
print(f"  P-value: {p_value:.6f}")
print(f"  Result: {'✓ Normal' if p_value > 0.01 else '✗ Not normal'}")

# Test 2: Autocorrelation
print("\n2. AUTOCORRELATION ANALYSIS")
print("-"*70)

# Calculate autocorrelation for lags 1-10
autocorr = []
for lag in range(1, 11):
    corr = np.corrcoef(noise[:-lag], noise[lag:])[0, 1]
    autocorr.append(corr)
    print(f"Lag {lag:2d}: {corr:+.6f} (should be ~0)")

max_autocorr = max(abs(c) for c in autocorr)
print(f"\nMax |autocorr|: {max_autocorr:.6f}")
print(f"Result: {'✓ Independent' if max_autocorr < 0.05 else '⚠ Some correlation'}")

# Test 3: Entropy Rate
print("\n3. SHANNON ENTROPY ESTIMATION")
print("-"*70)

# Discretize into bins
n_bins = 100
hist, bin_edges = np.histogram(noise, bins=n_bins, density=True)
hist = hist / hist.sum()  # Normalize

# Calculate Shannon entropy
shannon_entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Avoid log(0)
max_entropy = np.log2(n_bins)

print(f"Shannon entropy: {shannon_entropy:.4f} bits")
print(f"Max possible: {max_entropy:.4f} bits")
print(f"Efficiency: {100*shannon_entropy/max_entropy:.2f}%")

# Test 4: Temperature Dependence
print("\n4. TEMPERATURE SCALING TEST")
print("-"*70)

temps = [273.15, 298.15, 310.15, 350.15]  # 0°C, 25°C, 37°C, 77°C
theoretical_std = []
measured_std = []

for temp in temps:
    # Create lattice at specific temperature
    lattice_temp = el.EntropyAwareLattice([10, 10], 2)
    # Note: We can't set temperature directly yet, but we can analyze current temp
    
    # Theoretical Johnson-Nyquist: σ = √(k_B * T)
    k_B = 1.380649e-23
    theory = np.sqrt(k_B * temp)
    theoretical_std.append(theory)
    
    print(f"T = {temp-273.15:6.2f}°C: σ_theory = {theory:.6e}")

print("\nCurrent system measurement:")
current_temp = lattice.get_system_temperature()
measured_noise = lattice.generate_thermal_noise(10000)
measured = measured_noise.std()
theoretical = np.sqrt(1.380649e-23 * current_temp)

print(f"Temperature: {current_temp:.2f} K")
print(f"Theoretical σ: {theoretical:.6e}")
print(f"Measured σ:    {measured:.6e}")
print(f"Ratio: {measured/theoretical:.4f} (should be ~1.0)")

# Test 5: System Entropy Quality
print("\n5. SYSTEM ENTROPY (Timing Jitter) ANALYSIS")
print("-"*70)

sys_entropy = lattice.sample_system_entropy(10000)
print(f"Samples: {len(sys_entropy)}")
print(f"Mean: {sys_entropy.mean():.6f}")
print(f"Std:  {sys_entropy.std():.6f}")
print(f"Min:  {sys_entropy.min():.6f}")
print(f"Max:  {sys_entropy.max():.6f}")
print(f"Range: {sys_entropy.max() - sys_entropy.min():.6f}")

# Coefficient of variation
cv = sys_entropy.std() / sys_entropy.mean() if sys_entropy.mean() > 0 else 0
print(f"Coeff. of variation: {cv:.4f}")

# Summary
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

checks = [
    ("Normal distribution", p_value > 0.01),
    ("Low autocorrelation", max_autocorr < 0.05),
    ("Johnson-Nyquist scaling", abs(measured/theoretical - 1.0) < 0.1),
    ("System entropy variance", cv > 0.01)  # Should have variation
]

passed = sum(1 for _, check in checks if check)
total = len(checks)

for name, result in checks:
    status = "✓" if result else "✗"
    print(f"{status} {name}")

print(f"\nPassed: {passed}/{total}")
print("="*70)
