import entropy_lattice as el

# Create entropy-aware lattice
lattice = el.EntropyAwareLattice([100, 100], 2)

# Get system info
print(lattice.get_system_info())

# Get temperature
temp = lattice.get_system_temperature()
print(f"System temperature: {temp:.2f} K ({temp-273.15:.2f}°C)")

# Sample entropy
entropy = lattice.sample_system_entropy(1000)
print(f"Entropy samples: {entropy.shape}, mean={entropy.mean():.3f}")

# Generate thermal noise
noise = lattice.generate_thermal_noise(10000)
print(f"Thermal noise: mean={noise.mean():.2e}, std={noise.std():.2e}")

# Get metrics
metrics = lattice.get_entropy_metrics()
print(f"Metrics: {metrics}")
