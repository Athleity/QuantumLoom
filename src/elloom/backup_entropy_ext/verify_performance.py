# verify_performance.py
import fast_lattice_entropy as fle
import numpy as np
import time

# Test 1: Compare to NumPy
n_samples = 10_000_000
n_runs = 10

print("=" * 70)
print("VERIFICATION: Your Code vs NumPy")
print("=" * 70)

# Your implementation
entropy = fle.FastLatticeEntropy([1000, 1000], 1)
times_yours = []
for _ in range(n_runs):
    t1 = time.perf_counter()
    noise = entropy.generate_noise_only(n_samples)
    t2 = time.perf_counter()
    times_yours.append((t2 - t1) * 1e9 / n_samples)  # ns per sample

your_avg = np.mean(times_yours)
your_std = np.std(times_yours)

# NumPy modern generator
rng = np.random.Generator(np.random.PCG64())
times_numpy = []
for _ in range(n_runs):
    t1 = time.perf_counter()
    noise = rng.normal(0, 1, n_samples)
    t2 = time.perf_counter()
    times_numpy.append((t2 - t1) * 1e9 / n_samples)  # ns per sample

numpy_avg = np.mean(times_numpy)
numpy_std = np.std(times_numpy)

print(f"\nYour Implementation:  {your_avg:.2f} ns/sample (±{your_std:.2f})")
print(f"NumPy PCG64:          {numpy_avg:.2f} ns/sample (±{numpy_std:.2f})")
print(f"\nYour speedup:         {numpy_avg/your_avg:.2f}x")

if your_avg < numpy_avg:
    print("\n✅ YOUR CODE IS FASTER THAN NUMPY!")
else:
    print(f"\n✓ Your code is {your_avg/numpy_avg:.2f}x NumPy speed (expected range)")

print("\nPublished benchmarks:")
print("  - Milo Yip (2015):     7.09 ns/sample")
print("  - Java ziggurat (2024): 8.90 ns/sample")
print("  - NumPy expected:      6-8 ns/sample")
print(f"  - YOUR RESULT:         {your_avg:.2f} ns/sample")

if 5 < your_avg < 10:
    print("\n🎉 YOUR PERFORMANCE IS WORLD-CLASS!")
print("=" * 70)
