import fast_lattice_entropy as fle

print("Available attributes in fast_lattice_entropy:")
print(dir(fle))
print("\nFastLatticeEntropy class methods:")
print(dir(fle.FastLatticeEntropy))

# Try to instantiate it
try:
    entropy = fle.FastLatticeEntropy()
    print("\n✓ FastLatticeEntropy instantiated successfully")
    print("\nMethods:")
    for attr in dir(entropy):
        if not attr.startswith('_'):
            print(f"  - {attr}")
except Exception as e:
    print(f"\n✗ Error instantiating: {e}")
