from loom.executor import EkaToPennylaneConverter

# interpreted_eka: InterpretationStep
converter = EkaToPennylaneConverter()

# Convert the Eka circuit to QASM string representation
# is_catalyst: whether to convert to PennyLane Catalyst or standard PennyLane
# import_prefix: optional prefix for PennyLane imports (default is "qml.")
QASM_string, quantum_reg_mapping, classical_reg_mapping = converter.convert(
    interpreted_eka, is_catalyst=True, import_prefix="qml."
)
