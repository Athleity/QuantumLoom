from loom.executor import EkaCircuitToStimConverter

converter = EkaCircuitToStimConverter()

# interpreted_eka: InterpretationStep
stim_circuit, quantum_reg_mapping, classical_reg_mapping = converter.convert(
    interpreted_eka
)
