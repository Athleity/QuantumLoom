.. _code_operations:

Code operations
================

In the context of quantum error correction (QEC), code operations refer to the set of fundamental operations that can be performed on blocks. In the stabilizer formalism, all code blocks typically support a common set of operations which are defined in :mod:`~loom.eka.operations`. The implementations of these operations for general cases are provided in :mod:`~loom.interpreter.applicator` module.

The standard code operations supported by default are:

- :class:`~loom.eka.operations.code_operation.MeasureBlockSyndromes` :
    Measure all the syndromes of a code block into ancilla qubits.
- :class:`~loom.eka.operations.code_operation.MeasureLogicalX` :
    Measure the logical X operator of a code block into an ancilla qubit.
- :class:`~loom.eka.operations.code_operation.MeasureLogicalZ` :
    Measure the logical Z operator of a code block into an ancilla qubit.
- :class:`~loom.eka.operations.code_operation.MeasureLogicalY` :
    Measure the logical Y operator of a code block into an ancilla qubit.
- :class:`~loom.eka.operations.code_operation.LogicalX` (resp. Y or Z) :
    Apply the logical X (Y or Z) operator on a code block.
- :class:`~loom.eka.operations.code_operation.ResetAllDataQubits` :
    Reset all data qubits in a code block to the given state (\|0⟩ by default).
- :class:`~loom.eka.operations.code_operation.ResetAllAncillaQubits` :
    Reset all ancilla qubits in a code block to the given state (\|0⟩ by default).
- :class:`~loom.eka.operations.code_operation.StateInjection` :
    Inject a given resource state into a code block.
- :class:`~loom.eka.operations.code_operation.ConditionalLogicalX` (resp. Y or Z) :
    apply a logical X (Y or Z) operator on a code block conditioned on the state of a classical register.