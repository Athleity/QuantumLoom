.. _validator:

Validator
==============

The :mod:`~loom.validator` module provides tools for validating the functionality of Clifford quantum circuits, particularly user-defined Quantum Error Correction (QEC) circuits. By running stabilizer tableau simulations for a variety of input states, users can ensure that their circuits behave as intended in a noise-free environment. The validator checks for three main types of circuit behavior:

1. **Code-switching**: Correct conversion of code stabilizers.
2. **Logical operations**: Proper alteration of logical states.
3. **Syndrome extraction**: Accurate measurement of code stabilizers.

If there are discrepancies between the simulated and expected outputs, the module provides a detailed debugging report to help identify and resolve issues.


Main Validation Function
^^^^^^^^^^^^^^^^^^^^^^^^

The main function of the validator is :func:`~loom.validator.circuit_validation.is_circuit_valid`, which has the following signature:

.. code-block:: python

    is_circuit_valid(
        circuit: Circuit,
        input_block: Block | tuple[Block, ...],
        output_block: Block | tuple[Block, ...],
        output_stabilizers_parity: dict[Stabilizer, tuple[str | int, ...]],
        output_stabilizers_with_any_value: list[Stabilizer],
        logical_state_transformations_with_parity: dict[
            LogicalState,
            tuple[LogicalState, dict[int, tuple[str | int, ...]]],
        ],
        logical_state_transformations: list[tuple[LogicalState, tuple[LogicalState, ...]]],
        measurement_to_input_stabilizer_map: dict[int, Stabilizer],
    ) -> DebugData

Parameters:

- ``circuit``: The circuit to be tested. The labels of the qubit channels should match the labels of the qubits of the input/output :class:`~loom.eka.block.Block` objects.
- ``input_block``: Describes the input code object(s) on which the circuit acts.
- ``output_block``: Describes the expected output code object(s) into which the input shall be transformed.
- ``output_stabilizers_parity``: If some of the stabilizers of the output are not deterministically projected but dependent on the parity of some classical channels (``str``) and constant parity changes (``int``), it can be described here.
- ``output_stabilizers_with_any_value``: If some of the stabilizers of the output are not deterministically projected, then the constraint of them being on some specific value in the output can be relaxed here.
- ``logical_state_transformations_with_parity``: Describes the logical transformations that the circuit should implement with the additional information of classical channels (``str``) and constant parity changes (``int``). Each logical operator in the output corresponds to a specific parity change definition.
- ``logical_state_transformations``: Describes the logical transformations that the circuit should implement. Each element of the list is a tuple of an input logical state that should get mapped to one of the logical states in a tuple of output logical states.
- ``measurement_to_input_stabilizer_map``: Defines where some stabilizers have been measured. Each element of the dictionary maps the index of a measurement operation within the circuit to the stabilizer that it should be measuring.


Note that any of the above criteria can be trivially set. For example:

- A syndrome extraction circuit is a trivial code-switch (input code is the same as output code) and acts trivially on the logical states (input logical state is the same as output logical state). Only syndrome extraction occurs.
- A logical operation circuit is a trivial code-switch and does not measure any stabilizers. Only the logical states are altered.



Interpreting Debug Data
^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~loom.validator.debug_dataclass.DebugData` is a dataclass containing all the necessary information to debug the circuit.

If the circuit is valid and fulfills all the criteria defined by the user, then ``debug_data.valid`` will be ``True``. If not, then at least one of the ``debug_data.checks`` has failed to pass.

Code Stabilizer Check
-----------------------------

If the circuit acting on the input code leads to an output that does not contain all the stabilizers of the ``output_block``, then the check fails. If the test fails, the debug data field will contain all of the stabilizers that were not found in the output. This will be found in the :class:`~loom.validator.check_code_stabilizers.CodeStabilizerCheck` dataclass.

*How this is checked:* The validator initializes the input logical state :math:`|00...0⟩`, runs the circuit on it, and checks if the output state contains each of the ``output_block`` stabilizers. It allows for the negative-valued stabilizer to pass if it's in the ``output_stabilizers_with_any_value`` argument.


Logical Transformation Check
----------------------------

If any input state is not mapped to one of the desired output states, the check fails. The debug data field of this check will return all of the logical transformations that failed, in the same way that they were defined by the user in the :class:`~loom.validator.check_logical_ops.LogicalOperatorCheck` dataclass.

*How this is checked:* The input logical state is initialized, and then the circuit acts on it. If the output does not contain all the logical operators of one of the output logical states, then the ``tuple[LogicalState, tuple[LogicalState, ...]]`` is returned.


Stabilizer Measurement Check
----------------------------

If any of the measurement operations do not measure their assigned stabilizer of the ``input_block``, then the check fails. This correspondence between :class:`~loom.eka.stabilizer.Stabilizer` and measurement operations can be defined using the argument ``measurement_to_input_stabilizer_map``.

The output :class:`~loom.validator.check_stabilizer_measurement.StabilizerMeasurementCheck` of this check will contain, for every failed measurement index, the stabilizers that were measured instead. If a measurement was probabilistic, then this is also considered erroneous, and the index is added to a specific list of the debug data field.

*How this is checked:* The validator always selects the :math:`|00...0⟩` state as input logical state. It runs the circuit once and records the value of the measurement results at the specified indices. Then, it reruns the circuit by flipping one-by-one all of the stabilizers. If by flipping one stabilizer :math:`S_i`, the value of a measurement :math:`j` flips, then the measurement :math:`j` records contributions from the stabilizer :math:`S_i`. Then it compares if what is measured is indeed what was specified by the user in the ``measurement_to_input_stabilizer_map``.



Utility Functions
^^^^^^^^^^^^^^^^^^

To facilitate the user experience, several utilities and wrapper functions of :func:`~loom.validator.is_circuit_valid` have been introduced.

Getting Logical Transformations for a Clifford Operation
--------------------------------------------------------

A Clifford Operation is often described by how it transforms the basis logical operators :math:`X_i` and :math:`Z_i`. For example, a :math:`\text{CNOT}_{0→1}` operation will cause the transformation:

.. math::
    X_0 → X_0X_1, \quad X_1 → X_1, \quad Z_0 → Z_0, \quad Z_1 → Z_0Z_1 

The validator allows users to translate this into a list of logical state transformations to be used as input using the function :func:`~loom.validator.utilities.logical_state_transformations_to_check`:

.. code-block:: python

    logical_state_transformations_to_check(
        x_operators_sparse_pauli_map: list[str], 
        z_operators_sparse_pauli_map: list[str]
    ) -> list[tuple[LogicalState, tuple[LogicalState]]]:

The underlying assumption is that we can check the set of all of the following input states:

.. code-block:: text

    |00...0⟩
    |++...+⟩
    |+0...0⟩, |0+0...0⟩, ..., |00...+⟩
    |0+...+⟩, |+0+...+⟩, ..., |++...0⟩

By specifying how the :math:`X_i` and :math:`Z_i` are transformed, we can deduce the expected transformation of each of those states.

Syndrome Extraction Circuit Wrapper
------------------------------------------

We can easily validate syndrome extraction circuits using the wrapper function :func:`~loom.validator.circuit_validation_wrappers.is_syndrome_extraction_circuit_valid`:

.. code-block:: python

    is_syndrome_extraction_circuit_valid(
        circuit: Circuit,
        input_block: Block | tuple[Block, ...],
        measurement_to_input_stabilizer_map: dict[int, Stabilizer],
    ) -> DebugData:

This function will check that the circuit acts as an identity on the logical level while not altering any of the code stabilizers. Users can define where each stabilizer is expected to be measured.

Logical Operation Circuit Wrapper
------------------------------------------

We can easily check logical Clifford operation circuits using the wrapper :func:`~loom.validator.circuit_validation_wrappers.is_logical_operation_circuit_valid`:

.. code-block:: python

    is_logical_operation_circuit_valid(
        circuit: Circuit,
        input_block: Block | tuple[Block, ...],
        x_operators_sparse_pauli_map: list[str],
        z_operators_sparse_pauli_map: list[str],
    ) -> DebugData:

It's assumed that no stabilizers are being measured and the code stabilizers are not altered. Users can define the operation in terms of how the :math:`X_i` and :math:`Z_i` logical operators should be transformed.

Usage of this function is illustrated in the :doc:`example notebook </notebooks/validator_syndrome_extraction>`.

Limitations and Caveats
^^^^^^^^^^^^^^^^^^^^^^^^

Although the validator is quite robust in its operation, it is worth noting some limitations that stem from the way it works.

Inconsistencies in Probabilistic Circuits
------------------------------------------

Sometimes, if user input is not well thought out, it may result in false (in)validity results and inconsistent results among validator runs. Let us illustrate this through the example of a GROW lattice surgery operation.

*Example:* In the GROW operation, some of the new stabilizers are projected probabilistically onto their +/- value. Now say the user defines the circuit and all other inputs to the validator correctly *except* they forget to include those probabilistic :class:`~loom.eka.stabilizer.Stabilizer` objects in the ``output_stabilizers_with_any_value`` argument. When running the validator, sometimes the :class:`~loom.eka.stabilizer.Stabilizer` will be projected onto ``+`` and this will pass the check, but sometimes it will be projected onto ``-`` and it will fail to pass it.

Similarly, this can happen if a :class:`~loom.eka.logical_state.LogicalState` can get mapped to more than one :class:`~loom.eka.logical_state.LogicalState` but the user fails to mention it.

Probabilistic circuits can lead to:

- False valid results if the circuit is wrong
- False invalid results if the rest of the arguments are wrong

Repeating the validator run should asymptotically eliminate these issues, especially since probabilistic results are 50% biased in stabilizer simulations.

Non-Proven States to Validate Clifford Operations
-----------------------------------------------------------

As mentioned previously, because we are working with full states, we cannot simply propagate the basis logical operators and check whether the action of the circuit on them is the expected one. Instead, we select the set of input logical states:

.. code-block:: text

    |00...0⟩
    |++...+⟩
    |+0...0⟩, |0+0...0⟩, ..., |00...+⟩
    |0+...+⟩, |+0+...+⟩, ..., |++...0⟩

and verify that these are transformed as expected. This selection is an unproven heuristic such that all the X/Z states are mixed and the action of the Clifford operation is fully encapsulated.

Because this is not proven or disproven yet, it should not be 100% trusted.
