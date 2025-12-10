Backends
========
Loom supports conversion to different formats to run quantum experiments, allowing users to easily compare and choose the most suitable platform for their needs. These tools are provided in the :mod:`~loom.executor` package. It also provides a flexible way to define circuit error models that can be integrated into the execution of experiments.

Circuit Error Model
-------------------

The :class:`~executor.circuit_error_model.CircuitErrorModel` (also referred to as "CEM") is designed to represent an error model applied to a circuit. It can represent a wide range of error models and is meant to be used as a base class to specify any error model.

An error model instance is tied to a circuit instance and has 3 constant parameters that must be defined:

1. **is_time_dependent**: This is a boolean specifying whether the model is time-dependent or not. A time-dependent model will have its error probabilities defined as functions of time.
2. **error_type**: One error model class is meant to encode only one type of error operation. This must be a member of the :class:`~executor.circuit_error_model.ErrorType` enum specified in the module. This typically includes Pauli flip, Pauli channel, depolarization, etc.
3. **application_mode**: This specifies how the error is applied to the circuit. It must be a member of the :class:`~executor.circuit_error_model.ApplicationMode` enum. The supported application modes are: before gates, after gates, end of tick, and idle tick-wise. The first two are intuitively applied gate-wise, the third applies error to all qubits on a tick basis after all the gates within the given tick have been executed. The idle tick-wise applies noise based on the time each channel spends as "idle" (meaning no gate is applied to it) at each tick (noise instruction is applied at the end of the tick).

The error probabilities are given as a function that may take time (float) parameters. The return type is a list of probabilities, as some error types require multiple probabilities (the expected number of values are specified for each :class:`~executor.circuit_error_model.ErrorType`). 

If the error is applied gate-wise, it must be defined for each gate type (x, cnot, h, ...) in ``gate_error_probabilities``. These are given as a dictionary that maps gate names (strings, similar to those used in Eka circuits) to callables that can take one float parameter for time from the beginning of the circuit and return the probability. When creating an instance, zero probability (probability of 0.0) will be assigned to gates that aren't included in the mapping. 

If the error application mode is tick-wise, one must provide a single probability function that may depend on the time from the start of the circuit and/or on some tick-related time parameter (specified by the application mode).

If the model is time dependent, one must define a duration (float value) for each gate type used in the circuit. This is stored in the ``gate_durations`` attribute. A validation check will throw an error if a gate in the circuit has no duration defined in a time-dependent model.

The error model class provides the following interface:

- :meth:`get_gate_error_probability(gate: Circuit) <~executor.circuit_error_model.CircuitErrorModel.get_gate_error_probability>`: Provides the error probability given a gate (Circuit with empty circuit attribute). Used if the error is applied gate-wise.
- :meth:`get_tick_error_probability(tick_index: int) <~executor.circuit_error_model.CircuitErrorModel.get_tick_error_probability>`: Provides the error probability for a given tick index. Used if the error is applied with :attr:`~executor.circuit_error_model.ApplicationMode.END_OF_TICK` or :attr:`~executor.circuit_error_model.ApplicationMode.IDLE_END_OF_TICK`.

Homogeneous Error Models
~~~~~~~~~~~~~~~~~~~~~~~~

The module provides subclasses to define error models that are homogeneous with respect to a subset of gates. Two subclasses are provided for both time-dependent and time-independent cases:

- :class:`~executor.circuit_error_model.HomogeneousTimeIndependentCEM`
- :class:`~executor.circuit_error_model.HomogeneousTimeDependentCEM`

These allow for easier definition of commonly used error models. They only require specifying an error type, the subset of gates concerned, and the error probability.

For example, one can define a model that applies bit-flip before measurement with probability 0.05:

.. code-block:: python

   model = HomogeneousTimeIndependentCEM(
       circuit=any_crd_circuit,
       error_type=ErrorType.BIT_FLIP,
       application_mode=ApplicationMode.BEFORE_GATE,
       error_probability=0.05,
       target_gates=["measure_x", "measure_y", "measure_z"]
   )

Or a time-dependent model to apply depolarization after Pauli gates:

.. code-block:: python

   model = HomogeneousTimeDependentCEM(
       circuit=any_crd_circuit,
       error_type=ErrorType.DEPOLARIZING1,
       application_mode=ApplicationMode.AFTER_GATE,
       error_probability=lambda t: [t*0.05],
       target_gates=["x", "y", "z"],
       gate_durations=dict(...)  # This must define durations for each type of gate used in the circuit.
   )

We encourage users to define their own specific error models as subclasses for more clarity. One can override the functions to get probabilities if necessary.

Usage in Executor
~~~~~~~~~~~~~~~~~

The error models can be used in converters to run experiments. Currently, only the Stim converter supports error models. The model is given as parameter to the convert function. In order to define more complex models, one can provide a list of models that will be stacked in an additive manner.

For example, to define a model that applies depolarization to all Clifford gates, some bit-flip to reset operations, and before measurement, one can do the following:

.. code-block:: python

   circ = Circuit(...)

   class FlipBeforeMeasurement(HomogeneousTimeIndependentCEM):
       """Flip error model applied before measurement gates."""

       error_probability: float
       application_mode: ApplicationMode = ApplicationMode.BEFORE_GATE
       error_type: ErrorType = ErrorType.PAULI_X
       target_gates: list[str] = ["measurement", "measure_z", "measure_x", "measure_y"]


   class CliffordDepolarization1(HomogeneousTimeIndependentCEM):
       """Clifford depolarization error model applied to single-qubit gates."""

       error_probability: float
       application_mode: ApplicationMode = ApplicationMode.AFTER_GATE
       error_type: ErrorType = ErrorType.DEPOLARIZING1
       target_gates: list[str] = ["x", "y", "z", "h", "hadamard", "i", "identity"]


   class CliffordDepolarization2(HomogeneousTimeIndependentCEM):
       """Clifford depolarization error model applied to two-qubit gates."""

       error_probability: float
       application_mode: ApplicationMode = ApplicationMode.AFTER_GATE
       error_type: ErrorType = ErrorType.DEPOLARIZING2
       target_gates: list[str] = ["cx", "cz", "cy", "cnot", "swap"]


   class FlipAfterReset(HomogeneousTimeIndependentCEM):
       """Flip error model applied after reset gates."""

       error_probability: float
       application_mode: ApplicationMode = ApplicationMode.AFTER_GATE
       error_type: ErrorType = ErrorType.PAULI_X
       target_gates: list[str] = [
           "reset",
           "reset_0",
           "reset_1",
           "reset_+",
           "reset_-",
           "reset_+i",
           "reset_-i",
       ]


   # Define error models to be applied to the circuit
   error_models = [
       FlipBeforeMeasurement(circuit=circ, error_probability=0.01),
       CliffordDepolarization1(circuit=circ, error_probability=0.01),
       CliffordDepolarization2(circuit=circ, error_probability=0.01),
       FlipAfterReset(circuit=circ, error_probability=0.01),
   ]

   # This can then be passed to the converter
   output = converter.convert(
       interpreted_eka=..., error_models=error_models
   )

OpenQASM 3.0
------------
`OpenQASM3 <https://github.com/openqasm/openqasm>`_ is an imperative programming language for describing quantum circuits. :class:`~loom.executor.eka_to_qasm_converter.EkaToQasmConverter` allows you to convert Loom experiments into OpenQASM3 format, enabling execution on any platform that supports OpenQASM3. The following example shows how to convert a Loom's :class:`~loom.interpreter.interpretation_step.InterpretationStep` into a OpenQASM3 string:

.. literalinclude:: ../../python/executor/qasm_conversion.py
   :language: python

Stim
----
`Stim <https://github.com/quantumlib/Stim>`_ is an open-source tool for high-performance simulation of quantum stabilizer circuits. Loom experiments can be converted to Stim format using :class:`~loom.executor.eka_to_stim_converter.EkaToStimConverter`. This allows for efficient simulation and analysis of quantum circuits. The following example shows how to convert an :class:`~loom.interpreter.interpretation_step.InterpretationStep` to Stim (check the list of Stim's supported operations `here <https://github.com/quantumlib/Stim/blob/main/doc/gates.md>`_):

.. literalinclude:: ../../python/executor/stim_conversion.py
   :language: python

Pennylane
---------
`Pennylane <https://github.com/PennyLaneAI/pennylane>`_ is an open-source python framework for quantum programming built by Xanadu. Loom experiments can be converted to a format that is compatible with Pennylane's simulators using :class:`~loom.executor.eka_to_pennylane_converter.EkaToPennylaneConverter`. The output format can also be used for Pennylane's catalyst simulator via the ``is_catalyst`` input boolean. Note that in order to use this exector with loom, you are required to have had installed :mod:`pennylane` and :mod:`catalyst` beforehand. We recommend getting `pennylane-catalyst` version `0.13.0`:

.. literalinclude:: ../../python/executor/pennylane_conversion.py
   :language: python

Cudaq
-----
`CudaQ <https://github.com/NVIDIA/cuda-quantum>`_ is an open-source quantum development platform for the orchestration of software and hardware resources designed for large-scale quanatum computing applications built by Nvidia. Loom experiments can be converted to a format that is compatible with the cudaq simulators using :class:`~loom.executor.eka_to_cudaq_converter.EkaToCudaqConverter`. Note that in order to use this executor with loom, you are required to have had installed :mod:`cudaq` beforehand. We recommend getting :mod:`cudaq` and :mod:`cuda-quantum-cu12` at version `0.12.0`:

.. literalinclude:: ../../python/executor/cudaq_conversion.py
   :language: python
