.. _cliffordsim:

CliffordSim 
==============

The :mod:`~loom.cliffordsim` module provides a simulator for quantum circuits composed of Clifford gates.
It was designed with a focus on efficient analysis of quantum error correction codes.
The simulator leverages software design techniques that enable future performance improvements.

The simulator is based on the `CHP simulator <https://arxiv.org/abs/quant-ph/0406196>`_ by Aaronson and Gottesman.



Key Features
-----------------

- **Tableau State Propagation**: The simulator uses tableau representations to track the evolution of stabilizer states through Clifford operations.
- **Pauli Frame Tracking**: Allows the user to define and track Pauli frames, which are essential for understanding error syndromes in quantum error correction and assessing the impact of errors on logical qubits.


Usage
-----------------

The basic object of the module is the :class:`~loom.cliffordsim.engine.Engine`, which manages the state of the quantum system and processes operations applied to it.
Users interact with the CliffordSim it through :class:`~loom.cliffordsim.operations.base_operation.Operation` objects.
The result of executing these operations can be inspected through the :class:`~loom.cliffordsim.data_store.DataStore` class.


Operations
^^^^^^^^^^^
The module provides a variety of operations that can be applied to the quantum state, including:

- **Gate Operations**: Standard Clifford gates such as `Hadamard`, `CNOT`, and `Phase`.
- **Measurement Operations**: Projective measurements in any basis or resetting qubits to specific states.
- **Resize Operations**: Dynamically add or remove qubits from the simulation.
- **Classical Control Operations**: Conditional operations based on measurement outcomes, enabling the simulation of adaptive circuits.
- **Classical Operations**: Basic classical operations on bits, such as `AND`, `OR`, and `NOT`.
- **Data Manipulation Operations**: Operations to create and record Pauli frames.


DataStore
^^^^^^^^^^^^^
The :class:`~loom.cliffordsim.data_store.DataStore` can be found as an attribute of the :class:`~loom.cliffordsim.engine.Engine` after execution as `Engine.data_store`.

It contains all records of:

- Measurement outcomes
- Classical register values
- Propagation of Pauli frames



To see more and have a deeper look into the data structures, see the  :doc:`CliffordSim example </notebooks/cliffordsim_example>` notebook.