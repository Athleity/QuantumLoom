Steane Code Factory
===================
Steane Code Block
-----------------
The :class:`~loom_steane_code.code_factory.steane_code.SteaneCode` is a simple :math:`[7, 1, 3]` error-correcting code that encodes a single logical qubit with :math:`7` physical qubits.

.. code-block:: python
    
    from loom_steane_code.code_factory import SteaneCode
    from loom.eka import Lattice

    lattice = Lattice.square_2d((10, 10))

    # Create a block for a steane code
    myLogicalqubit = SteaneCode.create(
        lattice=lattice,
        unique_label="steane_code",
        position=(0, 0),
    )


Operations on Steane Code
-------------------------
We currently do not provide any special code operations specific to the Steane code. 
The operations available are the ones defined in loom's :mod:`~loom.eka.operations` module.