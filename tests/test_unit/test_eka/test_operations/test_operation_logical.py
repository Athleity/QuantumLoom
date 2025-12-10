"""
Copyright 2024 Entropica Labs Pte Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from loom.eka.utilities import SingleQubitPauliEigenstate, loads, dumps
from loom.eka.operations import (
    Operation,
    Reset,
    CNOT,
    Hadamard,
    Phase,
    PhaseInverse,
    X,
    Y,
    Z,
    T,
)


def test_creation():
    """Test the creation of logical operations."""
    # Test the creation of Reset
    reset = Reset(
        target_qubit="q",
        state=SingleQubitPauliEigenstate.ZERO,
    )
    assert reset.target_qubit == "q"
    assert reset.state == SingleQubitPauliEigenstate.ZERO
    assert reset.__class__.__name__ == "Reset"
    # Test the loads/dumps both using the right class and the abstract base class
    assert reset == loads(Reset, dumps(reset))
    assert reset == loads(Operation, dumps(reset))

    # Test the creation of CNOT
    cnot = CNOT(
        target_qubit="t",
        control_qubit="c",
    )
    assert cnot.target_qubit == "t"
    assert cnot.control_qubit == "c"
    assert cnot.__class__.__name__ == "CNOT"
    # Test the loads/dumps both using the right class and the abstract base class
    assert cnot == loads(CNOT, dumps(cnot))
    assert cnot == loads(Operation, dumps(cnot))

    # Test the creation of Hadamard
    hadamard = Hadamard(target_qubit="q")
    assert hadamard.target_qubit == "q"
    assert hadamard.__class__.__name__ == "Hadamard"
    # Test the loads/dumps both using the right class and the abstract base class
    assert hadamard == loads(Hadamard, dumps(hadamard))
    assert hadamard == loads(Operation, dumps(hadamard))

    # Test the creation of Phase
    phase = Phase(target_qubit="q")
    assert phase.target_qubit == "q"
    assert phase.__class__.__name__ == "Phase"
    # Test the loads/dumps both using the right class and the abstract base class
    assert phase == loads(Phase, dumps(phase))
    assert phase == loads(Operation, dumps(phase))

    # Test the creation of PhaseInverse
    phase_inverse = PhaseInverse(target_qubit="q")
    assert phase_inverse.target_qubit == "q"
    assert phase_inverse.__class__.__name__ == "PhaseInverse"
    # Test the loads/dumps both using the right class and the abstract base class
    assert phase_inverse == loads(PhaseInverse, dumps(phase_inverse))
    assert phase_inverse == loads(Operation, dumps(phase_inverse))

    # Test the creation of X
    x = X(target_qubit="q")
    assert x.target_qubit == "q"
    assert x.__class__.__name__ == "X"
    # Test the loads/dumps both using the right class and the abstract base class
    assert x == loads(X, dumps(x))
    assert x == loads(Operation, dumps(x))

    # Test the creation of Y
    y = Y(target_qubit="q")
    assert y.target_qubit == "q"
    assert y.__class__.__name__ == "Y"
    # Test the loads/dumps both using the right class and the abstract base class
    assert y == loads(Y, dumps(y))
    assert y == loads(Operation, dumps(y))

    # Test the creation of Z
    z = Z(target_qubit="q")
    assert z.target_qubit == "q"
    assert z.__class__.__name__ == "Z"
    # Test the loads/dumps both using the right class and the abstract base class
    assert z == loads(Z, dumps(z))
    assert z == loads(Operation, dumps(z))

    # Test the creation of T
    t = T(target_qubit="q")
    assert t.target_qubit == "q"
    assert t.__class__.__name__ == "T"
    # Test the loads/dumps both using the right class and the abstract base class
    assert t == loads(T, dumps(t))
    assert t == loads(Operation, dumps(t))
