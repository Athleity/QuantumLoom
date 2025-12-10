"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

Pytest configuration for Loom unit tests.

This file defines fixtures of useful objects for unit tests.

In particular, it defines:
- n_rsc_block_factory: A factory to construct n 3x3 rotated surface code blocks,
without using the loom_rotated_surface_code package.

"""

from typing import Callable
import pytest

from loom.eka import (
    Lattice,
    ChannelType,
    Channel,
    Circuit,
    SyndromeCircuit,
    Block,
    Stabilizer,
    PauliOperator,
    Eka,
)
from loom.eka.operations import ResetAllDataQubits, MeasureBlockSyndromes

# pylint: disable=redefined-outer-name, duplicate-code


@pytest.fixture(scope="session")
def rsc_syndrome_circuits() -> list[SyndromeCircuit]:
    """Syndrome circuits for the rotated surface code block on arbitrary channels"""
    channels = {
        "a": [
            Channel(type=ChannelType.QUANTUM, label="q_"),
        ],
        "q": [Channel(type=ChannelType.QUANTUM, label=f"d{i}") for i in range(4)],
        "c": [Channel(type=ChannelType.CLASSICAL, label="c0")],
    }
    xxxx_circuit = SyndromeCircuit(
        name="xxxx",
        pauli="XXXX",
        circuit=Circuit(
            name="xxxx",
            circuit=(
                (Circuit("Reset_0", channels=channels["a"]),),
                (Circuit("H", channels=channels["a"]),),
                (Circuit("CX", channels=[channels["a"][0], channels["q"][0]]),),
                (Circuit("CX", channels=[channels["a"][0], channels["q"][1]]),),
                (Circuit("CX", channels=[channels["a"][0], channels["q"][2]]),),
                (Circuit("CX", channels=[channels["a"][0], channels["q"][3]]),),
                (Circuit("H", channels=channels["a"]),),
                (
                    Circuit(
                        "Measurement", channels=[channels["a"][0], channels["c"][0]]
                    ),
                ),
            ),
            channels=channels["q"] + channels["a"] + channels["c"],
        ),
    )
    zzzz_circuit = SyndromeCircuit(
        name="zzzz",
        pauli="ZZZZ",
        circuit=Circuit(
            name="zzzz",
            circuit=(
                (
                    Circuit(
                        "Reset_0",
                        channels=channels["a"],
                    ),
                ),
                (Circuit("H", channels=channels["a"]),),
                (Circuit("CZ", channels=[channels["a"][0], channels["q"][0]]),),
                (Circuit("CZ", channels=[channels["a"][0], channels["q"][1]]),),
                (Circuit("CZ", channels=[channels["a"][0], channels["q"][2]]),),
                (Circuit("CZ", channels=[channels["a"][0], channels["q"][3]]),),
                (Circuit("H", channels=channels["a"]),),
                (
                    Circuit(
                        "Measurement",
                        channels=[channels["a"][0], channels["c"][0]],
                    ),
                ),
            ),
            channels=channels["q"] + channels["a"] + channels["c"],
        ),
    )
    left_xx_circuit = SyndromeCircuit(
        pauli="XX",
        name="left_xx",
        circuit=Circuit(
            name="left_xx",
            circuit=(
                (
                    Circuit(
                        "Reset_0",
                        channels=channels["a"],
                    ),
                ),
                (Circuit("H", channels=channels["a"]),),
                (Circuit("CX", channels=[channels["a"][0], channels["q"][0]]),),
                (Circuit("CX", channels=[channels["a"][0], channels["q"][1]]),),
                (),
                (),
                (Circuit("H", channels=channels["a"]),),
                (
                    Circuit(
                        "Measurement",
                        channels=[channels["a"][0], channels["c"][0]],
                    ),
                ),
            ),
            channels=channels["q"][:2] + channels["a"] + channels["c"],
        ),
    )
    right_xx_circuit = SyndromeCircuit(
        pauli="XX",
        name="right_xx",
        circuit=Circuit(
            name="right_xx",
            circuit=(
                (
                    Circuit(
                        "Reset_0",
                        channels=channels["a"],
                    ),
                ),
                (Circuit("H", channels=channels["a"]),),
                (),
                (),
                (Circuit("CX", channels=[channels["a"][0], channels["q"][0]]),),
                (Circuit("CX", channels=[channels["a"][0], channels["q"][1]]),),
                (Circuit("H", channels=channels["a"]),),
                (
                    Circuit(
                        "Measurement",
                        channels=[channels["a"][0], channels["c"][0]],
                    ),
                ),
            ),
            channels=channels["q"][:2] + channels["a"] + channels["c"],
        ),
    )
    top_zz_circuit = SyndromeCircuit(
        pauli="ZZ",
        name="top_zz",
        circuit=Circuit(
            name="top_zz",
            circuit=(
                (
                    Circuit(
                        "Reset_0",
                        channels=channels["a"],
                    ),
                ),
                (Circuit("H", channels=channels["a"]),),
                (),
                (),
                (Circuit("CZ", channels=[channels["a"][0], channels["q"][0]]),),
                (Circuit("CZ", channels=[channels["a"][0], channels["q"][1]]),),
                (Circuit("H", channels=channels["a"]),),
                (
                    Circuit(
                        "Measurement",
                        channels=[channels["a"][0], channels["c"][0]],
                    ),
                ),
            ),
            channels=channels["q"][:2] + channels["a"] + channels["c"],
        ),
    )
    bottom_zz_circuit = SyndromeCircuit(
        name="bottom_zz",
        pauli="ZZ",
        circuit=Circuit(
            name="bottom_zz",
            circuit=(
                (
                    Circuit(
                        "Reset_0",
                        channels=channels["a"],
                    ),
                ),
                (Circuit("H", channels=channels["a"]),),
                (Circuit("CZ", channels=[channels["a"][0], channels["q"][0]]),),
                (Circuit("CZ", channels=[channels["a"][0], channels["q"][1]]),),
                (),
                (),
                (Circuit("H", channels=channels["a"]),),
                (
                    Circuit(
                        "Measurement",
                        channels=[channels["a"][0], channels["c"][0]],
                    ),
                ),
            ),
            channels=channels["q"][:2] + channels["a"] + channels["c"],
        ),
    )
    return [
        xxxx_circuit,
        zzzz_circuit,
        left_xx_circuit,
        right_xx_circuit,
        top_zz_circuit,
        bottom_zz_circuit,
    ]


@pytest.fixture(scope="session")
def n_rsc_block_factory(rsc_syndrome_circuits) -> Callable[int, list[Block]]:
    """Factory to create several simple rsc blocks next to each others."""

    def _create(n: int) -> Block:
        """Create n rsc blocks next to each others in x direction."""
        rsc_blocks = []
        for i in range(n):
            rsc_stabilizers = (
                Stabilizer(
                    "ZZZZ",
                    (
                        (1 + 4 * i, 0, 0),
                        (0 + 4 * i, 0, 0),
                        (1 + 4 * i, 1, 0),
                        (0 + 4 * i, 1, 0),
                    ),
                    ancilla_qubits=((1 + 4 * i, 1, 1),),
                ),
                Stabilizer(
                    "ZZZZ",
                    (
                        (2 + 4 * i, 1, 0),
                        (1 + 4 * i, 1, 0),
                        (2 + 4 * i, 2, 0),
                        (1 + 4 * i, 2, 0),
                    ),
                    ancilla_qubits=((2 + 4 * i, 2, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    (
                        (1 + 4 * i, 1, 0),
                        (1 + 4 * i, 2, 0),
                        (0 + 4 * i, 1, 0),
                        (0 + 4 * i, 2, 0),
                    ),
                    ancilla_qubits=((1 + 4 * i, 2, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    (
                        (2 + 4 * i, 0, 0),
                        (2 + 4 * i, 1, 0),
                        (1 + 4 * i, 0, 0),
                        (1 + 4 * i, 1, 0),
                    ),
                    ancilla_qubits=((2 + 4 * i, 1, 1),),
                ),
                Stabilizer(
                    "XX",
                    ((0 + 4 * i, 0, 0), (0 + 4 * i, 1, 0)),
                    ancilla_qubits=((0 + 4 * i, 1, 1),),
                ),
                Stabilizer(
                    "XX",
                    ((2 + 4 * i, 1, 0), (2 + 4 * i, 2, 0)),
                    ancilla_qubits=((3 + 4 * i, 2, 1),),
                ),
                Stabilizer(
                    "ZZ",
                    ((2 + 4 * i, 0, 0), (1 + 4 * i, 0, 0)),
                    ancilla_qubits=((2 + 4 * i, 0, 1),),
                ),
                Stabilizer(
                    "ZZ",
                    ((1 + 4 * i, 2, 0), (0 + 4 * i, 2, 0)),
                    ancilla_qubits=((1 + 4 * i, 3, 1),),
                ),
            )
            rsc_blocks.append(
                Block(
                    unique_label=f"rsc_block_{i}",
                    stabilizers=rsc_stabilizers,
                    logical_x_operators=(
                        PauliOperator(
                            "XXX",
                            ((0 + 4 * i, 0, 0), (1 + 4 * i, 0, 0), (2 + 4 * i, 0, 0)),
                        ),
                    ),
                    logical_z_operators=(
                        PauliOperator(
                            "ZZZ",
                            ((0 + 4 * i, 0, 0), (0 + 4 * i, 1, 0), (0 + 4 * i, 2, 0)),
                        ),
                    ),
                    syndrome_circuits=rsc_syndrome_circuits,
                    stabilizer_to_circuit={
                        rsc_stabilizers[0].uuid: rsc_syndrome_circuits[1].uuid,
                        rsc_stabilizers[1].uuid: rsc_syndrome_circuits[1].uuid,
                        rsc_stabilizers[2].uuid: rsc_syndrome_circuits[0].uuid,
                        rsc_stabilizers[3].uuid: rsc_syndrome_circuits[0].uuid,
                        rsc_stabilizers[4].uuid: rsc_syndrome_circuits[2].uuid,
                        rsc_stabilizers[5].uuid: rsc_syndrome_circuits[3].uuid,
                        rsc_stabilizers[6].uuid: rsc_syndrome_circuits[4].uuid,
                        rsc_stabilizers[7].uuid: rsc_syndrome_circuits[5].uuid,
                    },
                )
            )
        return rsc_blocks

    return _create


@pytest.fixture(scope="session")
def empty_eka() -> Eka:
    """A empty eka fixture."""
    return Eka(Lattice.square_2d((10, 20)))


@pytest.fixture
def typical_eka(n_rsc_block_factory) -> Eka:
    """A typical eka fixture with 3 rsc blocks."""
    blocks = n_rsc_block_factory(3)
    operations = [
        [ResetAllDataQubits(input_block_name=block.unique_label) for block in blocks],
        [
            MeasureBlockSyndromes(input_block_name=block.unique_label, n_cycles=3)
            for block in blocks
        ],
    ]
    return Eka(
        lattice=Lattice.square_2d((20, 10)),
        blocks=blocks,
        operations=operations,
    )
