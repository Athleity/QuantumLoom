"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

Pytest configuration for eka unit tests.

This file defines fixtures and configurations specific to the eka unit tests. Providing
a centralized location for test setup helps maintain consistency and reduces redundancy
across multiple test files.

Note that we expect these fixtures to not require any dependency to other loom modules.

The fixtures objects are NOT intended to be used within their own unit test, e.g. do not
use the block fixture to test the Block class itself. Rather, they should be used to
test higher-level objects that depend on them, e.g. use the Block fixture to test the
Eka class.
"""

import pytest
import networkx as nx
import numpy as np

from loom.eka import (
    Lattice,
    PauliOperator,
    Block,
    Stabilizer,
    TannerGraph,
    ParityCheckMatrix,
)

# pylint: disable=duplicate-code, redefined-outer-name


@pytest.fixture
def steane_code_tanner_graph() -> nx.Graph:
    """Return the tanner graph representation of the Steane code."""

    # Steane code variables
    data_supports_ham = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]

    datas_ham = list(range(7))
    checks_ham = list(range(3))

    # Add extra index to datas and checks to match stabilizer qubit format
    x_nodes_steane = [((i, 1), {"label": "X"}) for i in checks_ham]
    z_nodes_steane = [((i + 3, 1), {"label": "Z"}) for i in checks_ham]
    data_nodes_steane = [((i, 0), {"label": "data"}) for i in datas_ham]

    x_edges_steane = [
        ((c, 1), (d, 0))
        for c, supp in zip(checks_ham, data_supports_ham, strict=True)
        for d in supp
    ]
    z_edges_steane = [((c[0] + 3, 1), d) for c, d in x_edges_steane]

    g_steane = nx.Graph()
    g_steane.add_nodes_from(x_nodes_steane)
    g_steane.add_nodes_from(z_nodes_steane)
    g_steane.add_nodes_from(data_nodes_steane)
    g_steane.add_edges_from(x_edges_steane + z_edges_steane)
    return TannerGraph(g_steane)


@pytest.fixture
def steane_code_parity_matrix() -> ParityCheckMatrix:
    """Parity check matrix fixture for the Steane code."""
    h_hamming = np.array(
        [[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1]],
        dtype=int,
    )

    h_steane = np.vstack(
        (
            np.hstack((h_hamming, np.zeros(h_hamming.shape, dtype=int))),
            np.hstack((np.zeros(h_hamming.shape, dtype=int), h_hamming)),
        )
    )
    return ParityCheckMatrix(h_steane)


@pytest.fixture
def shor_parity_matrix() -> ParityCheckMatrix:
    """Return the parity check matrix of the Shor code."""
    hx_shor = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 1]])
    hz_shor = np.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1],
        ]
    )
    h_shor = np.vstack(
        (
            np.hstack((hx_shor, np.zeros(hx_shor.shape, dtype=int))),
            np.hstack((np.zeros(hz_shor.shape, dtype=int), hz_shor)),
        )
    )
    return ParityCheckMatrix(h_shor)


@pytest.fixture
def shor_code_tanner_graph() -> TannerGraph:
    """Return the tanner graph representation of the Shor code."""
    g_shor = nx.Graph()
    # Add extra index to datas and checks to match stabilizer qubit format
    data_nodes_shor = [((i, 0), {"label": "data"}) for i in range(9)]
    x_nodes_shor = [((i, 1), {"label": "X"}) for i in range(2)]
    z_nodes_shor = [((i + 2, 1), {"label": "Z"}) for i in range(6)]

    x_edges_shor = [((i, 1), (j + 3 * i, 0)) for i in range(2) for j in range(6)]
    z_edges_shor = [
        ((i + 2, 1), (j + 3 * (i // 2) + (i % 2), 0))
        for i in range(6)
        for j in range(2)
    ]

    g_shor.add_nodes_from(x_nodes_shor)
    g_shor.add_nodes_from(z_nodes_shor)
    g_shor.add_nodes_from(data_nodes_shor)
    g_shor.add_edges_from(x_edges_shor + z_edges_shor)
    return TannerGraph(g_shor)


@pytest.fixture
def typical_lattice():
    """A typical lattice fixture."""
    # I make it big to avoid limitation in simple cases. This isn't a problem as it
    # doesn't scale in memory.
    return Lattice.square_2d(lattice_size=(40, 40))


@pytest.fixture
def typical_block(n_rsc_block_factory) -> Block:
    """A typical block fixture. This is a rotated surface code block."""
    #  Rotated Surface Code
    return n_rsc_block_factory(1)[0].rename("q1")


@pytest.fixture
def simple_block() -> Block:
    """A simple block fixture."""
    stabilizers = [
        Stabilizer(
            pauli="XX",
            data_qubits=(
                (0, 0),
                (1, 0),
            ),
            ancilla_qubits=((3, 1),),
        ),
        Stabilizer(
            pauli="XX",
            data_qubits=(
                (1, 0),
                (2, 0),
            ),
            ancilla_qubits=((4, 1),),
        ),
    ]
    logical_x = PauliOperator(pauli="XXX", data_qubits=((0, 0), (1, 0), (2, 0)))
    logical_z = PauliOperator(pauli="ZZZ", data_qubits=((0, 0), (1, 0), (2, 0)))

    return Block(
        stabilizers=stabilizers,
        logical_x_operators=[logical_x],
        logical_z_operators=[logical_z],
        unique_label="q1",
    )


@pytest.fixture(scope="class")
def rsc_block(n_rsc_block_factory) -> Block:
    """Fixture for a simple rotated surface code block."""
    return n_rsc_block_factory(1)[0].rename("q1")
