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

import pytest

from loom.eka import Lattice, Eka
from loom.eka.operations import MeasureBlockSyndromes

from loom.eka.utilities import dumps, loads


class TestEka:
    """
    Tests the Eka class. The tests are not exhaustive but check the most important
    functionality.
    """

    def test_creation_of_eka_without_blocks(self, typical_lattice):
        """
        Tests the creation of an Eka object without adding blocks.
        """
        eka = Eka(typical_lattice)

        assert eka.lattice == typical_lattice

    def test_creation_of_eka_non_overlapping_blocks(
        self, n_rsc_block_factory, typical_lattice
    ):
        """
        Tests the creation of an Eka object with valid, non-overlapping blocks.
        """
        rsc_blocks = n_rsc_block_factory(2)
        eka = Eka(typical_lattice, blocks=rsc_blocks)

        assert eka.blocks == tuple(rsc_blocks)

    def test_loads_dumps(self, n_rsc_block_factory, typical_lattice):
        """
        Test that the loads and dumps functions work correctly.
        """
        rsc_blocks = n_rsc_block_factory(2)
        eka = Eka(typical_lattice, blocks=rsc_blocks)
        eka_json = dumps(eka)
        loaded_eka = loads(Eka, eka_json)

        assert loaded_eka == eka

    def test_creation_of_eka_infinite_size(self, n_rsc_block_factory):
        """
        Tests the creation of an Eka object with infinite lattice. This should
        mainly check that the validation of qubit indices also works for infinite
        lattices.
        """
        rsc_blocks = n_rsc_block_factory(2)

        eka = Eka(Lattice.square_2d(), blocks=rsc_blocks)

        assert eka.lattice.n_dimensions == 2
        assert eka.lattice.unit_cell_size == 2
        assert eka.lattice.size is None

    def test_creation_of_eka_overlapping_blocks(self, typical_block, typical_lattice):
        """
        Tests the creation of an Eka object with overlapping blocks. In this case
        there should be an error raised.
        """
        q1 = typical_block.rename("q1")
        q2 = typical_block.shift((0, 2)).rename("q2")

        with pytest.raises(ValueError) as excinfo:
            Eka(typical_lattice, blocks=[q1, q2])

        err_msg = (
            "Block 'q1' and block 'q2' share the data qubits"
            + " {(0, 2, 0), (2, 2, 0), (1, 2, 0)}"
        )
        assert err_msg in str(excinfo.value)

    def test_validation_blocks_unique_labels(self, typical_block, typical_lattice):
        """
        Tests that an error is raised in validation when block labels are not
        unique.
        """
        q1 = typical_block.rename("q1")
        q2 = typical_block.shift((0, 4)).rename("q1")

        with pytest.raises(ValueError) as excinfo:
            Eka(typical_lattice, blocks=[q1, q2])

        err_msg = "Not all blocks have unique labels."
        assert err_msg in str(excinfo.value)

    def test_validation_blocks_bad_indices(self, typical_block, typical_lattice):
        """
        Tests that an error is raised in validation when data qubit indices are
        invalid (negative or larger than the lattice size).
        """
        q1 = typical_block.rename("q1")

        # Check that an error is raised when a data qubit has a negative index
        q2 = typical_block.shift((-2, 0)).rename("q2")

        with pytest.raises(ValueError) as excinfo:
            Eka(typical_lattice, blocks=[q1, q2])

        err_msg = "Block 'q2' has negative data qubit indices."
        assert err_msg in str(excinfo.value)

        # Check that an error is raised when a data or and ancilla qubit has an index
        # which is too large for the respective lattice dimension.
        # Since the lattice is 10 x 20, the maximum index for the first dimension is 9.
        # A 3x3 code starting a (6,*) is valid while starting at (7,*) is not.
        q3 = typical_block.shift((6, 0)).rename("q3")
        Eka(typical_lattice, blocks=[q1, q3])

        q4 = q1.shift((37, 0)).rename("q4")
        with pytest.raises(ValueError) as excinfo:
            Eka(typical_lattice, blocks=[q1, q4])

        err_msg = (
            "Block 'q4' has ancilla qubit indices which are too large for the lattice."
        )
        assert err_msg in str(excinfo.value)

        # Check the 2nd dimension as well
        q5 = typical_block.shift((0, 16)).rename("q5")
        Eka(typical_lattice, blocks=[q1, q5])

        q6 = typical_block.shift((0, 37)).rename("q6")
        with pytest.raises(ValueError) as excinfo:
            Eka(typical_lattice, blocks=[q1, q6])

        err_msg = (
            "Block 'q6' has ancilla qubit indices which are too large for the lattice."
        )
        assert err_msg in str(excinfo.value)

    def test_creation_of_eka_with_operations(self, typical_block, typical_lattice):
        """
        Tests the creation of an Eka object with operations.
        """
        q1 = typical_block.rename("q1")
        q2 = typical_block.shift((0, 4)).rename("q2")

        meas_q1 = MeasureBlockSyndromes(q1.unique_label)
        meas_q2 = MeasureBlockSyndromes(q2.unique_label)

        eka = Eka(
            typical_lattice,
            blocks=[q1, q2],
            operations=[meas_q1, meas_q2],
        )

        assert eka.blocks == (q1, q2)
        assert eka.operations == ((meas_q1,), (meas_q2,))

        # Test that tuple[tuple[Operation, ...], ...] is also accepted and yields
        # the same result
        eka_eq = Eka(
            typical_lattice,
            blocks=[q1, q2],
            operations=[(meas_q1,), (meas_q2,)],
        )
        assert eka == eka_eq
