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

from pydantic import ValidationError
import pytest

from loom.eka import Block, Stabilizer, PauliOperator
from loom.eka.utilities import loads, dumps, uuid_error

# pylint: disable=redefined-outer-name


@pytest.fixture(scope="class")
def rep_code_stabilizers() -> list[Stabilizer]:
    """Provides the stabilizers for a repetition code block."""
    return [
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


@pytest.fixture(scope="class")
def rep_code_logical_operators() -> tuple[PauliOperator, PauliOperator]:
    """Provides the logical operators for a repetition code block."""
    logical_x = PauliOperator(pauli="XXX", data_qubits=((0, 0), (1, 0), (2, 0)))
    logical_z = PauliOperator(pauli="ZZZ", data_qubits=((0, 0), (1, 0), (2, 0)))
    return logical_x, logical_z


@pytest.fixture(scope="class")
def steane_block() -> Block:
    """Return a Steane code block."""
    # Steane code variables
    data_supports_ham = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]

    return Block(
        stabilizers=[
            Stabilizer(
                pauli=p * 4,
                data_qubits=[(d, 0) for d in support],
                ancilla_qubits=[(i + 3 * j, 1)],
            )
            for i, support in enumerate(data_supports_ham)
            for j, p in enumerate("XZ")
        ],
        logical_x_operators=[
            PauliOperator(pauli="X" * 7, data_qubits=[(i, 0) for i in range(7)])
        ],
        logical_z_operators=[
            PauliOperator(pauli="Z" * 7, data_qubits=[(i, 0) for i in range(7)])
        ],
    )


@pytest.fixture(scope="class")
def shor_block() -> Block:
    """Return a Shor code block."""
    # Shor code variables
    x_indices = [[(i + 3 * j, 0) for i in range(6)] for j in range(2)]
    z_indices = [
        [(i + j + 3 * k, 0) for i in range(2)] for k in range(3) for j in range(2)
    ]
    paulis = ["X" * 6, "Z" * 2]

    return Block(
        stabilizers=[
            Stabilizer(
                pauli=paulis[i],
                data_qubits=data_qubits,
                ancilla_qubits=[(j + 2 * i, 1)],
            )
            for i, indices in enumerate([x_indices, z_indices])
            for j, data_qubits in enumerate(indices)
        ],
        logical_x_operators=[
            PauliOperator(pauli="X" * 9, data_qubits=[(i, 0) for i in range(9)])
        ],
        logical_z_operators=[
            PauliOperator(pauli="Z" * 9, data_qubits=[(i, 0) for i in range(9)])
        ],
    )


class TestBlock:  # pylint: disable=too-many-public-methods
    """
    Test for the Block class.
    """

    def test_creation_valid_block(
        self, rep_code_stabilizers, rep_code_logical_operators
    ):
        """
        Test the creation of a Block object.
        """
        stabilizers = rep_code_stabilizers
        logical_x, logical_z = rep_code_logical_operators

        block = Block(
            stabilizers=stabilizers,
            logical_x_operators=[logical_x],
            logical_z_operators=[logical_z],
            unique_label="q1",
        )

        assert block.unique_label == "q1"
        assert block.stabilizers == tuple(stabilizers)
        assert block.logical_x_operators == (logical_x,)
        assert block.logical_z_operators == (logical_z,)
        assert block.skip_validation is False

    # Test of the "before" validators
    # NOTE they are tested in the order they are executed with pydantic (bottom to top
    # in the code)
    def test_validators_raise(self, rsc_block, subtests):
        """Tests that the validators raise the appropriate errors."""
        test_case = [
            {
                "case_name": "empty_stabilizers",
                "unique_label": rsc_block.unique_label,
                "stabilizers": [],
                "logical_x": rsc_block.logical_x_operators,
                "logical_z": rsc_block.logical_z_operators,
                "err_msg": "List cannot be empty.",
            },
            {
                "case_name": "empty_logical_operators",
                "unique_label": rsc_block.unique_label,
                "stabilizers": rsc_block.stabilizers,
                "logical_x": [],
                "logical_z": [],
                "err_msg": "List cannot be empty.",
            },
            {
                "case_name": "non_distinct_stabilizers",
                "unique_label": rsc_block.unique_label,
                "stabilizers": rsc_block.stabilizers[:-1] + rsc_block.stabilizers[0:1],
                "logical_x": rsc_block.logical_x_operators,
                "logical_z": rsc_block.logical_z_operators,
                "err_msg": "Value error, Stabilizers must be distinct.",
            },
            {
                "case_name": "non_distinct_logical_x_operators",
                "unique_label": rsc_block.unique_label,
                "stabilizers": rsc_block.stabilizers
                + (Stabilizer(pauli="Z", data_qubits=((10, 10, 0),)),),
                "logical_x": rsc_block.logical_x_operators
                + rsc_block.logical_x_operators[0:1],
                "logical_z": rsc_block.logical_z_operators
                + (PauliOperator("Z", [(10, 10, 0)]),),
                "err_msg": "Logical X operators must be distinct.",
            },
            {
                "case_name": "non_distinct_logical_z_operators",
                "unique_label": rsc_block.unique_label,
                "stabilizers": rsc_block.stabilizers
                + (Stabilizer(pauli="Z", data_qubits=((10, 10, 0),)),),
                "logical_x": rsc_block.logical_x_operators
                + (PauliOperator("Z", [(10, 10, 0)]),),
                "logical_z": rsc_block.logical_z_operators
                + rsc_block.logical_z_operators[0:1],
                "err_msg": "Logical Z operators must be distinct.",
            },
            {
                "case_name": "unequal_logical_operators",
                "unique_label": rsc_block.unique_label,
                "stabilizers": rsc_block.stabilizers
                + (Stabilizer(pauli="Z", data_qubits=((10, 10, 0),)),),
                "logical_x": rsc_block.logical_x_operators,
                "logical_z": rsc_block.logical_z_operators
                + (
                    PauliOperator(
                        pauli="ZZZ", data_qubits=((1, 0, 0), (1, 1, 0), (1, 2, 0))
                    ),
                ),
                "err_msg": (
                    "Value error, The number of logical X operators must be equal to "
                    "the number of logical Z operators."
                ),
            },
            {
                "case_name": "qubit with wrong coordinate dimension",
                "unique_label": rsc_block.unique_label,
                "stabilizers": [
                    Stabilizer(
                        pauli="XX",
                        data_qubits=((0, 0), (1, 0)),
                        ancilla_qubits=((3, 0, 1),),  # Different dimensions
                    ),
                    Stabilizer(
                        pauli="XX",
                        data_qubits=((1, 0), (2, 0)),
                        ancilla_qubits=((4, 1),),
                    ),
                ],
                "logical_x": [
                    PauliOperator(pauli="XXX", data_qubits=((0, 0), (1, 0), (2, 0)))
                ],
                "logical_z": [
                    PauliOperator(pauli="XXX", data_qubits=((0, 0), (1, 0), (2, 0)))
                ],
                "err_msg": "All qubits coordinates must have the same dimension.",
            },
            {
                "case_name": "qubit of logical op not in stabilizers",
                "unique_label": rsc_block.unique_label,
                "stabilizers": rsc_block.stabilizers,
                "logical_x": (PauliOperator("X", [(10, 10, 0)]),),
                "logical_z": (PauliOperator("Z", [(10, 10, 0)]),),
                "err_msg": (
                    "Qubits {(10, 10, 0)} are not included in the stabilizers but"
                    " are used in the logical operators"
                ),
            },
            {
                "case_name": (
                    "number of logical qubits not compatible with qubits in"
                    " stabilizer"
                ),
                "unique_label": rsc_block.unique_label,
                "stabilizers": rsc_block.stabilizers
                + (
                    Stabilizer(
                        pauli="ZZ",
                        data_qubits=((10, 10, 0), (10, 11, 0)),
                    ),
                ),
                "logical_x": rsc_block.logical_x_operators,
                "logical_z": rsc_block.logical_z_operators,
                "err_msg": (
                    "The number of qubits and independent stabilizers in the"
                    " Block is not compatible with the number of logical qubits."
                ),
            },
            {
                "case_name": "non commuting stabilizers",
                "unique_label": rsc_block.unique_label,
                "stabilizers": rsc_block.stabilizers[:-1]
                + (Stabilizer(pauli="Z", data_qubits=((0, 0, 0),)),),
                "logical_x": rsc_block.logical_x_operators,
                "logical_z": rsc_block.logical_z_operators,
                "err_msg": "Stabilizers must commute with each other",
            },
            {
                "case_name": "non commuting logical x operators",
                "unique_label": rsc_block.unique_label,
                "stabilizers": rsc_block.stabilizers
                + (
                    Stabilizer(
                        pauli="ZZ",
                        data_qubits=((10, 10, 0), (10, 11, 0)),
                    ),
                ),
                "logical_x": rsc_block.logical_x_operators
                + (PauliOperator(pauli="Z", data_qubits=((0, 0, 0),)),),
                "logical_z": rsc_block.logical_z_operators
                + (PauliOperator(pauli="Z", data_qubits=((10, 10, 0),)),),
                "err_msg": "Logical X operators must commute with each other",
            },
            {
                "case_name": "non commuting stabilizer x logical X operators",
                "unique_label": rsc_block.unique_label,
                "stabilizers": rsc_block.stabilizers
                + (
                    Stabilizer(
                        pauli="ZZ",
                        data_qubits=((10, 10, 0), (10, 11, 0)),
                    ),
                ),
                "logical_x": rsc_block.logical_x_operators
                + (PauliOperator(pauli="X", data_qubits=((10, 10, 0),)),),
                "logical_z": rsc_block.logical_z_operators
                + (PauliOperator(pauli="Z", data_qubits=((10, 10, 0),)),),
                "err_msg": "Stabilizers must commute with logical X operators",
            },
            {
                "case_name": "non commuting stabilizer x logical Z operators",
                "unique_label": rsc_block.unique_label,
                "stabilizers": rsc_block.stabilizers
                + (
                    Stabilizer(
                        pauli="XX",
                        data_qubits=((10, 10, 0), (10, 11, 0)),
                    ),
                ),
                "logical_x": rsc_block.logical_x_operators
                + (PauliOperator(pauli="X", data_qubits=((10, 10, 0),)),),
                "logical_z": rsc_block.logical_z_operators
                + (PauliOperator(pauli="Z", data_qubits=((10, 10, 0),)),),
                "err_msg": "Stabilizers must commute with logical Z operators",
            },
            {
                "case_name": "Logical operators Z, X on same index don't commute",
                "unique_label": rsc_block.unique_label,
                "stabilizers": rsc_block.stabilizers
                + (
                    Stabilizer(
                        pauli="XX",
                        data_qubits=((10, 10, 0), (10, 11, 0)),
                    ),
                ),
                "logical_x": rsc_block.logical_x_operators
                + (PauliOperator(pauli="X", data_qubits=((10, 10, 0),)),),
                "logical_z": rsc_block.logical_z_operators
                + (PauliOperator(pauli="X", data_qubits=((10, 11, 0),)),),
                "err_msg": (
                    "Logical X and Z operators at the same index must"
                    " anticommute with each other"
                ),
            },
            {
                "case_name": "Logical operators Z, X on same index don't anticommute 2",
                "unique_label": rsc_block.unique_label,
                "stabilizers": rsc_block.stabilizers
                + (
                    Stabilizer(
                        pauli="XXX",
                        data_qubits=((0, 10, 0), (1, 10, 0), (2, 10, 0)),
                    ),
                ),
                "logical_x": rsc_block.logical_x_operators
                + (
                    PauliOperator(pauli="X", data_qubits=((0, 10, 0),)),
                    PauliOperator(pauli="XX", data_qubits=((1, 10, 0), (2, 10, 0))),
                ),
                "logical_z": rsc_block.logical_z_operators
                + (
                    PauliOperator(pauli="ZZ", data_qubits=((0, 10, 0), (1, 10, 0))),
                    PauliOperator(pauli="ZZ", data_qubits=((1, 10, 0), (2, 10, 0))),
                ),
                "err_msg": (
                    "Logical X and Z operators at different indices must commute"
                    " with each other"
                ),
            },
        ]

        for case in test_case:
            with subtests.test(message=case["case_name"]):
                with pytest.raises(ValueError) as cm:
                    _ = Block(
                        unique_label=case["unique_label"],
                        stabilizers=case["stabilizers"],
                        logical_x_operators=case["logical_x"],
                        logical_z_operators=case["logical_z"],
                    )
                assert case["err_msg"] in str(cm.value)

    def test_block_reducible_stabilizers(self, rsc_block):
        """
        Test that no error is raised when the stabilizers are reducible.
        """
        # Test that the stabilizers are reducible
        # Replace the usual stabilizers by reducible ones
        additional_stab = Stabilizer(
            pauli="ZZXX",
            data_qubits=((0, 2, 0), (1, 2, 0), (2, 1, 0), (2, 2, 0)),
        )
        b = Block(
            unique_label=rsc_block.unique_label,
            stabilizers=rsc_block.stabilizers + (additional_stab,),
            logical_x_operators=rsc_block.logical_x_operators,
            logical_z_operators=rsc_block.logical_z_operators,
        )

        assert isinstance(b, Block)

    def test_loads_dumps(self, rsc_block):
        """
        Test that the loads and dumps functions work correctly.
        """

        block_json = dumps(rsc_block)
        loaded_block = loads(Block, block_json)

        assert loaded_block == rsc_block

    def test_qubit_properties(self, rep_code_stabilizers, rep_code_logical_operators):
        """
        Test that the data_qubits, ancilla_qubits, and qubits properties work correctly.
        """
        data_qubits = ((0, 0), (1, 0), (2, 0))
        ancilla_qubits = ((3, 1), (4, 1))

        stabilizers = rep_code_stabilizers
        logical_x, logical_z = rep_code_logical_operators
        block = Block(
            stabilizers=stabilizers,
            logical_x_operators=[logical_x],
            logical_z_operators=[logical_z],
            unique_label="q1",
        )
        # Comparing sets because the order of the qubits is not guaranteed
        assert set(block.data_qubits) == set(data_qubits)
        assert set(block.ancilla_qubits) == set(ancilla_qubits)
        assert set(block.qubits) == set(data_qubits + ancilla_qubits)

    def test_shift_function(self, rsc_block):
        """
        Test whether the shift() function correctly shifts a block.
        """

        dx, dy = (3, 5)
        shifted = rsc_block.shift((dx, dy))

        for (ox, oy, _), (nx, ny, _) in zip(
            rsc_block.data_qubits, shifted.data_qubits, strict=True
        ):
            assert nx == ox + dx
            assert ny == oy + dy

    def test_shift_with_rename(self, rsc_block):
        """
        Test whether the shift() function also correctly renames a block.
        """
        block_shifted_same_label = rsc_block.shift((3, 5))
        block_shifted_new_label = rsc_block.shift((3, 5), new_label="q3")
        # Check for unique_label values
        assert (
            block_shifted_same_label.unique_label
            != block_shifted_new_label.unique_label
        )

        assert block_shifted_same_label.unique_label == "q1"
        assert block_shifted_new_label.unique_label == "q3"
        # Check that the class type is not changed.
        assert type(block_shifted_same_label) == type(block_shifted_new_label)

        # Check that the syndrome circuits are still the same
        assert block_shifted_same_label.syndrome_circuits == rsc_block.syndrome_circuits
        # Note that the stabilizer to syndrome circuit mapping is different because new
        # Stabilizer objects were created and thus the uuids changed

    def test_shift_function_invalid_input(self, rsc_block):
        """
        Test whether the input validation of shift() works.
        """
        # Position has the wrong dimension
        with pytest.raises(ValueError):
            _ = rsc_block.shift((3, 5, 7, 0))

    def test_rename(self, rsc_block):
        """
        Test the rename() function of a Block.
        """
        block_renamed = rsc_block.rename("new_qb")
        # Check that the name changed
        assert block_renamed.unique_label == "new_qb"
        # Check that all other fields are the same
        assert block_renamed.stabilizers == rsc_block.stabilizers
        assert block_renamed.logical_x_operators == rsc_block.logical_x_operators
        assert block_renamed.logical_z_operators == rsc_block.logical_z_operators
        assert block_renamed.syndrome_circuits == rsc_block.syndrome_circuits
        assert block_renamed.stabilizer_to_circuit == rsc_block.stabilizer_to_circuit
        # Check that the class type is not changed.
        assert type(rsc_block) == type(block_renamed)

    def test_block_creation_without_unique_label(
        self, rep_code_stabilizers, rep_code_logical_operators
    ):
        """
        Test whether a `unique_label` is generated automatically if not provided.
        """
        # Test uuid creation for manual creation of the block

        block = Block(
            stabilizers=rep_code_stabilizers,
            logical_x_operators=[rep_code_logical_operators[0]],
            logical_z_operators=[rep_code_logical_operators[1]],
        )
        uuid_error(block.unique_label)

    def test_pauli_charges(self, rsc_block):
        """
        Tests whether the Pauli charges for a distance-3 rotated surface code,
        created manually, are correctly calculated.
        """
        pauli_charges_expected = {
            (0, 1, 0): "Z",
            (1, 2, 0): "X",
            (2, 1, 0): "Z",
            (0, 0, 0): "Y",
            (1, 1, 0): "_",
            (2, 0, 0): "Y",
            (0, 2, 0): "Y",
            (2, 2, 0): "Y",
            (1, 0, 0): "X",
        }
        assert rsc_block.pauli_charges == pauli_charges_expected

    def test_validation_stab_to_synd_circ_map(
        self, rep_code_stabilizers, rep_code_logical_operators
    ):
        """
        Test whether the validation works that all uuids in the stabilizer to circuit
        map must exist.
        """

        with pytest.raises(ValueError) as cm:
            _ = Block(
                stabilizers=rep_code_stabilizers,
                logical_x_operators=[rep_code_logical_operators[0]],
                logical_z_operators=[rep_code_logical_operators[1]],
                stabilizer_to_circuit={"wrong_uuid": "test"},
            )
        assert (
            "Stabilizer with uuid wrong_uuid is not present in the stabilizers."
            in str(cm.value)
        )

        with pytest.raises(ValueError) as cm:
            _ = Block(
                stabilizers=rep_code_stabilizers,
                logical_x_operators=[rep_code_logical_operators[0]],
                logical_z_operators=[rep_code_logical_operators[1]],
                stabilizer_to_circuit={rep_code_stabilizers[0].uuid: "wrong_uuid"},
            )
        assert (
            "Syndrome circuit with uuid wrong_uuid is not present in the syndrome"
            " circuits"
        ) in str(cm.value)

    def test_bypass_validation_invalid_block(
        self, rep_code_stabilizers, rep_code_logical_operators
    ):
        """
        Test whether the skip_validation flag works correctly.
        """

        # Test that the Block is created without any issues, even though it should be
        # invalid (use same logical operator for X and Z)
        _ = Block(
            stabilizers=rep_code_stabilizers,
            logical_x_operators=[rep_code_logical_operators[0]],
            logical_z_operators=[rep_code_logical_operators[0]],
            skip_validation=True,
        )
        # Check that the validation fails, when enabled
        with pytest.raises(ValidationError):
            _ = Block(
                stabilizers=rep_code_stabilizers,
                logical_x_operators=[rep_code_logical_operators[0]],
                logical_z_operators=[rep_code_logical_operators[0]],
                skip_validation=False,
            )

    def test_bypass_validation_valid_block(
        self, rep_code_stabilizers, rep_code_logical_operators
    ):
        """
        Test whether that a block created with skip_validation flag is initialised
        in the same way as without.
        """

        block_without_validation = Block(
            unique_label="block",
            stabilizers=rep_code_stabilizers,
            logical_x_operators=[rep_code_logical_operators[0]],
            logical_z_operators=[rep_code_logical_operators[1]],
            skip_validation=True,
        )
        block_with_validation = Block(
            unique_label="block",
            stabilizers=rep_code_stabilizers,
            logical_x_operators=[rep_code_logical_operators[0]],
            logical_z_operators=[rep_code_logical_operators[1]],
        )
        assert block_without_validation == block_with_validation

    def test_from_blocks_constructor(self, rsc_block):
        """
        Test the from_blocks constructor of the Block class.
        """
        # block_shifted_no_overlap is shifted relative to
        # self.rotated_surface_code by (0,4), so it does not overlap with
        # self.rotated_surface_code
        block_shifted_no_overlap = rsc_block.shift((0, 4), new_label="q2")

        combined_block = Block.from_blocks([rsc_block, block_shifted_no_overlap])

        # Check that the combined block has the same qubits and data qubits as the two
        # individual blocks
        combined_qubits = set(rsc_block.qubits) | set(block_shifted_no_overlap.qubits)

        assert set(combined_block.qubits) == combined_qubits

        combined_data_qubits = set(rsc_block.data_qubits) | set(
            block_shifted_no_overlap.data_qubits
        )
        assert set(combined_block.data_qubits) == combined_data_qubits

        # Check that if the two blocks overlap, an exception is raised
        # block_shifted_overlap is shifted relative to rsc_block by
        # (1,2), so it overlaps with rsc_block
        block_shifted_overlap = rsc_block.shift((1, 2), new_label="q3")

        with pytest.raises(ValueError):
            Block.from_blocks([rsc_block, block_shifted_overlap])

        # And also if we use the same block twice
        with pytest.raises(ValueError):
            Block.from_blocks([rsc_block, rsc_block])

    def test_stabilizers_labels(self, rsc_block):
        """Test the stabilizers_labels property."""

        expected_labels = {}

        for stabilizer in rsc_block.stabilizers:
            expected_labels[stabilizer.uuid] = {
                "space_coordinates": stabilizer.ancilla_qubits[0]
            }

        assert rsc_block.stabilizers_labels == expected_labels

    def test_get_stabilizer_label(self, rsc_block):
        """Test the correct extraction of a single stabilizer label."""

        for stabilizer in rsc_block.stabilizers:
            expected_label = {"space_coordinates": stabilizer.ancilla_qubits[0]}
            label = rsc_block.get_stabilizer_label(stabilizer.uuid)
            assert label == expected_label

    def test_parity_check_matrix(self, steane_code_parity_matrix, steane_block):
        """Test the parity_check_matrix property."""
        assert steane_block.parity_check_matrix == steane_code_parity_matrix

    def test_tanner_graph(self, steane_code_tanner_graph, steane_block):
        """Test the tanner_graph property."""
        assert steane_block.tanner_graph == steane_code_tanner_graph
