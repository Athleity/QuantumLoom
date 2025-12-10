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

from copy import deepcopy

import pytest

from loom.eka import (
    Lattice,
    Eka,
    Block,
    Circuit,
    Channel,
    Stabilizer,
    PauliOperator,
)
from loom.eka.operations import MeasureBlockSyndromes, Operation
from loom.interpreter import InterpretationStep, interpret_eka, cleanup_final_step

# pylint: disable=redefined-outer-name, duplicate-code


@pytest.fixture(scope="module")
def rsc_block(n_rsc_block_factory) -> Block:
    """Fixture for a simple rotated surface code block."""
    return n_rsc_block_factory(1)[0].rename("q1")


@pytest.fixture(scope="function")
def rep_code_block() -> Block:
    """Fixture for a simple repetition code block."""
    return Block(
        stabilizers=(
            Stabilizer(
                pauli="ZZ",
                data_qubits=((0, 0, 0), (0, 1, 0)),
                ancilla_qubits=((0, 1, 1),),
            ),
            Stabilizer(
                pauli="ZZ",
                data_qubits=((0, 1, 0), (0, 2, 0)),
                ancilla_qubits=((0, 2, 1),),
            ),
        ),
        logical_x_operators=[
            PauliOperator(pauli="XXX", data_qubits=((0, 0, 0), (0, 1, 0), (0, 2, 0)))
        ],
        logical_z_operators=[PauliOperator(pauli="Z", data_qubits=((0, 0, 0),))],
        unique_label="q1",
    )


@pytest.fixture(scope="function")
def eka_on_2_rep_code_factory(rep_code_block) -> Eka:
    """Fixture that provides a factory to create Eka with 2 repetition code blocks. and
    given operations."""
    b1 = rep_code_block
    b2 = b1.shift(position=(4, 0), new_label="q2")

    lattice = Lattice.square_2d((10, 5))

    def _create(operations: list[list[Operation]]) -> Eka:
        return Eka(
            lattice,
            blocks=[b1, b2],
            operations=operations,
        )

    return _create


# pylint: disable=too-many-instance-attributes
class TestInterpreter:
    """
    Tests the interpreter API, interpret_eka, and abstracted functions like
    cleanup_final_step.
    """

    def test_run_interpreter_without_operations(self, empty_eka, rsc_block):
        """
        Tests where the interpreter runs without an error where the Eka does not
        include any operations yet.
        """

        # empty_eka
        final_step = interpret_eka(empty_eka)
        assert isinstance(final_step, InterpretationStep)

        # only blocks, no operations
        eka_with_blocks = Eka(Lattice.square_2d((5, 5)), blocks=[rsc_block])
        assert isinstance(interpret_eka(eka_with_blocks), InterpretationStep)

    def test_run_interpreter_parallel_operations(self, eka_on_2_rep_code_factory):
        """
        Tests the interpretation for multiple operations happening in parallel.
        """
        operations = [
            [MeasureBlockSyndromes("q1", 2), MeasureBlockSyndromes("q2", 2)],
        ]
        eka_w_ops = eka_on_2_rep_code_factory(operations)

        new_step = interpret_eka(eka_w_ops)
        circ_meas_block_1 = new_step.final_circuit.circuit[0][0]
        circ_meas_block_2 = new_step.final_circuit.circuit[0][1]
        # Check that the right circuits are generated
        assert circ_meas_block_1.name == "measure q1 syndromes 2 time(s)"
        assert circ_meas_block_2.name == "measure q2 syndromes 2 time(s)"
        # Check that both operations are indeed done in parallel
        assert len(new_step.final_circuit.circuit) == 12

    def test_run_interpreter_parallel_operations_different_lengths(
        self, eka_on_2_rep_code_factory
    ):
        """
        Tests the interpretation for multiple operations happening in parallel but
        they have different lengths.
        """
        operations = [
            [
                MeasureBlockSyndromes("q2", 2),
                MeasureBlockSyndromes("q1", 1),
            ],
        ]
        eka_w_ops = eka_on_2_rep_code_factory(operations)
        new_step = interpret_eka(eka_w_ops)
        circ_meas_block_2 = new_step.final_circuit.circuit[0][0]
        circ_meas_block_1 = new_step.final_circuit.circuit[0][1]
        # Check that the right circuits are generated
        assert circ_meas_block_2.name == "measure q2 syndromes 2 time(s)"
        assert circ_meas_block_1.name == "measure q1 syndromes 1 time(s)"
        # Check that both operations are indeed done in parallel
        assert len(new_step.final_circuit.circuit) == 12

    def test_change_final_step(self, eka_on_2_rep_code_factory):
        """
        Tests that an exception is raised if methods of InterpretationStep are called
        after the interpreter has finished and which would change the final step.
        """
        operations = []
        eka = eka_on_2_rep_code_factory(operations)
        final_step = interpret_eka(eka)

        err_msg = (
            "Cannot change properties of the final InterpretationStep after the "
            "interpretation is finished."
        )

        # Call those methods which mutate the InterpretationStep and check that an
        # exception is raised
        with pytest.raises(ValueError) as cm:
            final_step.get_channel_MUT("DUMMY_CHANNEL_NAME")
        assert err_msg in str(cm.value)

        with pytest.raises(ValueError) as cm:
            circ = Circuit(name="test", channels=[])
            final_step.append_circuit_MUT(circ)
        assert err_msg in str(cm.value)

        with pytest.raises(ValueError) as cm:
            final_step.get_new_cbit_MUT("c")
        assert err_msg in str(cm.value)

    def test_cleanup_final_step(self, eka_on_2_rep_code_factory):
        """
        Tests the clean up function.
        """
        # Check that channels that are not part of the final circuit are removed
        # from the channel_dict.
        operations = []
        eka = eka_on_2_rep_code_factory(operations)
        final_step = interpret_eka(eka)

        original_channel_dict = deepcopy(final_step.channel_dict)
        final_step.channel_dict["DUMMY_CHANNEL_NAME"] = Channel(
            label="DUMMY_CHANNEL_NAME"
        )
        # After the previous line, the channel_dict should have changed
        assert original_channel_dict != final_step.channel_dict
        # Check that the cleaned-up version corresponds to the original dict
        final_step = cleanup_final_step(final_step)
        assert original_channel_dict == final_step.channel_dict

        # Check that all the channels in channel_dict also appear in the circuit
        if final_step.final_circuit is not None:
            for ch in final_step.channel_dict.values():
                assert ch in final_step.final_circuit.channels

    def test_cleanup_final_step_with_circuit(self, eka_on_2_rep_code_factory):
        """
        Test that the cleanup function generates the right circuit from a given
        intermediate_circuit_sequence.
        """
        operations = []
        eka = eka_on_2_rep_code_factory(operations)
        final_step = interpret_eka(eka)
        channels = [Channel(label=f"{i}") for i in range(4)]
        bell_pair = Circuit(
            name="bell",
            circuit=[
                [Circuit("H", channels=[channels[0]])],
                [Circuit("CX", channels=channels[:2], duration=2)],
            ],
        )
        single_x = Circuit(
            name="single_x",
            circuit=[
                [Circuit("X", channels=[channel]) for channel in channels[2:]],
            ],
        )
        another_bell_pair = bell_pair.clone(channels[2:])
        final_step.intermediate_circuit_sequence = (
            (bell_pair, single_x),
            (another_bell_pair,),
        )

        final_step = cleanup_final_step(final_step)
        expected_circuit = Circuit(
            name="Final circuit",
            circuit=(
                (bell_pair, single_x),
                tuple(),
                tuple(),
                (another_bell_pair,),
                tuple(),
                tuple(),
            ),
        )

        assert final_step.final_circuit.circuit == expected_circuit.circuit

    def test_cleanup_final_step_sorted_channels(self, rsc_block):
        """Test that the channels after cleanup function are sorted."""
        # Channels get scrambled after MeasureBlockSyndromes operation

        eka = Eka(
            Lattice.square_2d((5, 5)),
            blocks=[rsc_block],
            operations=[MeasureBlockSyndromes(rsc_block.unique_label)],
        )
        final_step = interpret_eka(eka)

        # Sort channels in final circuit by channel labels
        sorted_channels = tuple(
            sorted(final_step.final_circuit.channels, key=lambda item: item.label)
        )

        # Ensure that sorted_channel_dict is equal to the channel_dict after cleanup
        assert final_step.final_circuit.channels == sorted_channels

    def test_typical_eka(self, typical_eka):
        """Test that the interpreter can run on a typical eka fixture."""
        final_step = interpret_eka(typical_eka)
        assert isinstance(final_step, InterpretationStep)

        # 3 blocks, measure 3 cycles each -> 3 * 3
        # There is 8 stabilizers per RSC block, so 3*3*8 = 72 syndromes from
        # measurements + 3*4 from resets
        assert len(final_step.syndromes) == 84

        # 3 * 17 quantum channels per block
        # + one classical channels per stab, per syndrome measurement
        # = 3 * 17 + 3* 8 * 3 = 123 channels
        assert len(final_step.final_circuit.channels) == 123
