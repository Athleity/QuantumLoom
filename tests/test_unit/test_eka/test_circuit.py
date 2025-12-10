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

# pylint: disable=too-many-lines, redefined-outer-name

import logging
import re

import pytest
from pydantic import ValidationError

from loom.eka import Circuit, Channel, ChannelType, IfElseCircuit
from loom.eka.utilities import dumps, loads


@pytest.fixture(scope="class")
def h_gate() -> Circuit:
    """
    simple hadamard gate on a channel
    """

    def _create(chan: Channel = Channel()) -> Circuit:
        return Circuit(name="hadamard", duration=1, channels=(chan,))

    return _create


@pytest.fixture(scope="class")
def cnot_gate() -> Circuit:
    """
    simple cnot gate on two channels
    """

    def _create(chan1: Channel = Channel(), chan2: Channel = Channel()) -> Circuit:
        return Circuit(name="cnot", duration=5, channels=(chan1, chan2))

    return _create


@pytest.fixture(scope="class")
def simple_circuit(h_gate, cnot_gate) -> Circuit:
    """
    simple circuit for testing
    """
    channels = tuple(Channel() for _ in range(2))
    return Circuit(
        "entangle",
        [h_gate(channels[0]), cnot_gate(channels[0], channels[1])],
    )


@pytest.fixture(scope="class")
def circ_with_if(h_gate, cnot_gate) -> Circuit:
    """
    circuit with an if-else construct for testing
    """
    c1 = Channel()
    c2 = Channel()
    if_circ = IfElseCircuit(if_circuit=cnot_gate(c1, c2), else_circuit=h_gate(c1))
    return Circuit(
        "circ_with_if1",
        [
            h_gate(c1),
            if_circ,
        ],
    )


@pytest.fixture(scope="class")
def circ_with_nested_if(h_gate, cnot_gate, circ_with_if) -> Circuit:
    """
    circuit with nested if-else constructs for testing
    """
    c1 = Channel()
    c2 = Channel()

    nested_if_circ1 = IfElseCircuit(
        if_circuit=cnot_gate(c1, c2), else_circuit=circ_with_if.clone([c1, c2])
    )
    return Circuit(
        "circ_with_nested_if",
        [h_gate(c1), nested_if_circ1],
    )


# pylint: disable=too-many-public-methods
class TestCircuit:
    """
    Test for the Circuit class.
    """

    def test_logging__eq__(self, h_gate, cnot_gate, caplog):
        """
        Test that the logging in __eq__ method of Circuit works as expected.

        Note: There is no way to test logging formatting, so we only check that the
        logging is called
        """
        # Set to ERROR to avoid cluttering the test output
        logging.getLogger().setLevel(logging.ERROR)

        with caplog.at_level(logging.DEBUG, logger="loom.eka.circuit"):
            assert h_gate() != cnot_gate()

        err_msg1 = "The two circuits have a different number of time slices."
        err_msg2 = f"{h_gate().duration} != {cnot_gate().duration}\n"
        assert caplog.messages == [err_msg1, err_msg2]
        assert caplog.records[0].levelname == "INFO"
        assert caplog.records[1].levelname == "DEBUG"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="loom.eka.circuit"):
            assert Circuit("1", [[], [cnot_gate()]]) != Circuit(
                "2", [[h_gate()], [cnot_gate()]]
            )

        err_msg1 = "The two circuits have a different number of gates in a time slice."
        err_msg2 = "0 != 1 for time slices () and (hadamard (base gate),)\n"
        assert caplog.messages == [err_msg1, err_msg2]
        assert caplog.records[0].levelname == "INFO"
        assert caplog.records[1].levelname == "DEBUG"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="loom.eka.circuit"):
            assert Circuit("1", [h_gate()]) != Circuit(
                "2", [Circuit("X", channels=Channel())]
            )

        err_msg1 = "The two circuits have different gates in a time slice."
        err_msg2 = (
            "For time steps 0: (hadamard (base gate),) and " + "0: (x (base gate),), \n"
            "    hadamard != x for gates hadamard (base gate) and x (base gate)\n"
        )
        assert caplog.messages == [err_msg1, err_msg2]
        assert caplog.records[0].levelname == "INFO"
        assert caplog.records[1].levelname == "DEBUG"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="loom.eka.circuit"):
            assert Circuit(
                "1", [Circuit("X", channels=[Channel(), Channel()])]
            ) != Circuit("2", [Circuit("X", channels=Channel())])

        err_msg1 = "The two circuits have different channels in a gate."
        err_msg2 = (
            "\n    [(<ChannelType.QUANTUM: 'quantum'>, 'data_qubit')]\n"
            "        !=\n"
            "    [(<ChannelType.QUANTUM: 'quantum'>, 'data_qubit'), "
            "(<ChannelType.QUANTUM: 'quantum'>, 'data_qubit')]\n"
        )
        assert caplog.messages == [err_msg1, err_msg2]
        assert caplog.records[0].levelname == "INFO"
        assert caplog.records[1].levelname == "DEBUG"

    def test_setup_basic(self):
        """
        Tests that the creation of a basic gate is done correctly:
        - test validation of the Circuit data class: non-empty channels and name
        - test correct inference in Circuit.create(): default and custom channels
        """
        with pytest.raises(Exception) as cm:
            Circuit("", [Channel()])
        assert "Names of Circuit objects need to have at least one letter." in str(
            cm.value
        )
        assert "Input should be a valid tuple" in str(cm.value)

        # Not assigning channels == no channels
        test_h = Circuit("hadamard")
        assert len(test_h.channels) == 0
        assert test_h.circuit == tuple()

        channel = Channel()
        testh_2 = Circuit("hadamard", channels=channel)
        assert testh_2.channels == (channel,)
        assert testh_2.circuit == tuple()

    def test_setup_composite_no_channels(self, h_gate, cnot_gate):
        """
        Tests that the channels are correctly inferred from the sub-circuit channels
        """
        channels = tuple(Channel() for _ in range(2))
        test_circuit = Circuit(
            "entangle", [h_gate(channels[0]), cnot_gate(channels[0], channels[1])]
        )
        assert len(test_circuit.channels) == 2

    def test_count_qubits(
        self, h_gate, cnot_gate, simple_circuit, circ_with_if, circ_with_nested_if
    ):
        """
        Tests that the number of qubits in a circuit is correctly counted.
        """
        assert h_gate(Channel()).nr_of_qubits_in_circuit() == 1
        assert cnot_gate(Channel(), Channel()).nr_of_qubits_in_circuit() == 2
        assert simple_circuit.nr_of_qubits_in_circuit() == 2
        assert circ_with_if.nr_of_qubits_in_circuit() == 2
        assert circ_with_nested_if.nr_of_qubits_in_circuit() == 2

    def test_base_gate_cloning1(self, h_gate, subtests):
        """
        Tests that the cloning of a base gate is done correctly:
        - the name is transferred,
        - the id is newly generated,
        - the inputs and outputs are given by the parameters in clone_circuit().
        """
        original_h = h_gate(Channel())

        with subtests.test("cloning simple gate with new channel"):
            new_in = Channel()
            new_hadamard = original_h.clone(new_in)
            assert new_hadamard.id != original_h.id
            assert new_hadamard.channels[0].id != original_h.channels[0].id
            assert new_hadamard.name == original_h.name
            assert new_hadamard.channels[0].id == new_in.id
            assert new_hadamard.duration == original_h.duration

        with subtests.test("cloning without providing channel"):

            new_hadamard = original_h.clone()
            assert new_hadamard.id != original_h.id
            assert new_hadamard.channels[0].id != original_h.channels[0].id
            assert new_hadamard.name == original_h.name
            assert new_hadamard.duration == original_h.duration

    def test_nested_circuit_cloning(self, simple_circuit):
        """
        Tests that the cloning of a nested gate is done correctly:
        - the name is transferred,
        - all circuit channel ids are newly generated
        - check that channel ids that appeared multiple times in the original circuit
            are also the same in the new circuit
        """
        clone = simple_circuit.clone()

        assert clone.name == simple_circuit.name
        assert clone.circuit[0][0].name == simple_circuit.circuit[0][0].name
        assert clone.circuit[1][0].name == simple_circuit.circuit[1][0].name
        assert clone.circuit[0][0].duration == simple_circuit.circuit[0][0].duration
        assert clone.circuit[1][0].duration == simple_circuit.circuit[1][0].duration

        for old, new in zip(simple_circuit.channels, clone.channels, strict=True):
            assert old.id != new.id

        for i, time_slice in enumerate(simple_circuit.circuit):
            for j, gate in enumerate(time_slice):
                assert gate.channels[0] in simple_circuit.channels
                assert clone.circuit[i][j].channels[0] in clone.channels

        assert simple_circuit.duration == clone.duration

    def test_clone_with_ifelse(self, circ_with_if, circ_with_nested_if):
        """
        Tests that the cloning of a Circuit that contains IfElseCircuits is done
        correctly.
        """
        clone_circ_with_if = circ_with_if.clone()

        assert clone_circ_with_if == circ_with_if
        assert clone_circ_with_if.name == circ_with_if.name
        assert clone_circ_with_if.id != circ_with_if.id
        assert clone_circ_with_if.channels != circ_with_if.channels

        clone_circ_with_nested_if = circ_with_nested_if.clone()

        assert clone_circ_with_nested_if == circ_with_nested_if
        assert clone_circ_with_nested_if.name == circ_with_nested_if.name
        assert clone_circ_with_nested_if.id != circ_with_nested_if.id
        assert clone_circ_with_nested_if.channels != circ_with_nested_if.channels

    def test_load_dump(self, simple_circuit):
        """
        Test that the load and dump functions work correctly.
        """
        test_circuit_j = dumps(simple_circuit)
        loaded_test_circuit = loads(Circuit, test_circuit_j)

        assert loaded_test_circuit == simple_circuit

    def test_setup_composite_correct_channels(self, h_gate, cnot_gate):
        """
        Tests that consistency is confirmed for the correct channels and given order
        of channels is maintained.
        """
        channels = tuple(Channel() for _ in range(2))
        hadamard = h_gate(channels[1])
        cnot = cnot_gate(channels[::-1][0], channels[::-1][1])

        test_circuit = Circuit("entangle", [hadamard, cnot], channels)
        # Check that the channels are consistent with the given channels
        assert set(test_circuit.channels) == set(channels)
        # Check that the sub-circuits still act on the right channels
        assert test_circuit.circuit[0][0].channels == (channels[1],)
        assert test_circuit.circuit[1][0].channels == channels[::-1]

    def test_setup_composite_order_channels(self):
        """
        Tests that the correct channels are inferred from the sub-circuit channels in
        the right order and that the idle channels are added.
        """
        channels = tuple(Channel() for _ in range(2))
        hadamard = Circuit("hadamard", channels=channels[1])
        cnot = Circuit("cnot", channels=channels[::-1])
        circuit_init = Circuit("entangle", [hadamard, cnot])
        meas = Circuit(
            "Measurement",
            channels=[
                Channel(),
                Channel(ChannelType.QUANTUM),
                Channel(ChannelType.CLASSICAL),
            ],
        )
        ancilla = Channel(ChannelType.QUANTUM)
        cchannels = tuple(Channel(ChannelType.CLASSICAL) for _ in range(2))
        circuit = Circuit(
            "complex_test",
            [
                circuit_init,
                meas.clone([circuit_init.channels[0], ancilla, cchannels[0]]),
                meas.clone([circuit_init.channels[1], ancilla, cchannels[1]]),
            ],
        )
        # Check that the channels are consistent with the given channels
        assert set(circuit.channels) == set(
            circuit_init.channels + (ancilla,) + cchannels
        )
        # Check that the sub-circuits still act on the right channels
        assert circuit.circuit[0][0].channels == circuit_init.channels
        assert circuit.circuit[1] == ()  # Empty timestep
        assert circuit.circuit[2][0].channels == (
            circuit_init.channels[0],
            ancilla,
            cchannels[0],
        )
        assert circuit.circuit[3][0].channels == (
            circuit_init.channels[1],
            ancilla,
            cchannels[1],
        )

        assert isinstance(circuit.channels, tuple)

    def test_setup_composite_empty_timesteps(self, cnot_gate, h_gate):
        """
        Tests that empty time steps are resolved correctly and that duration is
        correctly inferred.
        """
        channels = tuple(Channel() for _ in range(3))
        # The circuit is created with extra empty time steps
        circuit_w_wait1 = Circuit(
            "circuit_w_wait",
            [
                [cnot_gate(channels[0], channels[1])],
                [h_gate(channels[2])],
                [],
                [],
                [h_gate(channels[2])],
                [],
            ],
        )
        assert circuit_w_wait1.duration == 6
        # The circuit is created with empty time steps but also with a duration longer
        # than the existing time_steps (they are not all included)
        circuit_w_wait2 = Circuit(
            "circuit_w_wait",
            [
                [h_gate(channels[2])],
                [cnot_gate(channels[0], channels[1])],
                [h_gate(channels[2])],
                [],
            ],
        )
        assert circuit_w_wait2.duration == 6

    def test_duration(
        self, h_gate, cnot_gate, simple_circuit, circ_with_if, circ_with_nested_if
    ):
        """
        Tests that the duration of a circuit is correctly calculated.
        """
        assert h_gate(Channel()).duration == 1
        assert cnot_gate(Channel(), Channel()).duration == 5
        assert simple_circuit.duration == 6
        assert circ_with_if.duration == 6
        assert circ_with_nested_if.duration == 7

        channels = tuple(Channel() for _ in range(4))

        test_circuit = Circuit(
            "test",
            [
                [
                    h_gate(channels[1]),
                    cnot_gate(channels[0], channels[2]),
                ],
                [],
                [],
                [cnot_gate(channels[1], channels[3])],
            ],
        )
        assert test_circuit.duration == 8
        with pytest.raises(ValidationError) as cm:
            _ = Circuit(
                "fail",
                [
                    [cnot_gate(channels[0], channels[1])],
                    [cnot_gate(channels[2], channels[3])],
                ],
                duration=8,
            )

        assert (
            "Error while setting up composite circuit: Provided duration (8) does not"
            " match the duration of the sub-circuits (6)." in str(cm.value)
        )

    def test_op_timing(self, h_gate, cnot_gate):
        """
        Tests validation of timing of operations. A qubit can only be acted on by one
        gate each tick.
        """
        channels = tuple(Channel() for _ in range(4))
        with pytest.raises(ValidationError) as cm:
            _ = Circuit(
                "test",
                [
                    [h_gate(channels[1]), cnot_gate(channels[0], channels[2])],
                    [],
                    [],
                    [cnot_gate(channels[1], channels[3])],
                    [h_gate(channels[1])],
                ],
            )
        # Replace the random 6-digit uuid hex string to be able to use the assert
        # statement
        cleaned_message = re.sub(
            r"data_qubit\([a-z0-9]{6}\.\.\)", "data_qubit(uuid)", str(cm.value)
        )
        assert (
            "Error while setting up composite circuit: Channel data_qubit(uuid) is"
            " subject to more than one operation at tick 4." in cleaned_message
        )

    def test_as_gate(self):
        """
        Tests gate construction with the as_gate() convenience function
        """
        test_gate = Circuit.as_gate(
            "test_gate", nr_qchannels=3, nr_cchannels=2, duration=5
        )

        assert test_gate.name == "test_gate"
        assert len(test_gate.channels) == 5
        assert test_gate.duration == 5
        assert test_gate.channels[0].type == ChannelType.QUANTUM
        assert test_gate.channels[0].label == "data_qubit"
        assert test_gate.channels[1].type == ChannelType.QUANTUM
        assert test_gate.channels[2].type == ChannelType.QUANTUM
        assert test_gate.channels[3].type == ChannelType.CLASSICAL
        assert test_gate.channels[4].type == ChannelType.CLASSICAL

        test_gate = Circuit.as_gate("default_gate", 1)

        assert test_gate.name == "default_gate"
        assert len(test_gate.channels) == 1
        assert test_gate.duration == 1
        assert test_gate.channels[0].type == ChannelType.QUANTUM

    def test_from_circuit(self, h_gate, cnot_gate, circ_with_if, circ_with_nested_if):
        """
        Tests circuit construction via the from_circuit() convenience function
        """
        test_circuit = Circuit.from_circuits(
            "test_circuit",
            [(h_gate(), [0]), (cnot_gate(), [0, 1])],
        )

        assert test_circuit.name == "test_circuit"
        assert len(test_circuit.channels) == 2
        assert test_circuit.duration == 6
        assert test_circuit.channels[0].type == ChannelType.QUANTUM
        assert test_circuit.channels[1].type == ChannelType.QUANTUM
        assert (
            test_circuit.circuit[0][0].channels[0]
            == test_circuit.circuit[1][0].channels[0]
        )
        assert (
            test_circuit.circuit[0][0].channels[0]
            != test_circuit.circuit[1][0].channels[1]
        )
        with pytest.raises(ValueError) as cm:
            test_circuit = Circuit.from_circuits("test_circuit")
        assert (
            "Error while creating circuit via from_circuit(): "
            + "The circuit must be a list of circuits. If the intention is to copy "
            + "a circuit to deal with Channel objects directly, use the clone() "
            + "method instead."
        ) in str(cm.value)

        # check Channel type consistency
        zmeasure = Circuit.as_gate(name="measure_z", nr_qchannels=1, nr_cchannels=1)
        _ = Circuit.from_circuits(
            "measure_xx",
            [
                (h_gate(), [0]),
                (cnot_gate(), [0, 2]),
                (cnot_gate(), [0, 3]),
                (h_gate(), [0]),
                (zmeasure, [0, 1]),
            ],
        )
        with pytest.raises(ValueError) as cm:
            _ = Circuit.from_circuits(
                "measure_xx",
                [
                    (h_gate(), [0]),
                    (cnot_gate(), [0, 2]),
                    (cnot_gate(), [0, 3]),
                    (h_gate(), [0]),
                    (zmeasure, [0, 2]),
                ],
            )
        assert (
            "Provided channel indices are not consistent with respect to their types. "
            f"Offending channel 2 has type {ChannelType.CLASSICAL} but has previously "
            f"been used with a channel of type {ChannelType.QUANTUM}."
        ) in str(cm.value)

        test_with_if = Circuit.from_circuits(
            "test_with_if",
            [(circ_with_if, range(3)), (circ_with_nested_if, range(4))],
        )
        assert test_with_if.name == "test_with_if"
        assert len(test_with_if.channels) == 4
        assert test_with_if.duration == 13
        assert test_with_if.circuit[0][0] == circ_with_if
        assert test_with_if.circuit[6][0] == circ_with_nested_if

    def test_flatten_circuit_is_flat(
        self, simple_circuit, h_gate, circ_with_if, circ_with_nested_if
    ):
        """
        Tests whether the flattening function returns a flat circuit,
        i.e. whether all subcircuits have no further subcircuits.
        """

        def is_flat(circuit: Circuit):
            for subcirc in circuit.circuit:
                if (
                    len(subcirc) > 0
                    and len(subcirc[0].circuit) > 0
                    and not hasattr(subcirc[0], "_loom_ifelse_marker")
                ):
                    return False  # Subcircuit contains more subcircuits
            return True

        non_flat_circuit = Circuit(
            "non_flat",
            circuit=[simple_circuit, h_gate()],
        )
        assert not is_flat(non_flat_circuit)
        assert non_flat_circuit.flatten()

        full_circ_w_if = Circuit("full_circ_w_if", [simple_circuit, circ_with_if])
        assert is_flat(full_circ_w_if.flatten())
        assert not is_flat(full_circ_w_if)

        full_circ_w_nif = Circuit(
            "full_circ_w_nif", [simple_circuit, circ_with_nested_if]
        )

        assert not is_flat(full_circ_w_nif)
        assert is_flat(full_circ_w_nif.flatten())

    def test_flatten_several_levels_nesting(self):
        """
        Tests the flattening function for a circuit with more levels of nesting.
        """
        data_qbs = [Channel(label=f"D{i}") for i in range(3)]
        data_cregs = [
            Channel(type=ChannelType.CLASSICAL, label=f"creg_D{i+1}") for i in range(3)
        ]
        data_inits = [Circuit("Reset", channels=[q]) for q in data_qbs]
        data_init_circ = Circuit(
            "Initialization", channels=data_qbs, circuit=data_inits
        )

        hadamards = Circuit(
            "Block1",
            circuit=(
                (Circuit("H", channels=[data_qbs[0]])),
                (Circuit("H", channels=[data_qbs[1]])),
            ),
        )
        cnots = Circuit(
            "Block2",
            circuit=(
                (Circuit("CNOT", channels=[data_qbs[0], data_qbs[1]])),
                (Circuit("CNOT", channels=[data_qbs[1], data_qbs[2]])),
            ),
        )
        blocks_circ = Circuit("Combined blocks", circuit=(hadamards, cnots))
        data_measurements = [
            Circuit("Measurement", channels=[q, creg])
            for q, creg in zip(data_qbs, data_cregs, strict=True)
        ]
        data_meas_circ = Circuit(
            "Final data qubit readout",
            channels=data_qbs + data_cregs,
            circuit=data_measurements,
        )
        full_circ = Circuit(
            "Full circuit", circuit=(data_init_circ, blocks_circ, data_meas_circ)
        )

        # This is the flattened circuit we expect
        flat_circuit_expected = Circuit(
            "Full circuit",
            circuit=(
                (Circuit("Reset", channels=[data_qbs[0]])),
                (Circuit("Reset", channels=[data_qbs[1]])),
                (Circuit("Reset", channels=[data_qbs[2]])),
                (Circuit("H", channels=[data_qbs[0]])),
                (Circuit("H", channels=[data_qbs[1]])),
                (Circuit("CNOT", channels=[data_qbs[0], data_qbs[1]])),
                (Circuit("CNOT", channels=[data_qbs[1], data_qbs[2]])),
                (Circuit("Measurement", channels=[data_qbs[0], data_cregs[0]])),
                (Circuit("Measurement", channels=[data_qbs[1], data_cregs[1]])),
                (Circuit("Measurement", channels=[data_qbs[2], data_cregs[2]])),
            ),
        )

        for tick_flattened, tick_exp in zip(
            full_circ.flatten().circuit, flat_circuit_expected.circuit, strict=True
        ):
            assert tick_flattened[0].name == tick_exp[0].name
            assert tick_flattened[0].channels == tick_exp[0].channels

    def test_flatten_multiple_circuits_per_tick(self):
        """
        Tests the flattening function for a circuit with multiple circuits per tick.
        """
        qreg = [Channel(label=f"q{i}") for i in range(3)]
        tick1 = [
            Circuit("a", channels=qreg[2]),
            Circuit("b", channels=qreg[1]),
            Circuit("c", channels=qreg[0]),
        ]
        tick2 = [
            Circuit("A", channels=qreg[0]),
            Circuit("B", channels=qreg[1]),
            Circuit("C", channels=qreg[2]),
        ]
        tick3 = [Circuit("d", channels=qreg[2], duration=3)]
        new_circ = Circuit(name="circ", circuit=[tick1, tick2, tick3])
        new_circ = new_circ.flatten()
        flat_circuit = Circuit(name="flat_circuit", circuit=tick1 + tick2 + tick3)
        assert new_circ.circuit == flat_circuit.circuit

    def test_unroll_circuit(self):
        """
        Tests the unroll method for a circuit defined recursively.
        """
        qreg = [Channel(label=f"q{i}") for i in range(4)]
        circ_1a = Circuit(
            "circ_1a",
            circuit=[
                [
                    Circuit("a0", channels=qreg[0]),
                    Circuit("a1", channels=qreg[1]),
                ],
                [
                    Circuit("a0", channels=qreg[0]),
                    Circuit("a1", channels=qreg[1]),
                ],
            ],
        )
        circ_1b = Circuit(
            "circ_1b",
            circuit=[
                [
                    Circuit("b2", channels=qreg[2]),
                    Circuit("b3", channels=qreg[3]),
                ],
                [
                    Circuit("b2", channels=qreg[2]),
                    Circuit("b3", channels=qreg[3]),
                ],
            ],
        )
        circ_2a = Circuit(
            "circ_2a",
            circuit=[
                [
                    Circuit("aa0", channels=qreg[0]),
                    Circuit("aa1", channels=qreg[1]),
                    Circuit("aa2", channels=qreg[2]),
                    Circuit("aa3", channels=qreg[3]),
                ],
                [
                    Circuit("aa0", channels=qreg[0]),
                    Circuit("aa1", channels=qreg[1]),
                ],
            ],
        )
        circ_3a = Circuit("circ_3a", circuit=(), channels=qreg)
        composite_circuit = Circuit(
            "composite_circuit",
            circuit=[
                [circ_1a, circ_1b],
                [],
                [circ_2a],
                [],
                [circ_3a],
            ],
        )

        unrolled_circ = Circuit.unroll(composite_circuit)
        expected_circ = Circuit(
            "expected_circuit",
            circuit=[
                [
                    Circuit("a0", channels=qreg[0]),
                    Circuit("a1", channels=qreg[1]),
                    Circuit("b2", channels=qreg[2]),
                    Circuit("b3", channels=qreg[3]),
                ],
                [
                    Circuit("a0", channels=qreg[0]),
                    Circuit("a1", channels=qreg[1]),
                    Circuit("b2", channels=qreg[2]),
                    Circuit("b3", channels=qreg[3]),
                ],
                [
                    Circuit("aa0", channels=qreg[0]),
                    Circuit("aa1", channels=qreg[1]),
                    Circuit("aa2", channels=qreg[2]),
                    Circuit("aa3", channels=qreg[3]),
                ],
                [
                    Circuit("aa0", channels=qreg[0]),
                    Circuit("aa1", channels=qreg[1]),
                ],
                [
                    Circuit("circ_3a", channels=qreg),
                ],
            ],
        )
        assert unrolled_circ == expected_circ.circuit

    def test_unroll_circuit_with_empty_time_steps(self):
        """
        Test the unroll method for a circuit with empty time steps.
        """
        qreg = [Channel(label=f"q{i}") for i in range(4)]
        circ1 = Circuit(
            "circ1",
            circuit=[
                [Circuit("x", channels=qreg[0]), Circuit("y", channels=qreg[1])],
                [Circuit("z", channels=qreg[2])],
            ],
        )
        circ2 = Circuit(
            "circ2",
            circuit=[
                [Circuit("phase", channels=qreg[3])],
                [],
                [],
                [Circuit("phase", channels=qreg[3])],
            ],
        )
        circ_w_wait = Circuit(
            "circ_w_wait",
            circuit=[
                [circ1, circ2],
                [],
                [],
                [Circuit("zzz", channels=qreg[2])],
                [],
                [],
            ],
        )
        unrolled_circ = Circuit(
            "unrolled_circuit",
            circuit=Circuit.unroll(circ_w_wait),
        )
        expected_circ = Circuit(
            "expected_circuit",
            circuit=[
                [
                    Circuit("x", channels=qreg[0]),
                    Circuit("y", channels=qreg[1]),
                    Circuit("phase", channels=qreg[3]),
                ],
                [Circuit("z", channels=qreg[2])],
                [],
                [Circuit("zzz", channels=qreg[2]), Circuit("phase", channels=qreg[3])],
                [],
                [],
            ],
        )
        assert expected_circ == unrolled_circ

        # Check that unrolling a circuit twice gives the same result

    def test_unroll_ifelsecircuit(self):
        """
        Test the unroll method for a circuit with an IfElseCircuit.
        """
        # Set up a circuit with an IfElseCircuit
        base = Circuit("base", duration=1)
        circ = Circuit("circ", [[base], [Circuit("circ2", [[base], [], [base]])]])
        if_circ = IfElseCircuit(if_circuit=base, else_circuit=circ)

        circ_with_if = Circuit("circ_with_if1", [if_circ, circ])

        # Create unrolled and expected circuits
        unrolled_circ_with_if = Circuit.unroll(circ_with_if)
        expected_circ_with_if = Circuit(
            "expected_circ",
            circuit=[
                [if_circ],
                [],
                [],
                [],
                [base],
                [base],
                [],
                [base],
            ],
        )

        # Assertions
        # The IfElseCircuit itself contains unrolled circuits
        assert unrolled_circ_with_if == expected_circ_with_if.circuit
        assert unrolled_circ_with_if[0][0].if_circuit.circuit == base.unroll(base)
        assert unrolled_circ_with_if[0][0].else_circuit.circuit == circ.unroll(circ)

    def test_circuit_equivalence(self):
        """
        Check the equivalence of two Circuit objects.
        """
        qb_set1 = [Channel(label=f"D{i}") for i in range(3)]
        qb_set2 = [Channel(label=f"D{i}") for i in range(3)]

        c1 = Circuit("CNOT", channels=[qb_set1[0], qb_set1[1]])
        c2 = Circuit("CNOT", channels=[qb_set2[0], qb_set2[1]])
        c3 = Circuit("CZ", channels=[qb_set2[0], qb_set2[1]])
        # c1 and c2 contain the same gate (CNOT) on the same qubits, although they
        # are different objects (with different uuids)
        assert c1 == c2
        assert c2 != c3  # Different gate (CNOT vs CZ)

        two_cnots_1 = Circuit(
            name="2CNOTS",
            circuit=(
                (Circuit("CNOT", channels=[qb_set1[0], qb_set1[1]])),
                (Circuit("CNOT", channels=[qb_set1[1], qb_set1[2]])),
            ),
        )
        two_cnots_2 = Circuit(
            name="2CNOTS",
            circuit=(
                (Circuit("CNOT", channels=[qb_set2[0], qb_set2[1]])),
                (Circuit("CNOT", channels=[qb_set2[1], qb_set2[2]])),
            ),
        )
        two_cnots_3 = Circuit(
            name="2CNOTS",
            circuit=(
                (Circuit("CNOT", channels=[qb_set2[0], qb_set2[2]])),
                (Circuit("CNOT", channels=[qb_set2[1], qb_set2[2]])),
            ),
        )
        # Same gates on the same qubits:
        assert two_cnots_1 == two_cnots_2
        # Different qubits involved: CNOT(0,1), CNOT(1,2) vs CNOT(0,2), CNOT(1,2)
        assert two_cnots_1 != two_cnots_3
        # Test a few more edge cases:
        # The following two circuits are equivalent although there are a few differences
        # - the two circuits are nested in different ways
        # - the two circuits have different names
        # - subcircuits which are not physical gates have different names
        # - the two circuits use the same quantum channels but in a permuted order
        #   (qubit 0 -> qubit 2, qubit 1 -> qubit 0, qubit 2 -> qubit 1)
        nested1 = Circuit(
            name="nested_circ",
            circuit=(
                (
                    Circuit(
                        name="block1",
                        circuit=(
                            (Circuit("H", channels=[qb_set1[0]])),
                            (Circuit("CNOT", channels=[qb_set1[0], qb_set1[1]])),
                            (Circuit("CNOT", channels=[qb_set1[1], qb_set1[2]])),
                        ),
                    )
                ),
                (
                    Circuit(
                        name="block2",
                        circuit=(
                            (Circuit("Z", channels=[qb_set1[1]])),
                            (Circuit("CY", channels=[qb_set1[0], qb_set1[2]])),
                        ),
                    )
                ),
            ),
        )
        nested2 = Circuit(
            name="differently_nested_circ",
            circuit=(
                (
                    Circuit(
                        name="DifferentNameButThisIsIgnored",
                        circuit=((Circuit("H", channels=[qb_set1[2]])),),
                    )
                ),
                (
                    Circuit(
                        name="DifferentNameButThisIsIgnored",
                        circuit=(
                            (
                                Circuit(
                                    name="EvenMoreNesting",
                                    circuit=(
                                        (
                                            Circuit(
                                                "CNOT",
                                                channels=[qb_set1[2], qb_set1[0]],
                                            )
                                        ),
                                        (
                                            Circuit(
                                                "CNOT",
                                                channels=[qb_set1[0], qb_set1[1]],
                                            )
                                        ),
                                    ),
                                )
                            ),
                            (Circuit("Z", channels=[qb_set1[0]])),
                        ),
                    )
                ),
                (Circuit("CY", channels=[qb_set1[2], qb_set1[1]])),
            ),
        )
        assert nested1 == nested2
        # Test eq method for cloned circuits
        nested2_clone = nested2.clone()
        assert nested2_clone == nested2

    def test_eq_method_for_multiple_ticks(self):
        """
        Tests the __eq__() function for a circuit with multiple circuits per tick.
        """
        qreg = [Channel(label=f"q{i}") for i in range(3)]
        tick1 = [
            Circuit("a", channels=qreg[2]),
            Circuit("b", channels=qreg[1]),
            Circuit("c", channels=qreg[0]),
        ]
        tick2 = [
            Circuit("a2", channels=qreg[0]),
            Circuit("b2", channels=qreg[1]),
            Circuit("c2", channels=qreg[2]),
        ]
        tick3 = [Circuit("d", channels=qreg[2], duration=3)]
        circ_w_ticks = Circuit(name="circ_w_ticks", circuit=[tick1, tick2, tick3])
        circ_wo_ticks = Circuit(name="circ_wo_ticks", circuit=tick1 + tick2 + tick3)
        # The two circuit are not equal because of the tick structure
        assert circ_w_ticks != circ_wo_ticks
        # The two circuits are equal when flattened
        assert circ_w_ticks.flatten() == circ_wo_ticks

    def test_equality_w_time_structure(self):
        """
        Tests the __eq__() function for circuits with different_time_structure.
        """
        # Case 1 - two circuits with different time structure
        qubits = [Channel(label=f"q{i}") for i in range(3)]
        circ_1a = Circuit(
            name="circ_1a",
            circuit=[
                [Circuit("H", channels=[qubits[0]])],
                [Circuit("H", channels=[qubits[1]])],
            ],
        )
        circ_1b = Circuit(
            name="circ_1b",
            circuit=[
                [
                    Circuit("H", channels=[qubits[0]]),
                    Circuit("H", channels=[qubits[1]]),
                ],
            ],
        )
        assert circ_1a != circ_1b
        # Case 2 - example with multiple time-steps and change in ordering of the
        # channels. Note we use the cnot gate to enforce ordering of the qubits
        circ_2a = Circuit(
            name="circ_2a",
            circuit=[
                [
                    Circuit("H", channels=[qubits[0]]),
                ],
                [
                    Circuit("X", channels=[qubits[1]]),
                ],
                [
                    Circuit("CNOT", channels=[qubits[0], qubits[1]]),
                ],
            ],
        )
        # This circuit is equivalent to circ_2a but with other Channel objects
        circ_2a_copy = Circuit(
            name="circ_2a",
            circuit=[
                [
                    Circuit("H", channels=[qubits[1]]),
                ],
                [
                    Circuit("X", channels=[qubits[0]]),
                ],
                [
                    Circuit("CNOT", channels=[qubits[1], qubits[0]]),
                ],
            ],
        )
        # This circuit is different from circ_2a, because the CNOT acts on 0 then 1
        circ_2b = Circuit(
            name="circ_2b",
            circuit=[
                [
                    Circuit("H", channels=[qubits[0]]),
                ],
                [
                    Circuit("X", channels=[qubits[1]]),
                ],
                [
                    Circuit("CNOT", channels=[qubits[1], qubits[0]]),
                ],
            ],
        )
        circ_2c = Circuit(
            name="circ_2c",
            circuit=[
                [
                    Circuit("H", channels=[qubits[0]]),
                    Circuit("X", channels=[qubits[1]]),
                ],
                [
                    Circuit("CNOT", channels=[qubits[0], qubits[1]]),
                ],
            ],
        )
        assert circ_2a == circ_2a_copy
        assert circ_2a != circ_2b
        assert circ_2b != circ_2c
        assert circ_2a != circ_2c

    def test_equality_empty_time_steps(self):
        """
        Tests the __eq__() function for circuits with or without empty time steps.
        """
        qubits = [Channel(label="q")]
        circ_1 = Circuit(
            name="circ_1",
            circuit=[
                [Circuit("H", channels=qubits)],
                [],
            ],
        )
        circ_2 = Circuit(
            name="circ_2",
            circuit=[
                [Circuit("H", channels=qubits)],
            ],
        )
        assert circ_1 != circ_2

    def test_equality_permuted_steps(self):
        """
        Tests the __eq__() function for circuits with permuted steps.
        """
        qubits = [Channel(label=f"q{i}") for i in range(2)]
        circ_1 = Circuit(
            name="circ_1",
            circuit=[
                [Circuit("H", channels=[qubits[0]])],
                [Circuit("X", channels=[qubits[1]])],
            ],
        )
        circ_2 = Circuit(
            name="circ_2",
            circuit=[
                [Circuit("X", channels=[qubits[1]])],
                [Circuit("H", channels=[qubits[0]])],
            ],
        )
        assert circ_1 != circ_2

    def test_construct_padded_circuit_time_sequence(self, cnot_gate):
        """
        Tests that the function construct_padded_circuit_time_sequence constructs
        a padded circuit time sequence correctly.
        """
        channels = tuple(Channel(label=f"q{i}") for i in range(3))
        hadamard = Circuit("h", duration=1, channels=[channels[0]])
        if_circ = IfElseCircuit(
            if_circuit=cnot_gate(channels[0], channels[1]), else_circuit=hadamard
        )
        long_hadamard_1 = Circuit("h", duration=3, channels=[channels[1]])
        long_hadamard_2 = Circuit("h", duration=3, channels=[channels[2]])
        cnot = Circuit("cnot", duration=2, channels=channels[0:2])
        toffoli = Circuit("toffoli", duration=4, channels=channels)

        initial_and_padded_sequences = [
            (  # Time sequence with a single sub-circuit and no duration
                ((hadamard,),),  # initial
                ((hadamard,),),  # padded sequence is the same
            ),
            (  # Time sequence with a single sub-circuit and a duration
                ((long_hadamard_1,),),  # initial
                ((long_hadamard_1,), (), ()),  # padded sequence
            ),
            (  # Time sequence where the gates can be executed in parallel
                # but exist in different timesteps
                ((long_hadamard_1,), (long_hadamard_2,)),  # initial
                ((long_hadamard_1,), (long_hadamard_2,), (), ()),  # padded sequence
            ),
            (  # Time sequence with multiple sub-circuits of different duration
                # on different channels
                (
                    (
                        long_hadamard_2,
                        hadamard,
                    ),
                ),  # initial
                (
                    (
                        long_hadamard_2,
                        hadamard,
                    ),
                    (),
                    (),
                ),  # padded sequence
            ),
            (  # Time sequence with multiple sub-circuits on a single channel
                ((hadamard,), (cnot,)),  # initial
                ((hadamard,), (cnot,), ()),  # padded sequence
            ),
            (  # Time seq with multiple sub-circuits on a single channel (diff order)
                ((cnot,), (hadamard,)),  # initial
                ((cnot,), (), (hadamard,)),  # padded sequence
            ),
            (  # Time sequence with multiple time steps and sub-circuits
                # of different duration on different channels
                (
                    (
                        long_hadamard_2,
                        hadamard,
                    ),
                    (cnot,),
                    (toffoli,),
                ),  # initial
                (
                    (
                        long_hadamard_2,
                        hadamard,
                    ),
                    (cnot,),
                    (),
                    (toffoli,),
                    (),
                    (),
                    (),
                ),  # padded sequence
            ),
            (  # Time sequence with multiple time steps and sub-circuits of
                # different duration on different channels
                (
                    (
                        long_hadamard_1,
                        hadamard,
                    ),
                    (cnot,),
                    (toffoli,),
                ),  # initial
                (
                    (
                        long_hadamard_1,
                        hadamard,
                    ),
                    (),
                    (),
                    (cnot,),
                    (),
                    (toffoli,),
                    (),
                    (),
                    (),
                ),  # padded sequence
            ),
            (  # Time sequence with extra empty time steps that are not modified
                ((hadamard,), (), (cnot,)),  # initial with extra padding
                ((hadamard,), (), (cnot,), ()),  # padded sequence
            ),
            (  # Time seq with extra empty t steps that are part of the expected padding
                ((long_hadamard_1,), (), (cnot,)),  # initial with extra padding
                ((long_hadamard_1,), (), (), (cnot,), ()),  # padded sequence
            ),
            (
                ((hadamard,), (if_circ,)),  # initial
                (
                    (hadamard,),
                    (if_circ,),
                    (),
                    (),
                    (),
                    (),
                ),  # padded sequence
            ),
        ]
        for initial_sequence, expected_padded_sequence in initial_and_padded_sequences:
            padded_circ_timeseq = Circuit.construct_padded_circuit_time_sequence(
                initial_sequence
            )
            assert padded_circ_timeseq == expected_padded_sequence

    def test_empty_circuit(self):
        """Tests that an empty circuit is handled correctly."""
        empty_circ = Circuit("empty")
        assert empty_circ.duration == 1
        assert empty_circ.channels == ()
        assert empty_circ.circuit == ()
