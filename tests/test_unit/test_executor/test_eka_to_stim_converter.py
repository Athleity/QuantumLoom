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

import builtins

import pytest
import stim
import numpy as np

from loom.eka import ChannelType, Circuit, Channel
from loom.executor import EkaToStimConverter

# pylint: disable=no-member


class TestEkaToStimConverter:
    """
    Test the conversion functionality of the class,
    and verify that conversion happens with consistency
    """

    convert_expected_json = "convert_stim.json"
    emit_expected_json = "emit_stim.json"

    SUPP_CIRCUIT_FIXTURES = [
        "empty_circuit",
        "simple_circuit",
        "bell_state_circuit",
        "circuit_with_nested_circuits",
        "circuit_with_multiple_nested_levels",
        "circuit_w_all_two_qubit_ops",
        "circuit_w_all_measurement_ops",
        "circuit_w_all_reset_ops",
        "circuit_if_with_empty_else",
        "circuit_surface_code_experiment",
    ]

    # We expect these to fail due to unsupported features in Stim
    UNSUPPORTED_CIRCUIT_FIXTURES = [
        "circuit_with_simple_if_else",
        "circuit_with_boolean_logic_condition",
        "circuit_w_nested_if_else",
        "circuit_with_boolean_logic_condition_multibit",
    ]

    @property
    def converter(self) -> EkaToStimConverter:
        """Return an instance of the EkaToStimConverter."""
        return EkaToStimConverter()

    @pytest.mark.parametrize("input_fixture", SUPP_CIRCUIT_FIXTURES, indirect=True)
    @pytest.mark.parametrize(
        "load_expected_data", [convert_expected_json], indirect=True
    )
    def test_generic_circuit_conversion(self, input_fixture, load_expected_data):
        """Test conversion of generic circuits to Stim format."""

        # Get the circuit name
        fixture_content, fixture_name = input_fixture

        # Convert the circuit using the converter
        result = self.converter.convert_circuit(fixture_content)

        circuit, q_map, c_map = result

        assert isinstance(circuit, (stim.Circuit, str))
        assert isinstance(c_map, dict)
        assert isinstance(q_map, dict)

        assert set(c for c in fixture_content.channels if c.is_classical()).issubset(
            set(c_map.keys())
        ), "Not all classical channels mapped to a outcome register"

        # Here we skip the empty circuit as it has no quantum channels, but the
        # Eka.circuit is actually adding one for no reason
        if fixture_name != "empty_circuit":
            assert set(c for c in fixture_content.channels if c.is_quantum()).issubset(
                set(q_map.keys())
            ), "Not all quantum channels mapped to a stim qubit register"

        # If we have expected output data for this fixture, perform detailed assertions
        if fixture_name not in load_expected_data:
            pytest.fail(
                f"Expected data for fixture '{fixture_name}' not found in expected "
                "data."
            )
        expected = load_expected_data[fixture_name]
        observed = list(str(circuit).split("\n"))
        assert (
            expected["circuit"] == observed
        ), f"Circuit mismatch, expected {expected['circuit']}, got {observed}"

    @pytest.mark.parametrize(
        "input_fixture", UNSUPPORTED_CIRCUIT_FIXTURES, indirect=True
    )
    def test_generic_unsupported_circuit_conversion(self, input_fixture):
        """Test conversion of unsupported circuits to Stim format."""

        fixture_content, _ = input_fixture

        # Convert the circuit using the converter
        with pytest.raises(
            ValueError, match="Unsupported operation for Stim conversion"
        ):
            self.converter.convert_circuit(fixture_content)

    SUPP_ISTEP_FIXTURES = [
        "interpreted_empty_lscrd",
        "rsc_memory_experiment",
    ]

    @pytest.mark.parametrize("input_fixture", SUPP_ISTEP_FIXTURES, indirect=True)
    @pytest.mark.parametrize(
        "load_expected_data", [convert_expected_json], indirect=True
    )
    def test_generic_istep_conversion(self, input_fixture, load_expected_data):
        """Test conversion of generic interpretation steps to Stim format."""

        # Get the circuit name
        fixture_content, fixture_name = input_fixture

        # Convert the circuit using the converter
        circ, qreg, creg = self.converter.convert(fixture_content)

        assert isinstance(circ, stim.Circuit)
        assert isinstance(creg, dict)
        assert isinstance(qreg, dict)

        assert set(
            c for c in fixture_content.final_circuit.channels if c.is_classical()
        ).issubset(
            set(creg.keys())
        ), "Not all classical channels mapped to a outcome register"

        # Here we skip the empty circuit as it has no quantum channels, but the
        # Eka.circuit is actually adding one for no reason
        if fixture_name != "interpreted_empty_lscrd":
            assert set(
                c for c in fixture_content.final_circuit.channels if c.is_quantum()
            ).issubset(
                set(qreg.keys())
            ), "Not all quantum channels mapped to a stim qubit register"

        # If we have expected output data for this fixture, perform detailed assertions
        if fixture_name not in load_expected_data:
            pytest.fail(
                f"Expected data for fixture '{fixture_name}' not found "
                "in expected data."
            )
        expected = load_expected_data[fixture_name]
        observed = list(str(circ).split("\n"))

        assert (
            expected["circuit"] == observed
        ), f"Circuit mismatch, expected {expected['circuit']}, got {observed}"

    @pytest.mark.parametrize(
        "cem_instance",
        [
            {
                "circuit_fixture": "simple_circuit",
                "cem_class_fixture": "all_error_types_cem",
            },
            {
                "circuit_fixture": "circuit_surface_code_experiment",
                "cem_class_fixture": "all_error_types_cem",
            },
            {
                "circuit_fixture": "circuit_with_nested_circuits",
                "cem_class_fixture": "all_application_modes_cem",
            },
            {
                "circuit_fixture": "circuit_surface_code_experiment",
                "cem_class_fixture": "complex_cem",
            },
            {
                "circuit_fixture": "simple_circuit",
                "cem_class_fixture": "complex_cem",
            },
        ],
        indirect=True,  # pass the dict to the fixture, not the test directly
    )
    @pytest.mark.parametrize(
        "load_expected_data", [convert_expected_json], indirect=True
    )
    def test_conversion_with_noise(self, cem_instance, load_expected_data):
        """Test conversion of circuits with CircuitErrorModel to Stim format."""
        # Get the circuit content and fixture name
        cem, circuit, test_name = cem_instance

        expected = load_expected_data[test_name]

        if not isinstance(cem, list):
            cem = [cem]
        # Convert the circuit using the converter
        circ, _, __ = self.converter.convert_circuit(
            circuit, with_ticks=True, error_models=cem
        )

        observed = list(str(circ).split("\n"))

        assert observed == expected["circuit"], f"Mismatch in circuit for {test_name}"

    def test_parse_outcome(self):
        """Test parsing of stim run output into expected format."""

        stim_output = np.array(
            [
                [False, True, False, True],
                [True, False, True, False],
                [False, False, True, True],
            ],
            dtype=bool,
        )

        classical_map = {
            Channel(label="c0", type=ChannelType.CLASSICAL): 0,
            Channel(label="c1", type=ChannelType.CLASSICAL): 1,
            Channel(label="c2", type=ChannelType.CLASSICAL): 2,
            Channel(label="c3", type=ChannelType.CLASSICAL): 3,
        }

        expected_parsed = {
            "c0": [False, True, False],
            "c1": [True, False, False],
            "c2": [False, True, True],
            "c3": [True, False, True],
        }

        parsed = EkaToStimConverter.parse_target_run_outcome(
            (stim_output, classical_map)
        )
        assert (
            parsed == expected_parsed
        ), f"Parsed outcome mismatch, expected {expected_parsed}, got {parsed}"

    @pytest.mark.parametrize("load_expected_data", [emit_expected_json], indirect=True)
    def test_emit_functions(
        self, circuit_to_init_and_instructions_to_append, load_expected_data, subtests
    ):
        """Test the emit functions of the converter."""
        # If we have expected output data for this fixture, perform detailed assertions
        circuit, instructions_to_append = circuit_to_init_and_instructions_to_append

        q_chan = [c for c in circuit.channels if c.type == ChannelType.QUANTUM]
        c_chan = [c for c in circuit.channels if c.type == ChannelType.CLASSICAL]

        expected = load_expected_data

        with subtests.test(msg="Checking emit initialisation", case_id="init"):
            str_init, qreg_map, creg_map = self.converter.emit_init_instructions(
                circuit
            )
            str_mismatch_msg = (
                f"Initialization string does not match:\n"
                f"    - got      : {str_init}\n"
                f"    - expected : {expected['str_init']}"
            )
            assert str_init == expected["str_init"], str_mismatch_msg
            assert all(
                k in qreg_map for k in q_chan
            ), "Not all channels are present in the quantum map"
            assert all(
                k in creg_map for k in c_chan
            ), "Not all channels are present in the classical map"

        for new_instruction_name, new_instruction in instructions_to_append.items():
            if not isinstance(new_instruction, Circuit):
                pytest.fail(
                    f"Instruction to append must be a Circuit, "
                    f"got {type(new_instruction)}"
                )
            with subtests.test(
                msg=f"Checking emit of {new_instruction_name} instruction",
                case_id=new_instruction_name,
            ):
                instruction_expected = expected[new_instruction_name]
                if instruction_expected["success"]:
                    str_inst = self.converter.emit_leaf_circuit_instruction(
                        new_instruction,
                        qreg_map,
                        creg_map,
                        measurement_record_counter=2,
                    )
                    inst_mismatch_msg = (
                        f"Instruction string does not match "
                        f"for {new_instruction_name}:\n"
                        f"    - got      : {str_inst}\n"
                        f"    - expected : {instruction_expected['str_inst']}"
                    )
                    assert (
                        str_inst == instruction_expected["str_inst"]
                    ), inst_mismatch_msg
                else:
                    with pytest.raises(
                        getattr(builtins, instruction_expected["expected_exception"])
                    ):
                        self.converter.emit_leaf_circuit_instruction(
                            new_instruction, qreg_map, creg_map
                        )

    @pytest.mark.parametrize(
        "input_fixture",
        [
            "rsc_memory_experiment",
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "load_expected_data", [convert_expected_json], indirect=True
    )
    def test_crumble_generation(self, input_fixture, load_expected_data):
        """Test the generation of crumble instructions of the rotated surface code"""
        fixture_content, fixture_name = input_fixture
        expected_output = load_expected_data[fixture_name]["crumble_circuit"]
        generated_crumble_str = self.converter.print_stim_circuit_for_crumble(
            fixture_content
        )

        assert generated_crumble_str.split("\n") == expected_output
