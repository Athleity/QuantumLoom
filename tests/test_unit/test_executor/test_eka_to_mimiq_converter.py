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
from functools import cached_property

import pytest

from loom.eka import ChannelType, Circuit
from loom.executor import EkaToMimiqConverter


class TestEkaToMimiqConverter:
    """Unit tests for the Eka to MIMIQ circuit converter."""

    convert_expected_json = "convert_mimiq.json"
    emit_expected_json = "emit_mimiq.json"

    SUPP_CIRCUIT_FIXTURES = [
        "empty_circuit",
        "simple_circuit",
        "bell_state_circuit",
        "circuit_with_nested_circuits",
        "circuit_with_multiple_nested_levels",
        "circuit_w_all_single_qubit_ops",
        "circuit_w_all_two_qubit_ops",
        "circuit_w_all_measurement_ops",
        "circuit_w_all_reset_ops",
        "circuit_if_with_empty_else",
        "circuit_surface_code_experiment",
    ]

    @cached_property
    def converter(self) -> EkaToMimiqConverter:
        """Return the converter instance to be tested."""
        return EkaToMimiqConverter()

    @pytest.mark.parametrize("input_fixture", SUPP_CIRCUIT_FIXTURES, indirect=True)
    @pytest.mark.parametrize(
        "load_expected_data", [convert_expected_json], indirect=True
    )
    def test_using_generic_cases(self, input_fixture, load_expected_data):
        """Test the converter using generic circuit cases."""

        # Get the circuit name
        fixture_content, fixture_name = input_fixture

        converted_circuit, quantum_reg_mapping, classical_reg_mapping = (
            self.converter.convert_circuit(fixture_content)
        )

        assert isinstance(converted_circuit, str)
        assert isinstance(quantum_reg_mapping, dict)
        assert isinstance(classical_reg_mapping, dict)

        assert set(c for c in fixture_content.channels if c.is_classical()).issubset(
            set(classical_reg_mapping.keys())
        ), "Not all classical channels mapped to a outcome register"

        assert set(c for c in fixture_content.channels if c.is_quantum()).issubset(
            set(quantum_reg_mapping.keys())
        ), "Not all quantum channels mapped to a stim qubit register"

        if fixture_name not in load_expected_data:
            pytest.fail(
                "Expected data for fixture "
                f"'{fixture_name}' not found in expected data."
            )
        expected = load_expected_data[fixture_name]
        observed = list(str(converted_circuit).split("\n"))
        assert (
            expected["program"] == observed
        ), f"Circuit mismatch, expected {expected['program']}, got {observed}"

    # We expect these to fail due to unsupported features in Stim
    UNSUPPORTED_CIRCUIT_FIXTURES = [
        "circuit_with_simple_if_else",
        "circuit_with_boolean_logic_condition",
        "circuit_w_nested_if_else",
        "circuit_with_boolean_logic_condition_multibit",
    ]

    @pytest.mark.parametrize(
        "input_fixture", UNSUPPORTED_CIRCUIT_FIXTURES, indirect=True
    )
    def test_generic_unsupported_circuit_conversion(self, input_fixture):
        """Test conversion of unsupported circuits to Mimiq format."""

        fixture_content, _ = input_fixture

        # Convert the circuit using the converter
        with pytest.raises(
            ValueError, match="Unsupported operation for Mimiq conversion"
        ):
            self.converter.convert_circuit(fixture_content)

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
                    "Instruction to append must "
                    f"be a Circuit, got {type(new_instruction)}"
                )
            with subtests.test(
                msg=f"Checking emit of {new_instruction_name} instruction",
                case_id=new_instruction_name,
            ):
                instruction_expected = expected[new_instruction_name]
                if instruction_expected["success"]:
                    str_inst = self.converter.emit_leaf_circuit_instruction(
                        new_instruction, qreg_map, creg_map
                    )
                    inst_mismatch_msg = (
                        "Instruction string does not"
                        f" match for {new_instruction_name}:\n"
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
