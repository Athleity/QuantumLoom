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


from loom.eka import ChannelType, Circuit

from loom.executor import EkaToPennylaneConverter


class TestEkaToPennyLaneConverter:
    """Tests for the Eka to PennyLane converter."""

    convert_expected_json = "convert_pennylane.json"
    emit_expected_json = "emit_pennylane.json"

    SUPP_CIRCUIT_FIXTURES = [
        "empty_circuit",
        "simple_circuit",
        "bell_state_circuit",
        "circuit_with_nested_circuits",
        "circuit_with_multiple_nested_levels",
        "circuit_w_all_single_qubit_ops",
        "circuit_w_all_two_qubit_ops",
        "circuit_w_all_reset_ops",
        "circuit_w_all_measurement_ops",
        "circuit_with_simple_if_else",
        "circuit_if_with_empty_else",
        "circuit_with_boolean_logic_condition",
        "circuit_with_boolean_logic_condition_multibit",
        "circuit_surface_code_experiment",
    ]

    @property
    def converter(self) -> EkaToPennylaneConverter:
        """Return an instance of the EkaToPennylaneConverter."""
        return EkaToPennylaneConverter(is_catalyst=False)

    @property
    def converter_catalyst(self) -> EkaToPennylaneConverter:
        """Return an instance of the EkaToPennylaneConverter for catalyst."""
        return EkaToPennylaneConverter(is_catalyst=True)

    @pytest.mark.parametrize("input_fixture", SUPP_CIRCUIT_FIXTURES, indirect=True)
    @pytest.mark.parametrize(
        "load_expected_data", [convert_expected_json], indirect=True
    )
    @pytest.mark.parametrize("use_catalyst", [False, True])
    def test_using_generic_cases(self, input_fixture, load_expected_data, use_catalyst):
        """Test the converter using generic circuit fixtures."""

        fixture_content, fixture_name = input_fixture

        # Convert the circuit using the converter
        if use_catalyst:
            converter = self.converter_catalyst
        else:
            converter = self.converter
        result = converter.convert_circuit(fixture_content)

        circuit, q_map, c_map = result

        assert isinstance(circuit, str)
        assert isinstance(c_map, dict)
        assert isinstance(q_map, dict)

        assert set(c.id for c in fixture_content.channels if c.is_classical()).issubset(
            set(c_map.keys())
        ), "Not all classical channels mapped to a outcome register"

        assert set(c.id for c in fixture_content.channels if c.is_quantum()).issubset(
            set(q_map.keys())
        ), "Not all quantum channels mapped to a stim qubit register"

        # If we have expected output data for this fixture, perform detailed assertions
        if fixture_name not in load_expected_data:
            pytest.fail(
                f"Expected data for fixture '{fixture_name}'"
                " not found in expected data."
            )
        expected = load_expected_data[fixture_name]
        expected_program = (
            expected["program"] if not use_catalyst else expected["program_catalyst"]
        )
        circuit_program = circuit.splitlines()

        # line with the channels id, we ignore it
        if len(circuit_program) > 3 and len(expected_program) > 3:
            circuit_program[3] = ""
            expected_program[3] = ""

        assert len(circuit_program) == len(expected_program), (
            f"Program length mismatch for fixture '{fixture_name}': "
            f"expected {len(expected_program)}, got {len(circuit_program)}"
        )
        for i, (got_line, expected_line) in enumerate(
            zip(circuit_program, expected_program, strict=True)
        ):
            assert repr(got_line) == repr(expected_line), (
                f"Line {i} mismatch for fixture '{fixture_name}':"
                f" expected {expected_line}, got {got_line}"
            )

    def test_parsing_method(self):
        """Test the parse_target_run_outcome static method of the converter."""
        outcome = {
            "c_0": [True],
            "c_1": [False, True],
        }
        expected = {
            "c_0": [1],
            "c_1": [0, 1],
        }

        assert EkaToPennylaneConverter.parse_target_run_outcome(outcome) == expected

    @pytest.mark.parametrize("load_expected_data", [emit_expected_json], indirect=True)
    def test_emit_functions(
        self, circuit_to_init_and_instructions_to_append, load_expected_data, subtests
    ):
        """Test the emit functions of the converter."""
        # If we have expected output data for this fixture, perform detailed assertions
        circuit, instructions_to_append = circuit_to_init_and_instructions_to_append

        q_chan = [c for c in circuit.channels if c.type == ChannelType.QUANTUM]

        expected = load_expected_data

        with subtests.test(msg="Checking emit initialisation", case_id="init"):
            _, q_reg, c_reg = self.converter.emit_init_instructions(circuit)
            assert all(
                k.id in q_reg for k in q_chan
            ), "Not all channels are present in the quantum map"

        for new_instruction_name, new_instruction in instructions_to_append.items():
            if not isinstance(new_instruction, Circuit):
                pytest.fail(
                    "Instruction to append must be a "
                    f"Circuit, got {type(new_instruction)}"
                )
            with subtests.test(
                msg=f"Checking emit of {new_instruction_name} instruction",
                case_id=new_instruction_name,
            ):
                instruction_expected = expected[new_instruction_name]
                if instruction_expected["success"]:
                    str_inst = self.converter.emit_leaf_circuit_instruction(
                        new_instruction, q_reg, c_reg
                    )
                    inst_mismatch_msg = (
                        "Instruction string does not "
                        f"match for {new_instruction_name}:\n"
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
                            new_instruction, q_reg, c_reg
                        )
