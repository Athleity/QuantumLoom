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
from loom.executor.circuit_error_model import (
    CircuitErrorModel,
)
from loom.eka import Circuit


class TestCircuitErrorModel:
    """Unit tests for the CircuitErrorModel class."""

    @staticmethod
    @pytest.mark.parametrize(
        "cem_instance",
        [
            {
                "circuit_fixture": "simple_circuit",
                "cem_class_fixture": "empty_error_model",
            },
            {
                "circuit_fixture": "circuit_surface_code_experiment",
                "cem_class_fixture": "empty_error_model",
            },
        ],
        indirect=True,  # pass the dict to the fixture, not the test directly
    )
    def test_empty_cem(cem_instance):
        """Test that an empty CircuitErrorModel can be instantiated without errors.
        Also test that get_gate_error_probability returns None for all gates.
        """
        cem, _, __ = cem_instance
        assert isinstance(cem, CircuitErrorModel)

        unroll_circuit = Circuit.unroll(cem.circuit)
        assert all(
            cem.get_gate_error_probability(gate) is None
            for tick in unroll_circuit
            for gate in tick
        )

    @staticmethod
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
        ],
        indirect=True,  # pass the dict to the fixture, not the test directly
    )
    def test_all_error_types_cem(cem_instance):
        """Test that a CircuitErrorModel can be instantiated with all error types."""
        cem_list, _, __ = cem_instance
        assert isinstance(cem_list, list)

        for cem in cem_list:
            unroll_circuit = Circuit.unroll(cem.circuit)
            assert all(
                (
                    cem.get_gate_error_probability(gate)[0] == 0.02
                    if gate.name == "cx"
                    else True
                )
                for tick in unroll_circuit
                for gate in tick
            )

    @staticmethod
    @pytest.mark.parametrize(
        "cem_instance",
        [
            {
                "circuit_fixture": "simple_circuit",
                "cem_class_fixture": "all_application_modes_cem",
            },
            {
                "circuit_fixture": "circuit_surface_code_experiment",
                "cem_class_fixture": "all_application_modes_cem",
            },
        ],
        indirect=True,  # pass the dict to the fixture, not the test directly
    )
    def test_all_application_modes(cem_instance):
        """
        Test that CircuitErrorModel can be instantiated with all application modes.
        """
        cem_list, _, __ = cem_instance
        assert isinstance(cem_list, list)

        for cem in cem_list:
            assert isinstance(cem, CircuitErrorModel)
