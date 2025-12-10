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

from loom.eka import Circuit
from loom.interpreter import InterpretationStep

from .y_wall_out_utilities import (
    get_final_block_syndrome_measurement_cnots_circuit,
)
from ..move_block import (
    generate_syndrome_measurement_circuit_and_cbits,
    generate_and_append_block_syndromes_and_detectors,
)
from ...code_factory import RotatedSurfaceCode


def y_wall_out_final_qec_rounds(
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    same_timeslice: bool,
    debug_mode: bool,  # pylint: disable=unused-argument
) -> InterpretationStep:
    """
    Perform the final syndrome measurement rounds for the Y wall out operation.
    This operation performs d-2 rounds of syndrome measurement on the block in such
    a way that the final circuit is fault-tolerant.

    - A.) Begin composite operation for potentially multiple rounds

    - B.) Circuit generation
        - B.1) Generate ancilla initialization circuit
        - B.2) Generate CNOT circuit
        - B.3) Generate measurement circuit and classical bits
        - B.4) Assemble and append circuit

    - C.) Syndrome and detector generation
        - C.1) Generate and append block syndromes and detectors

    - D.) End the composite operation and append the circuit

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step.
    block : RotatedSurfaceCode
        The rotated surface code block to perform the final syndrome measurement rounds
        on.
    same_timeslice : bool
        Whether to append the final circuit in the same timeslice as the previous one.
    debug_mode : bool
        Whether to enable debug mode. (Unused since no new Blocks are created here)

    Returns
    -------
    InterpretationStep
        The updated interpretation step.
    """

    # Run d-2 rounds of syndrome measurement
    distance = min(block.size)
    n_total_rounds = distance - 2

    # A) Begin composite operation for potentially multiple rounds
    interpretation_step.begin_composite_operation_session_MUT(
        same_timeslice=same_timeslice,
        circuit_name=(
            f"final {n_total_rounds} syndrome measurement round(s) "
            f"on block {block.unique_label}"
        ),
    )

    for _ in range(n_total_rounds):
        # B) Circuit generation
        # B.1) Generate ancilla initialization circuit
        final_block_syndrome_measurement_reset_circuit = Circuit(
            name="Initialization of syndrome measurement ancilla",
            circuit=[
                [
                    Circuit(
                        f"reset_{'0' if stab.pauli_type == 'Z' else '+'}",
                        channels=[
                            interpretation_step.get_channel_MUT(stab.ancilla_qubits[0])
                        ],
                    )
                    for stab in block.stabilizers
                ]
            ],
        )

        # B.2) Generate CNOT circuit
        final_block_syndrome_measurement_cnot_circuit = (
            get_final_block_syndrome_measurement_cnots_circuit(
                block, interpretation_step
            )
        )

        # B.3) Generate measurement circuit and classical bits
        final_block_syndrome_measurement_measure_circuit, final_block_meas_cbits = (
            generate_syndrome_measurement_circuit_and_cbits(
                interpretation_step,
                block,
            )
        )

        # B.4) Assemble and append circuit
        interpretation_step.append_circuit_MUT(
            Circuit(
                name="one round of final syndrome measurement",
                circuit=Circuit.construct_padded_circuit_time_sequence(
                    (
                        (final_block_syndrome_measurement_reset_circuit,),
                        (final_block_syndrome_measurement_cnot_circuit,),
                        (final_block_syndrome_measurement_measure_circuit,),
                    )
                ),
            ),
            same_timeslice=False,
        )

        # C) Syndrome and detector generation
        generate_and_append_block_syndromes_and_detectors(
            interpretation_step, block, final_block_meas_cbits
        )

    # D) End the composite operation and append the circuit
    wrapped_circuit = interpretation_step.end_composite_operation_session_MUT()
    interpretation_step.append_circuit_MUT(wrapped_circuit, same_timeslice)

    return interpretation_step
