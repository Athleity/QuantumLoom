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

import stim


# pylint: disable=no-member


def noise_annotated_stim_circuit(
    stim_circ: stim.Circuit,
    before_measure_flip_probability: float = 0,
    after_clifford_depolarization: float = 0,
    after_reset_flip_probability: float = 0,
) -> stim.Circuit:
    """
    This function takes as input a pure (sans noise) stim
    circuit, and outputs a the circuit with the desired noise model

    Parameters
    ----------
    stim_circ : stim.Circuit
        The input noiseless stim circuit
    before_measure_flip_probability: float, optional
        X_ERROR probability before a measurement. Default set to
        0 will add no measurement errors
    after_clifford_depolarization: float, optional
        applies DEPOLARIZING_ERROR1 and DEPOLARIZING_ERROR2
        after each single and two qubit clifford gate in the circuit.
        Default set to 0 will add no depolarization errors
    after_reset_flip_probability: float, optional
        Apply an X_ERROR with this probability after a reset gate.
        Default set to 0 will add no reset errors

    Returns
    -------
    stim.Circuit
        stim circuit annotated with the input noise model
    """

    stim_one_qubit_ops = ["H", "X", "Y", "Z", "I"]
    stim_two_qubit_ops = ["CX", "CY", "CZ", "SWAP"]

    def return_annotated_operation(op: stim.CircuitInstruction):
        """
        Append/Prepend an appropriate stim annotation to the
        corresponding stim operation based on how the
        converter has been configured.
        For e.g.
            This operation appends X_ERROR annotation
            before each measurement round, if the
            corresponding error is turned on.
        """
        op_name = op.name
        targets = op.targets_copy()

        annotated_ops_list = [
            {"name": op_name, "targets": targets, "gate_args": op.gate_args_copy()}
        ]

        if op_name == "M" and before_measure_flip_probability > 0:
            annotation = [
                {
                    "name": "X_ERROR",
                    "targets": targets,
                    "gate_args": [before_measure_flip_probability],
                }
            ]
            annotated_ops_list = annotation + annotated_ops_list
        if op_name in stim_one_qubit_ops and after_clifford_depolarization > 0:
            annotation = [
                {
                    "name": "DEPOLARIZE1",
                    "targets": targets,
                    "gate_args": [after_clifford_depolarization],
                }
            ]
            annotated_ops_list = annotated_ops_list + annotation
        if op_name in stim_two_qubit_ops and after_clifford_depolarization > 0:
            annotation = [
                {
                    "name": "DEPOLARIZE2",
                    "targets": targets,
                    "gate_args": [after_clifford_depolarization],
                }
            ]
            annotated_ops_list = annotated_ops_list + annotation
        if op_name == "R" and after_reset_flip_probability > 0:
            annotation = [
                {
                    "name": "X_ERROR",
                    "targets": targets,
                    "gate_args": [after_reset_flip_probability],
                }
            ]
            annotated_ops_list = annotated_ops_list + annotation

        annotated_stim_ops_list = [
            stim.CircuitInstruction(
                name=args_dict["name"],
                targets=args_dict["targets"],
                gate_args=args_dict["gate_args"],
            )
            for args_dict in annotated_ops_list
        ]

        return annotated_stim_ops_list

    annotated_stim_circuit = stim.Circuit()
    for op in stim_circ:
        stim_annotated_op_list = return_annotated_operation(op)
        for annotated_op in stim_annotated_op_list:
            annotated_stim_circuit.append(annotated_op)

    return annotated_stim_circuit
