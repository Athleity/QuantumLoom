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

from abc import ABC, abstractmethod
from collections import deque
from functools import cached_property
from typing import Any, Callable, Generic, Optional, TypeVar

from pydantic import BaseModel, Field, model_validator

from ..eka import Circuit
from ..interpreter import InterpretationStep

from .op_signature import ALL_EKA_OP_SIGNATURES, OpSignature, OpType


# Define generic type variables for the converter
# Each of the child classes will specify these types

# TargetType is the type returned by the convert methods, usually this will be a
# string of the program
TargetType = TypeVar("TargetType")  # pylint: disable=invalid-name

# TargetRunResultType is the type returned by running the converted circuit in the
# target language, if applicable
TargetRunResultType = TypeVar("TargetRunResultType")  # pylint: disable=invalid-name

# Define a type for the callable that maps operation and targets to instruction string
# This is usually used as type for the value in the operations_map property of each
# converter
OpAndTargetToInstrCallable = Callable[[list[str], list[str], Optional[str]], str]


class Converter(ABC, BaseModel, Generic[TargetType, TargetRunResultType]):
    """
    Abstract base class for converting Eka circuits and experiments to a specific
    format.

    This class defines the structure for converters that can transform Eka operations
    into a target language or format.
    Subclasses must implement the abstract methods to provide the actual conversion
    logic.
    And functions to evaluate the output format in terms of Eka language.

    Attributes
    ----------
    SUPPORTED_OPERATIONS : list[OpSignature]
        A set of all supported quantum operations.

    ALLOW_ERROR_MODELS: bool
        Flag indicating whether the converter allows error models to be passed during
        conversion.

    Properties
    ----------
    operations_map : dict[OpSignature, OpAndTargetToInstrCallable]
        A mapping of the supported operations to their corresponding implementation
        functions in the target language.
    eka_op_by_name : dict[str, OpSignature]
        A mapping of operation names to corresponding OpSignature objects.

    Methods
    -------
    convert(interpreted_eka: InterpretationStep) -> Any
        Convert the input interpretation step to a specific format.
    convert_circuit(input_circuit: Circuit) -> Any
        Convert a Circuit to a specific format.
    parse_target_run_outcome(run_output: TargetRunResultType)-> dict[str, list[int]]
        Parse the run output of the target language into a dictionary mapping the
        Eka channel labels to integer values measured at each shot.
    emit_circuit_program(
        input_circuit: Circuit,
        q_register: dict[str, Any],
        c_register: dict[str, Any],
    ) -> str
        Emit the full program string for a given circuit, given some quantum and
        classical register mappings.
    emit_init_instructions(
        circuit: Circuit
    ) -> tuple[str, dict[str, Any], dict[str, Any]]
        Provide the code to initialize the quantum and classical registers, and return
        the mapping from Eka channel id to register.
    emit_leaf_circuit_instruction(
        circuit: Circuit,
        quantum_channel_map: dict[str, Any],
        classical_channel_map: dict[str, Any],
    ) -> str
        Provide the code to emit an Eka instruction in the target language.

    Raises
    ------
    TypeError
        If the mapping is not a dict.
    ValueError
        If the mapping is missing any of the required keys.
    """

    model_config = {"frozen": True}  # makes instances immutable

    SUPPORTED_OPERATIONS: frozenset[OpSignature]
    ALLOW_ERROR_MODELS: bool = False

    # Separator string for else conditions in if-else constructs
    separator_for_else_in_condition: str = Field(
        default=", is_else=",
        frozen=True,
        init=False,
        description="The separator string used in the description for else conditions.",
    )

    @model_validator(mode="after")
    def _validate_ops(self):
        """
        Validate that the converter supports all the required quantum operations
        specified.

        This validator checks:

        1. operations_map builds successfully
        2. The operations_map covers all supported operations

        Returns
        -------
        self
            The validated instance

        Raises
        ------
        ValueError
            If validation fails for any of the above conditions
        """
        # 1. Check that operations_map builds successfully
        try:
            ops_map = self.operations_map
        except Exception as e:
            raise ValueError(f"Failed to build operations_map: {e}") from e

        # 2. Check that all supported operations are present in operations_map
        missing_ops = [
            op.name for op in self.SUPPORTED_OPERATIONS if op.name not in ops_map
        ]
        if missing_ops:
            raise ValueError(
                "operations_map is missing implementations for "
                f"operations: {missing_ops}"
            )

        return self

    @property
    @abstractmethod
    def operations_map(self) -> dict[str, OpAndTargetToInstrCallable]:
        """Returns a mapping of the supported operations to their corresponding
        instructions in the target language."""
        raise NotImplementedError(
            "Subclasses must implement the operations_map property."
        )

    @cached_property
    def op_by_eka_name(self) -> dict[str, OpSignature]:
        """Map of operation names to corresponding OpSignature objects."""
        return {op.name: op for op in ALL_EKA_OP_SIGNATURES}

    def convert(self, interpreted_eka: InterpretationStep) -> TargetType:
        """Convert a InterpretationStep. By default, it converts the final circuit of
        the step. This can differ from convert_circuit for converters that support
        detector and observable objects (like Stim).
        """
        return self.convert_circuit(interpreted_eka.final_circuit)

    @abstractmethod
    def convert_circuit(self, input_circuit: Circuit) -> TargetType:
        """Convert a Circuit into a program in the target language."""

    def _validate_ops_args(
        self, op_name: str, num_q_target: int, num_c_target: int
    ) -> None:
        """
        Validate the arguments for the operation.
        This validation checks that the number of quantum and classical targets
        provided for the operation matches the expected numbers defined in the
        OpSignature.

        For the BOOL_LOGIC OpType (used for condition in classical control flow),
        it allows for any positive number of classical targets greater than or equal to
        the required number, except for mono-bit boolean operations (Not and Match) that
        allows only 1 classical channel.

        The UTILS OpType operations are always valid by default.

        Parameters
        ----------
        op_name : str
            The name of the operation to validate.
        num_q_target : int
            The number of quantum targets provided for the operation.
        num_c_target : int
            The number of classical targets provided for the operation.

        Raises
        ------
        TypeError
            If num_q_target or num_c_target is not an integer.
        ValueError
            If the operation is unsupported or if the number of targets does not match.
        """
        if not isinstance(num_q_target, int):
            raise TypeError(f"{op_name} quantum target must be an integer")
        if not isinstance(num_c_target, int):
            raise TypeError(f"{op_name} classical target must be an integer")

        if op_name not in self.op_by_eka_name:
            raise ValueError(f"Unsupported operation '{op_name}'")

        eka_sig = self.op_by_eka_name[op_name]
        if eka_sig.quantum_input != num_q_target:
            raise ValueError(
                f"{op_name} quantum target must have "
                f"{eka_sig.quantum_input} qubits, "
                f"but got {num_q_target}."
            )
        if eka_sig.classical_input != num_c_target:
            if eka_sig.op_type == OpType.UTILS:
                # pass utils operations, used for comments, etc.
                return
            if eka_sig.op_type == OpType.BOOL_LOGIC:
                if (
                    num_c_target <= eka_sig.classical_input
                    and eka_sig.classical_input != 1
                ):
                    # Special case: allow n classical bits for boolean logic ops as long
                    # as n >= required bits (and need at least 2 bits)
                    # And this doesn't apply to mono-bit boolean ops (Not and Match)
                    raise ValueError(
                        f"{op_name} must have at least {eka_sig.classical_input}"
                        " classical bits"
                    )
                return
            raise ValueError(
                f"{op_name} classical target must have "
                f"{eka_sig.classical_input} bits, "
                f"but got {num_c_target}."
            )

    @staticmethod
    def _validate_import_prefix(import_prefix: str) -> None:
        """
        Validate the import prefix string.
        It must be a string that ends with a dot (.)
        or be an empty string.

        Parameters
        ----------
        import_prefix : str
            The import prefix string to validate.

        Raises
        ------
        TypeError
            If import_prefix is not a string.
        ValueError
            If import_prefix is not empty and does not end with a dot (.).
        """
        if not isinstance(import_prefix, str):
            raise TypeError("import_prefix must be a string")
        if import_prefix == "":
            return
        if not import_prefix.endswith("."):
            raise ValueError(f"import_prefix '{import_prefix}' must end with a dot (.)")

    @staticmethod
    @abstractmethod
    def parse_target_run_outcome(
        run_output: TargetRunResultType,
    ) -> dict[str, int | list[int]]:
        """Parse the run output of the target language into a dictionary mapping the
        eka channel labels to values measured at each shot."""

    # pylint: disable=too-many-branches
    def emit_circuit_program(
        self,
        input_circuit: Circuit,
        q_register: dict[str, str],
        c_register: dict[str, str | None],
    ) -> str:
        """Emit the full program string for a given circuit, given some quantum and
        classical register mappings.

        This method process the circuit in a depth-first manner, handling nested
        circuits and control flow constructs (if-else) appropriately. It uses a stack
        to keep track of circuits to process, and maintains an indentation level for
        formatting the output.

        Parameters
        ----------
        input_circuit : Circuit
            The input circuit to emit the program for.
        q_register : dict[str, str]
            The mapping from eka quantum channel ids to target quantum register names.
        c_register : dict[str, str | None]
            The mapping from eka classical channel ids to target classical register
            names.

        Returns
        -------
        str
            The emitted program as a string.
        """
        if not isinstance(input_circuit, Circuit):
            raise TypeError("Input must be a Circuit instance.")

        program_lines = []

        stack_to_process = deque([input_circuit])
        indent_level = 0
        indent = "    "

        while stack_to_process:
            processed_circuit = stack_to_process.popleft()
            if processed_circuit.name == "empty_circuit":
                continue

            if processed_circuit.name in self.op_by_eka_name:
                processed_op = self.op_by_eka_name[processed_circuit.name]

                if processed_op.op_type == OpType.UTILS:
                    match processed_op.name:
                        case "indent_more":
                            indent_level += 1
                        case "indent_less":
                            indent_level = max(0, indent_level - 1)
                        case "comment":
                            comment_str = self.emit_leaf_circuit_instruction(
                                processed_circuit, {}, {}
                            )
                            program_lines.append(
                                f"{indent * indent_level}{comment_str}"
                            )
                else:
                    # Leaf operation: just append op strings
                    op_str = self.emit_leaf_circuit_instruction(
                        processed_circuit, q_register, c_register
                    )
                    for line in op_str.splitlines():
                        program_lines.append(f"{indent * indent_level}{line}")

            elif hasattr(processed_circuit, "_loom_ifelse_marker"):
                condition_instruction = self.emit_leaf_circuit_instruction(
                    processed_circuit.condition_circuit,
                    quantum_channel_map=q_register,
                    classical_channel_map=c_register,
                )
                is_else_present = False
                if_else_instruction = [
                    (Circuit(name="classical_if", description=condition_instruction)),
                    (Circuit(name="indent_more")),
                    (processed_circuit.if_circuit),
                ]
                if processed_circuit.else_circuit.name != "empty_branch":
                    is_else_present = True
                    if_else_instruction += [
                        (Circuit(name="indent_less")),
                        (
                            Circuit(
                                name="classical_else", description=condition_instruction
                            )
                        ),
                        (Circuit(name="indent_more")),
                        (processed_circuit.else_circuit),
                    ]
                if_else_instruction += [
                    (Circuit(name="indent_less")),
                    (
                        Circuit(
                            name="end_if",
                            description=condition_instruction
                            + self.separator_for_else_in_condition
                            + str(is_else_present),
                        )
                    ),
                ]
                for line in reversed(if_else_instruction):
                    stack_to_process.appendleft(line)

            else:  # Nested circuit
                nested_instructions = [
                    c for tick in processed_circuit.circuit for c in tick
                ]
                if not processed_circuit.name.startswith("wrapped_"):
                    start_comment = [
                        Circuit(
                            name="comment",
                            description=f"Start of circuit: {processed_circuit.name}",
                        )
                    ]
                    nested_instructions = start_comment + nested_instructions
                    nested_instructions += [
                        Circuit(
                            name="comment",
                            description=f" End of circuit: {processed_circuit.name}",
                        )
                    ]
                for line in reversed(nested_instructions):
                    stack_to_process.appendleft(line)

        return "\n".join(program_lines)

    @abstractmethod
    def emit_init_instructions(
        self, circuit: Circuit
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Provide the python code (as a string) to initializes the
        quantum and classical registers, and return the mapping from eka channel id to
        register."""

    @abstractmethod
    def emit_leaf_circuit_instruction(
        self,
        circuit: Circuit,
        quantum_channel_map: dict[str, Any],
        classical_channel_map: dict[str, Any],
    ) -> str:
        """Provide the python code (as a string) to emit an Eka instruction in the
        target language."""
