"""
Pytest Fixtures for Quantum Circuit Testing
===========================================

This module provides a set of pytest fixtures and helper functions
to make testing in Executor easier.

Key points of pytest fixtures:

- Fixtures are reusable setup functions that prepare data or objects
  for your tests.
- `input_fixture` allows tests to be parametrized using fixture names,
  so you can run the same test with different circuits easily.
- `load_expected_data` loads JSON files containing expected results
  for comparison in tests.
- Helper functions like `get_n_channels` simplify creating channels
  for circuits.
- **circuit fixtures**, which are pre-built
  quantum circuits (empty, simple, Bell state, nested, etc.).
- **InterpretationStep fixtures** for testing conversion that involves detectors
- **CircuitErrorModel fixtures** for testing error models with various configurations.

"""

from collections import defaultdict
from functools import partial
import json
import math
from pathlib import Path
from pydantic import Field, field_validator, model_validator
import pytest

from loom.eka import Circuit, Channel, ChannelType, Lattice, Eka
from loom.eka.ifelse_circuit import IfElseCircuit
from loom.eka.operations import MeasureLogicalZ, MeasureBlockSyndromes
from loom.eka.utilities.enums import BoolOp
from loom.executor.op_signature import (
    CLIFFORD_GATES_SIGNATURE,
)
from loom.interpreter import interpret_eka
from loom.executor.circuit_error_model import (
    ApplicationMode,
    CircuitErrorModel,
    ErrorProbProtocol,
    ErrorType,
    HomogeneousTimeDependentCEM,
    HomogeneousTimeIndependentCEM,
)

# ----------------
# Helper Functions
# ----------------


def get_n_channels(
    n: int, prefix: str = None, chan_type: ChannelType = ChannelType.QUANTUM
) -> list[Channel]:
    """Helper function to create a list of n channels."""
    if prefix is None:
        if chan_type == ChannelType.CLASSICAL:
            prefix = "c"
        else:
            prefix = "q"
    return [Channel(label=f"{prefix}{i}", type=chan_type) for i in range(n)]


# ------------------------------
# Circuit Fixtures
#
# Provides a circuit and it's expected supported operations.
# ------------------------------


@pytest.fixture(scope="class")
def load_expected_data(request):
    """Load expected data JSON for the current test class."""
    filename = getattr(request, "param", None)
    if filename is None:
        pytest.fail(
            "You must pass a JSON file name as a parameter, "
            "to be used to load expected data."
        )

    filepath = (
        Path(request.fspath).parent / "test_data" / "expec_fixtures_outputs" / filename
    )

    if not filepath.exists():
        pytest.fail(
            f"No expected_json defined for this test class: {request.node.cls.__name__}"
        )

    with filepath.open() as f:
        return json.load(f)


@pytest.fixture
def input_fixture(request):
    """This is used to resolve the input fixture by name.
    So that we can use it in the to use a parametrize list of fixtures."""
    # request.param is the string name of the actual fixture
    return request.getfixturevalue(request.param), request.param


@pytest.fixture
def empty_circuit():
    """Fixture to provide an empty circuit."""
    return Circuit(name="empty_circuit", circuit=[], channels=[])


@pytest.fixture
def simple_circuit():
    """
    Fixture to provide a simple circuit with a reset, an X gate, and a measurement.
    """
    q_chan = get_n_channels(1)
    c_chan = get_n_channels(1, chan_type=ChannelType.CLASSICAL)
    return Circuit(
        name="simple_circuit",
        circuit=[
            Circuit(name="reset", channels=[q_chan[0]]),
            Circuit(name="x", channels=[q_chan[0]]),
            Circuit(name="measurement", channels=[q_chan[0], c_chan[0]]),
        ],
    )


@pytest.fixture
def bell_state_circuit():
    """Fixture to provide a circuit that creates a Bell state."""
    q_chans = get_n_channels(2)
    c_chans = get_n_channels(2, chan_type=ChannelType.CLASSICAL)
    return Circuit(
        name="bell_state_circuit",
        circuit=[
            Circuit(name="h", channels=[q_chans[0]]),
            Circuit(name="cnot", channels=[q_chans[0], q_chans[1]]),
            Circuit(name="measurement", channels=[q_chans[0], c_chans[0]]),
            Circuit(name="measurement", channels=[q_chans[1], c_chans[1]]),
        ],
    )


@pytest.fixture
def circuit_with_nested_circuits():
    """Fixture to provide a circuit with nested circuits."""
    q_chans = get_n_channels(3)
    c_chans = get_n_channels(3, chan_type=ChannelType.CLASSICAL)
    return Circuit(
        name="nested_circuit",
        circuit=[
            [
                Circuit(name="reset", channels=[q_chans[0]]),
                Circuit(name="h", channels=[q_chans[1]]),
            ],
            [
                Circuit(name="cnot", channels=[q_chans[0], q_chans[1]]),
            ],
            [
                Circuit(name="h", channels=[q_chans[0]]),
                Circuit(name="cnot", channels=[q_chans[1], q_chans[2]]),
            ],
            [
                Circuit(name="measurement", channels=[q_chans[0], c_chans[0]]),
                Circuit(name="measurement", channels=[q_chans[1], c_chans[1]]),
                Circuit(name="measurement", channels=[q_chans[2], c_chans[2]]),
            ],
        ],
    )


@pytest.fixture
def circuit_with_multiple_nested_levels():
    """Fixture to provide a circuit with multiple nested levels."""
    q_chans = get_n_channels(4)
    c_chans = get_n_channels(2, chan_type=ChannelType.CLASSICAL)
    return Circuit(
        name="multi_nested_circuit",
        circuit=Circuit.construct_padded_circuit_time_sequence(
            [
                [
                    Circuit(name="reset", channels=[q_chans[0]]),
                    Circuit(name="h", channels=[q_chans[1]]),
                ],
                [
                    Circuit(name="cnot", channels=[q_chans[0], q_chans[3]]),
                    Circuit(name="cnot", channels=[q_chans[1], q_chans[2]]),
                ],
                [
                    Circuit(name="h", channels=[q_chans[0]]),
                    Circuit(
                        name="sub_nested",
                        circuit=[
                            [
                                Circuit(name="cnot", channels=[q_chans[2], q_chans[3]]),
                                Circuit(name="reset", channels=[q_chans[1]]),
                            ],
                        ],
                    ),
                ],
                [
                    Circuit(name="measurement", channels=[q_chans[0], c_chans[0]]),
                    Circuit(name="measurement", channels=[q_chans[1], c_chans[1]]),
                ],
            ]
        ),
    )


@pytest.fixture
def circuit_w_all_single_qubit_ops():
    """Fixture to provide a circuit with all single qubit operations."""
    q_chans = get_n_channels(1)
    return Circuit(
        name="circuit_w_all_single_qubit_ops",
        circuit=[
            Circuit(name="i", channels=[q_chans[0]]),
            Circuit(name="x", channels=[q_chans[0]]),
            Circuit(name="y", channels=[q_chans[0]]),
            Circuit(name="z", channels=[q_chans[0]]),
            Circuit(name="h", channels=[q_chans[0]]),
            Circuit(name="phase", channels=[q_chans[0]]),
            Circuit(name="phaseinv", channels=[q_chans[0]]),
        ],
    )


@pytest.fixture
def circuit_w_all_two_qubit_ops():
    """Fixture to provide a circuit with all two qubit operations."""
    q_chans = get_n_channels(2)
    return Circuit(
        name="circuit_w_all_two_qubit_ops",
        circuit=[
            Circuit(name="cnot", channels=[q_chans[0], q_chans[1]]),
            Circuit(name="cx", channels=[q_chans[0], q_chans[1]]),
            Circuit(name="cy", channels=[q_chans[0], q_chans[1]]),
            Circuit(name="cz", channels=[q_chans[0], q_chans[1]]),
            Circuit(name="swap", channels=[q_chans[0], q_chans[1]]),
        ],
    )


@pytest.fixture
def circuit_w_all_reset_ops():
    """Fixture to provide a circuit with all reset operations."""
    q_chans = get_n_channels(1)
    return Circuit(
        name="circuit_w_all_reset_ops",
        circuit=[
            Circuit(name="reset", channels=[q_chans[0]]),
            Circuit(name="reset_0", channels=[q_chans[0]]),
            Circuit(name="reset_1", channels=[q_chans[0]]),
            Circuit(name="reset_+", channels=[q_chans[0]]),
            Circuit(name="reset_-", channels=[q_chans[0]]),
            Circuit(name="reset_+i", channels=[q_chans[0]]),
            Circuit(name="reset_-i", channels=[q_chans[0]]),
        ],
    )


@pytest.fixture
def circuit_w_all_measurement_ops():
    """Fixture to provide a circuit with all measurement operations."""
    q_chans = get_n_channels(1)
    c_chans = get_n_channels(1, chan_type=ChannelType.CLASSICAL)
    return Circuit(
        name="circuit_w_all_measurement_ops",
        circuit=[
            Circuit(name="measurement", channels=[q_chans[0], c_chans[0]]),
            Circuit(name="reset", channels=[q_chans[0]]),
            Circuit(name="measure_z", channels=[q_chans[0], c_chans[0]]),
            Circuit(name="reset", channels=[q_chans[0]]),
            Circuit(name="measure_x", channels=[q_chans[0], c_chans[0]]),
            Circuit(name="reset", channels=[q_chans[0]]),
            Circuit(name="measure_y", channels=[q_chans[0], c_chans[0]]),
        ],
    )


@pytest.fixture
def circuit_with_simple_if_else():
    """Fixture to provide a circuit with if-else structure."""
    q_chans = get_n_channels(2)
    c_chans = get_n_channels(2, chan_type=ChannelType.CLASSICAL)
    return Circuit(
        name="circuit_with_if_else",
        circuit=[
            Circuit(name="h", channels=[q_chans[0]]),
            Circuit(name="measurement", channels=[q_chans[0], c_chans[0]]),
            IfElseCircuit(
                condition_circuit=Circuit(name=BoolOp.MATCH, channels=[c_chans[0]]),
                if_circuit=Circuit(name="x", channels=[q_chans[1]]),
                else_circuit=Circuit(name="z", channels=[q_chans[1]]),
            ),
            Circuit(name="measurement", channels=[q_chans[1], c_chans[1]]),
        ],
    )


@pytest.fixture
def circuit_if_with_empty_else():
    """Fixture to provide a circuit with if structure with empty else."""
    q_chans = get_n_channels(2)
    c_chans = get_n_channels(2, chan_type=ChannelType.CLASSICAL)
    return Circuit(
        name="circuit_with_if_empty_else",
        circuit=[
            Circuit(name="h", channels=[q_chans[0]]),
            Circuit(name="measurement", channels=[q_chans[0], c_chans[0]]),
            IfElseCircuit(
                condition_circuit=Circuit(name=BoolOp.MATCH, channels=[c_chans[0]]),
                if_circuit=Circuit(name="x", channels=[q_chans[1]]),
            ),
            Circuit(name="measurement", channels=[q_chans[1], c_chans[1]]),
        ],
    )


@pytest.fixture
def circuit_with_boolean_logic_condition():
    """Fixture to provide a circuit with if-else structure with boolean logic
    in the condition."""
    q_chans = get_n_channels(3)
    c_chans = get_n_channels(3, chan_type=ChannelType.CLASSICAL)
    return Circuit(
        name="circuit_with_advanced_condition",
        circuit=[
            Circuit(name="h", channels=[q_chans[0]]),
            Circuit(name="x", channels=[q_chans[1]]),
            Circuit(name="measurement", channels=[q_chans[0], c_chans[0]]),
            Circuit(name="measurement", channels=[q_chans[1], c_chans[1]]),
        ]
        + [
            IfElseCircuit(
                condition_circuit=Circuit(
                    name=bool_op,
                    channels=[c_chans[0], c_chans[1]],
                ),
                if_circuit=Circuit(name="x", channels=[q_chans[2]]),
                else_circuit=Circuit(name="z", channels=[q_chans[2]]),
            )
            for bool_op in BoolOp.multi_bit_list()
        ]
        + [
            IfElseCircuit(
                condition_circuit=Circuit(
                    name=bool_op,
                    channels=[c_chans[0]],
                ),
                if_circuit=Circuit(name="x", channels=[q_chans[2]]),
                else_circuit=Circuit(name="z", channels=[q_chans[2]]),
            )
            for bool_op in BoolOp.mono_bit_list()
        ]
        + [
            Circuit(name="measurement", channels=[q_chans[2], c_chans[2]]),
        ],
    )


@pytest.fixture
def circuit_with_boolean_logic_condition_multibit():
    """Fixture to provide a circuit with if-else structure with boolean logic
    in the condition with multiple bits."""
    q_chans = get_n_channels(3)
    c_chans = get_n_channels(6, chan_type=ChannelType.CLASSICAL)
    return Circuit(
        name="circuit_with_advanced_condition_multibit",
        circuit=[
            Circuit(name="h", channels=[q_chans[0]]),
            Circuit(name="x", channels=[q_chans[1]]),
            Circuit(name="measurement", channels=[q_chans[0], c_chans[0]]),
            Circuit(name="measurement", channels=[q_chans[0], c_chans[1]]),
            Circuit(name="measurement", channels=[q_chans[0], c_chans[2]]),
            Circuit(name="measurement", channels=[q_chans[0], c_chans[3]]),
            Circuit(name="measurement", channels=[q_chans[0], c_chans[4]]),
            IfElseCircuit(
                condition_circuit=Circuit(
                    name=BoolOp.NAND,
                    channels=[
                        c_chans[0],
                        c_chans[1],
                        c_chans[2],
                        c_chans[3],
                        c_chans[4],
                    ],
                ),
                if_circuit=Circuit(name="x", channels=[q_chans[2]]),
                else_circuit=Circuit(name="z", channels=[q_chans[2]]),
            ),
            Circuit(name="measurement", channels=[q_chans[2], c_chans[5]]),
        ],
    )


@pytest.fixture
def circuit_w_nested_if_else():
    """Fixture to provide a circuit with nested if-else structures."""
    q_chans = get_n_channels(2)
    c_chans = get_n_channels(3, chan_type=ChannelType.CLASSICAL)
    circuit = Circuit(
        name="circuit_with_nested_if_else",
        circuit=[
            Circuit(name="h", channels=[q_chans[0]]),
            Circuit(name="measurement", channels=[q_chans[0], c_chans[0]]),
            Circuit(name="measurement", channels=[q_chans[1], c_chans[1]]),
            IfElseCircuit(
                condition_circuit=Circuit(name=BoolOp.MATCH, channels=[c_chans[0]]),
                if_circuit=IfElseCircuit(
                    condition_circuit=Circuit(name=BoolOp.MATCH, channels=[c_chans[1]]),
                    if_circuit=Circuit(name="x", channels=[q_chans[1]]),
                    else_circuit=Circuit(name="y", channels=[q_chans[1]]),
                ),
                else_circuit=Circuit(name="z", channels=[q_chans[1]]),
            ),
            Circuit(name="measurement", channels=[q_chans[1], c_chans[2]]),
        ],
    )
    return circuit


@pytest.fixture
def circuit_surface_code_experiment(n_rsc_block_factory):
    """Fixture to provide a circuit that creates a Bell state."""
    n_cycles = 3
    lattice = Lattice.square_2d((6, 4))
    block = n_rsc_block_factory(1)[0]
    operations = [
        MeasureBlockSyndromes(input_block_name=block.unique_label, n_cycles=n_cycles),
        MeasureLogicalZ(input_block_name=block.unique_label),
    ]
    lscrd = Eka(lattice, blocks=[block], operations=operations)
    final_step = interpret_eka(lscrd)

    return final_step.final_circuit


# -----------------------------------------------
# CircuitErrorModel Fixtures
#
# Provides a circuit error model class that only needs a circuit to instantiate.
# One can use the fixture `cem_instance` to get an instance of CircuitErrorModel given
# a circuit fixture name and a cem class fixture name.
# -----------------------------------------------
@pytest.fixture
def empty_error_model():
    """Simple CircuitErrorModel with a few gate errors."""
    return partial(
        CircuitErrorModel,
        error_type=ErrorType.PAULI_X,
        is_time_dependent=False,
        application_mode=ApplicationMode.AFTER_GATE,
        gate_durations=None,
        gate_error_probabilities=defaultdict(lambda _: [0.0]),
        global_time_error_probability=None,
    )


@pytest.fixture
def homogeneous_time_independent_cem():
    """Simple HomogeneousTimeIndependentCEM with a few gate errors."""
    return partial(
        HomogeneousTimeIndependentCEM,
        error_type=ErrorType.PAULI_X,
        application_mode=ApplicationMode.AFTER_GATE,
        error_probability=0.01,
        target_gates=["x", "cnot", "cx", "measurement"],
    )


@pytest.fixture
def homogeneous_time_dependent_cem():
    """Simple HomogeneousTimeDependentCEM with a few gate errors."""
    return partial(
        HomogeneousTimeDependentCEM,
        error_type=ErrorType.PAULI_X,
        application_mode=ApplicationMode.AFTER_GATE,
        error_probability=lambda t: t * 0.01,
        target_gates=["x", "cnot", "cx", "measurement"],
    )


@pytest.fixture
def all_error_types_cem():
    """Generate a list of CEMs with all error types."""
    cems = []
    for e_type in ErrorType:
        gate_error_probabilities = {
            "x": lambda _, pc=e_type.param_count: [0.01] * pc,
            "cnot": lambda _, pc=e_type.param_count: [0.02] * pc,
            "cx": lambda _, pc=e_type.param_count: [0.02] * pc,
            "measurement": lambda _, pc=e_type.param_count: [0.03] * pc,
        }
        if e_type == ErrorType.DEPOLARIZING2:
            # Depolarizing2 only works with 2 qubit gates, so we need all other gates
            # to have 0 error probability.
            gate_error_probabilities = {
                "cnot": lambda _, pc=e_type.param_count: [0.02] * pc,
                "cx": lambda _, pc=e_type.param_count: [0.02] * pc,
            }
        cem = partial(
            CircuitErrorModel,
            error_type=e_type,
            is_time_dependent=False,
            application_mode=ApplicationMode.AFTER_GATE,
            gate_durations=None,
            gate_error_probabilities=gate_error_probabilities,
            global_time_error_probability=None,
        )
        cems.append(cem)

    return lambda circuit: [cem(circuit=circuit) for cem in cems]


@pytest.fixture
def all_application_modes_cem():
    """Generate a list of CEMs with all application modes."""
    cems = []
    gate_set = {gate.name for gate in CLIFFORD_GATES_SIGNATURE}
    gate_set.remove("cnot")
    gate_set.add("cx")  # include cx as alias for cnot
    for app_mode in ApplicationMode:
        cem = partial(
            CircuitErrorModel,
            error_type=ErrorType.PAULI_X,
            is_time_dependent=True,
            application_mode=app_mode,
            gate_durations={gate: 0.5 for gate in gate_set} | {"cnot": 2, "cx": 2},
            gate_error_probabilities={
                "x": lambda _: [0.01],
                "cnot": lambda _: [0.02],
                "cx": lambda _: [0.02],
                "measurement": lambda _: [0.03],
            },
            global_time_error_probability=lambda t, t2: [
                t * 0.001 + t2 * 0.002,
            ],
        )
        cems.append(cem)

    return lambda circuit: [cem(circuit=circuit) for cem in cems]


@pytest.fixture
def complex_cem():
    """A more complex CEM with time-dependent errors and global time errors."""
    gate_durations = (
        {g: 2 for g in ["cnot", "cx", "cy", "cz", "swap"]}
        | {
            g: 1
            for g in [
                "x",
                "y",
                "z",
                "h",
                "phase",
                "phaseinv",
                "i",
            ]
        }
        | {
            g: 0.5
            for g in [
                "reset",
                "reset_0",
                "reset_1",
                "reset_+",
                "reset_-",
                "reset_+i",
                "reset_-i",
                "measurement",
            ]
        }
    )

    class TwoQubitsSymmetricDepolarisingChannel(HomogeneousTimeIndependentCEM):
        """Two qubits depolarizing channel error model."""

        error_probability: float
        application_mode: ApplicationMode = ApplicationMode.AFTER_GATE
        error_type: ErrorType = ErrorType.DEPOLARIZING2
        target_gates: list[str] = [
            "cnot",
            "cx",
            "cy",
            "cz",
            "swap",
        ]

    class SingleQubitSymmetricDepolarisingChannel(HomogeneousTimeIndependentCEM):
        """Single qubit depolarizing channel error model."""

        error_probability: float
        application_mode: ApplicationMode = ApplicationMode.AFTER_GATE
        error_type: ErrorType = ErrorType.DEPOLARIZING1
        target_gates: list[str] = [
            "x",
            "y",
            "z",
            "h",
            "phase",
            "phaseinv",
        ]

    class AsymmetricIdleNoise(CircuitErrorModel):
        """Asymmetric idle noise model with T1 and T2 time constants."""

        t1: float
        t2: float
        is_time_dependent: bool = Field(default=True, init=False)
        application_mode: ApplicationMode = Field(
            default=ApplicationMode.IDLE_END_OF_TICK, init=False
        )
        error_type: ErrorType = Field(default=ErrorType.PAULI_CHANNEL, init=False)
        global_time_error_probability: ErrorProbProtocol = Field(
            default=lambda _, t: [0, 0, 0], init=False
        )

        @model_validator(mode="after")
        # pylint: disable=no-self-argument
        def validate_time_constants(cls, model) -> float:
            """t2 must be smaller or equal 2*t1."""
            if model.t2 > 2 * model.t1:
                raise ValueError("t2 must be smaller or equal than 2*t1.")
            return model

        @field_validator("t1", "t2")
        @classmethod
        def check_positive(cls, v, info):
            """Ensure time constants are non-negative."""
            if v < 0:
                raise ValueError(f"{info.field_name} must be non-negative.")
            return v

        def model_post_init(self, __context):
            object.__setattr__(
                self,
                "global_time_error_probability",
                lambda _, t: self._p(t),
            )
            super().model_post_init(__context)

        def _p(self, t: float) -> list[float]:
            """
            Internal method to compute error probabilities given time t.
            Using the following model:
            p_x = p_y = (1 - exp(-t / t1)) / 4
            p_z = (1 - exp(-t / t2)) / 2 - p_x

            Parameters:
            ----------
            t : float
                The time used to compute the error probabilities.

            Returns:
            -------
                List[float]: [p_x, p_y, p_z]
            """
            if self.t1 == 0:
                exp_t_t1 = 0
            else:
                exp_t_t1 = math.exp(-t / self.t1)
            p_x = p_y = (1 - exp_t_t1) / 4
            if self.t2 == 0:
                exp_t_t2 = 0
            else:
                exp_t_t2 = math.exp(-t / self.t2)
            p_z = (1 - exp_t_t2) / 2 - p_x
            return [p_x, p_y, p_z]

    class FlipBeforeMeasurement(HomogeneousTimeIndependentCEM):
        """Flip error model applied before measurement gates."""

        error_probability: float
        application_mode: ApplicationMode = ApplicationMode.BEFORE_GATE
        error_type: ErrorType = ErrorType.PAULI_X
        target_gates: list[str] = ["measurement", "measure_z", "measure_x", "measure_y"]

    class FlipAfterReset(HomogeneousTimeIndependentCEM):
        """Flip error model applied after reset gates."""

        error_probability: float
        application_mode: ApplicationMode = ApplicationMode.AFTER_GATE
        error_type: ErrorType = ErrorType.PAULI_X
        target_gates: list[str] = [
            "reset",
            "reset_0",
            "reset_1",
            "reset_+",
            "reset_-",
            "reset_+i",
            "reset_-i",
        ]

    return lambda circuit: [
        AsymmetricIdleNoise(circuit=circuit, t1=3, t2=2, gate_durations=gate_durations),
        SingleQubitSymmetricDepolarisingChannel(
            circuit=circuit, error_probability=0.01
        ),
        TwoQubitsSymmetricDepolarisingChannel(circuit=circuit, error_probability=0.02),
        FlipAfterReset(circuit=circuit, error_probability=0.005),
        FlipBeforeMeasurement(circuit=circuit, error_probability=0.005),
    ]


@pytest.fixture
def cem_instance(request):
    """
    Generate a CircuitErrorModel dynamically,
    given a circuit fixture, and a CEM fixture.
    """
    params = request.param
    circuit_fixture_name = params["circuit_fixture"]
    cem_class_fixture_name = params["cem_class_fixture"]

    # Dynamically get the circuit and CEM class
    circuit = request.getfixturevalue(circuit_fixture_name)
    cem_class = request.getfixturevalue(cem_class_fixture_name)

    # Instantiate the CEM class with circuit and additional kwargs
    cem_instanc_i = cem_class(circuit=circuit)
    return cem_instanc_i, circuit, circuit_fixture_name + "_" + cem_class_fixture_name


# ------------------------------
# InterpretationStep Fixtures
#
# Provides a interpretation step
# ------------------------------


@pytest.fixture
def interpreted_empty_lscrd():
    """Fixture to provide an interpreted empty circuit."""
    lscrd = Eka(lattice=Lattice.square_2d((6, 4)), blocks=[], operations=[])
    step = interpret_eka(lscrd)

    return step


@pytest.fixture
def rsc_memory_experiment(n_rsc_block_factory):
    """Fixture to provide an interpreted rotated surface code memory experiment."""
    n_cycles = 3
    lattice = Lattice.square_2d((6, 4))
    block = n_rsc_block_factory(1)[0]
    operations = [
        MeasureBlockSyndromes(input_block_name=block.unique_label, n_cycles=n_cycles),
        MeasureLogicalZ(input_block_name=block.unique_label),
    ]
    lscrd = Eka(lattice, blocks=[block], operations=operations)
    step = interpret_eka(lscrd)

    return step


@pytest.fixture
def rsc_two_block_experiment(n_rsc_block_factory):
    """Fixture to provide an interpreted rotated surface code two block experiment."""
    [block1, block2] = n_rsc_block_factory(2)
    n_cycles = 3
    lattice = Lattice.square_2d((20, 20))
    operations = [
        MeasureBlockSyndromes(input_block_name=block1.unique_label, n_cycles=n_cycles),
        MeasureBlockSyndromes(input_block_name=block2.unique_label, n_cycles=n_cycles),
        MeasureLogicalZ(
            input_block_name=block1.unique_label,
        ),
        MeasureLogicalZ(input_block_name=block2.unique_label),
    ]
    lscrd = Eka(lattice, blocks=[block1, block2], operations=operations)
    step = interpret_eka(lscrd)

    return step


@pytest.fixture
# pylint: disable=redefined-outer-name
def circuit_to_init_and_instructions_to_append(circuit_surface_code_experiment):
    """Fixture that provides a circuit to initialize the converter
    and a dict of instructions to append to it. This is used to test the emit functions
    of the converters
    """
    circuit = circuit_surface_code_experiment

    q_chans = sorted(
        [qc for qc in circuit.channels if qc.is_quantum()],
        key=lambda c: c.label,
    )
    c_chans = sorted(
        [cc for cc in circuit.channels if cc.is_classical()],
        key=lambda c: c.label,
    )

    instruction_to_append = {}

    instruction_to_append["non_existing_op"] = Circuit(
        name="non_existing_op", channels=[q_chans[0]], circuit=[]
    )
    instruction_to_append["non_atomic_op"] = Circuit(
        name="non_atomic_op",
        channels=[q_chans[0], q_chans[1]],
        circuit=[
            Circuit(name="h", channels=[q_chans[0]]),
            Circuit(name="cnot", channels=[q_chans[0], q_chans[1]]),
        ],
    )
    # instruction with wrong number of channels
    instruction_to_append["wrong_channel_count"] = Circuit(
        name="reset", channels=[c_chans[0]]
    )
    instruction_to_append["single_qb_gate"] = Circuit(
        name="h", channels=[q_chans[0]], circuit=[]
    )
    instruction_to_append["two_qb_gate"] = Circuit(
        name="cnot", channels=[q_chans[0], q_chans[1]], circuit=[]
    )
    instruction_to_append["measurement_gate"] = Circuit(
        name="measurement", channels=[q_chans[0], c_chans[0]], circuit=[]
    )
    instruction_to_append["reset_gate"] = Circuit(
        name="reset", channels=[q_chans[0]], circuit=[]
    )
    instruction_to_append["instruction_w_multiple_steps"] = Circuit(
        name="reset_-i", channels=[q_chans[0]], circuit=[]
    )

    return circuit, instruction_to_append
