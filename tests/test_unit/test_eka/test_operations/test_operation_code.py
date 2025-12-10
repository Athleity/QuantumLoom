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


from loom.eka.utilities import (
    SingleQubitPauliEigenstate,
    Direction,
    Orientation,
    ResourceState,
    loads,
    dumps,
)
from loom.eka.operations import (
    Grow,
    Shrink,
    Merge,
    Split,
    MeasureLogicalX,
    MeasureLogicalY,
    MeasureLogicalZ,
    ResetAllDataQubits,
    ResetAllAncillaQubits,
    Operation,
    StateInjection,
)


# pylint: disable=protected-access
class TestCodeOperation:
    """
    Test the creation of the logical operator measurement and qubit reset
    operations.
    """

    def test_measure_logical_x(self):
        """
        Test the creation of a logical X measurement operation"""

        meas_log_x = MeasureLogicalX(input_block_name="q1")
        assert meas_log_x.input_block_name == "q1"
        assert meas_log_x.logical_qubit == 0
        assert meas_log_x.__class__.__name__ == "MeasureLogicalX"
        assert meas_log_x._inputs == ("q1",)
        assert meas_log_x._outputs == ("q1",)
        # Test the loads/dumps both using the right class and the abstract base class
        assert meas_log_x == loads(MeasureLogicalX, dumps(meas_log_x))
        assert meas_log_x == loads(Operation, dumps(meas_log_x))

    def test_measure_logical_y(self):
        """
        Test the creation of a logical Y measurement operation
        """

        meas_log_y = MeasureLogicalY(input_block_name="q1")
        assert meas_log_y.input_block_name == "q1"
        assert meas_log_y.logical_qubit == 0
        assert meas_log_y.__class__.__name__ == "MeasureLogicalY"
        assert meas_log_y._inputs == ("q1",)
        assert meas_log_y._outputs == ("q1",)
        # Test the loads/dumps both using the right class and the abstract base class
        assert meas_log_y == loads(MeasureLogicalY, dumps(meas_log_y))
        assert meas_log_y == loads(Operation, dumps(meas_log_y))

    def test_measure_logical_z(self):
        """
        Test the creation of a logical Z measurement operation"""

        meas_log_z = MeasureLogicalZ(input_block_name="q1")
        assert meas_log_z.input_block_name == "q1"
        assert meas_log_z.logical_qubit == 0
        assert meas_log_z.__class__.__name__ == "MeasureLogicalZ"
        assert meas_log_z._inputs == ("q1",)
        assert meas_log_z._outputs == ("q1",)
        # Test the loads/dumps both using the right class and the abstract base class
        assert meas_log_z == loads(MeasureLogicalZ, dumps(meas_log_z))
        assert meas_log_z == loads(Operation, dumps(meas_log_z))

    def test_reset_all_data_qubits(self):
        """
        Test the creation of a logical reset operation"""

        # Test the creation of a logical reset operation
        for state in SingleQubitPauliEigenstate:
            logical_reset = ResetAllDataQubits(input_block_name="q1", state=state)
            assert logical_reset.input_block_name == "q1"
            assert logical_reset.__class__.__name__ == "ResetAllDataQubits"
            assert logical_reset._inputs == ("q1",)
            assert logical_reset._outputs == ("q1",)
            assert logical_reset.state == state
            # Test the loads/dumps both using the right class and the abstract base
            # class
            assert logical_reset == loads(ResetAllDataQubits, dumps(logical_reset))
            assert logical_reset == loads(Operation, dumps(logical_reset))

    def test_reset_all_ancilla_qubits(self):
        """
        Test the creation of an ancilla reset operation"""
        # Test the ancilla reset operation
        for state in SingleQubitPauliEigenstate:
            ancilla_reset = ResetAllAncillaQubits(input_block_name="q1", state=state)
            assert ancilla_reset.input_block_name == "q1"
            assert ancilla_reset.__class__.__name__ == "ResetAllAncillaQubits"
            assert ancilla_reset._inputs == ("q1",)
            assert ancilla_reset._outputs == ("q1",)
            assert ancilla_reset.state == state
            # Test the loads/dumps both using the right class and the abstract base
            # class
            assert ancilla_reset == loads(ResetAllAncillaQubits, dumps(ancilla_reset))
            assert ancilla_reset == loads(Operation, dumps(ancilla_reset))

    def test_grow(self):
        """Test the creation of a Grow operation"""
        grow = Grow(input_block_name="q1", direction=Direction.TOP, length=1)
        assert grow.input_block_name == "q1"
        assert grow.direction == Direction.TOP
        assert grow.length == 1
        assert grow.__class__.__name__ == "Grow"
        assert grow._inputs == ("q1",)
        assert grow._outputs == ("q1",)
        # Test the loads/dumps both using the right class and the abstract base class
        assert grow == loads(Grow, dumps(grow))
        assert grow == loads(Operation, dumps(grow))

        # Test invalid length input
        err_msg_length = "length has to be larger than 0."

        invalid_lengths = [-1, 0]
        for invalid_length in invalid_lengths:
            with pytest.raises(ValueError) as cm:
                grow = Grow(
                    input_block_name="q1",
                    direction=Direction.TOP,
                    length=invalid_length,
                )
            assert err_msg_length in str(cm.value)

    def test_shrink(self):
        """Test the creation of a Shrink operation"""

        shrink = Shrink(
            input_block_name="q1",
            direction=Direction.TOP,
            length=1,
        )
        assert shrink.input_block_name == "q1"
        assert shrink.direction == Direction.TOP
        assert shrink.length == 1
        assert shrink.__class__.__name__ == "Shrink"
        assert shrink._inputs == ("q1",)
        assert shrink._outputs == ("q1",)
        # Test the loads/dumps both using the right class and the abstract base class
        assert shrink == loads(Shrink, dumps(shrink))
        assert shrink == loads(Operation, dumps(shrink))

        # Test invalid length input
        err_msg_length = "length has to be larger than 0."

        invalid_lengths = [-1, 0]
        for invalid_length in invalid_lengths:
            with pytest.raises(ValueError) as cm:
                shrink = Shrink(
                    input_block_name="q1",
                    direction=Direction.TOP,
                    length=invalid_length,
                )
            assert err_msg_length in str(cm.value)

    def test_merge(self):
        """Test the creation of a Merge operation"""

        merge = Merge(
            input_blocks_name=["q1", "q2"],
            output_block_name="q3",
            orientation=Orientation.HORIZONTAL,
        )
        assert merge.input_blocks_name == ("q1", "q2")
        assert merge.output_block_name == "q3"
        assert merge.orientation == Orientation.HORIZONTAL
        assert merge.__class__.__name__ == "Merge"
        assert merge._inputs == ("q1", "q2")
        assert merge._outputs == ("q3",)
        # Test the loads/dumps both using the right class and the abstract base class
        assert merge == loads(Merge, dumps(merge))
        assert merge == loads(Operation, dumps(merge))

    def test_split(self):
        """Test the creation of a Split operation"""

        split = Split(
            input_block_name="q1",
            output_blocks_name=["q2", "q3"],
            orientation=Orientation.VERTICAL,
            split_position=3,
        )
        assert split.input_block_name == "q1"
        assert split.output_blocks_name == ("q2", "q3")
        assert split.orientation == Orientation.VERTICAL
        assert split.split_position == 3
        assert split.__class__.__name__ == "Split"
        assert split._inputs == ("q1",)
        assert split._outputs == ("q2", "q3")
        # Test the loads/dumps both using the right class and the abstract base class
        assert split == loads(Split, dumps(split))
        assert split == loads(Operation, dumps(split))

        # Test invalid creation of split
        err_msg_split_position = "split_position has to be larger than 0."

        with pytest.raises(ValueError) as cm:
            Split(
                input_block_name="q1",
                output_blocks_name=["q2", "q3"],
                orientation=Orientation.VERTICAL,
                split_position=-1,
            )
        assert err_msg_split_position in str(cm.value)

    def test_state_injection(self):
        """Test the creation of a state injection operation"""
        # Test the creation of a state injection operation
        for state in ResourceState:
            state_injection = StateInjection(
                input_block_name="q1", resource_state=state
            )
            assert state_injection.input_block_name == "q1"
            assert state_injection.__class__.__name__ == "StateInjection"
            assert state_injection._inputs == ("q1",)
            assert state_injection._outputs == ("q1",)
            assert state_injection.resource_state == state
            # Test the loads/dumps using the right class and the abstract base class
            assert state_injection == loads(StateInjection, dumps(state_injection))
            assert state_injection == loads(Operation, dumps(state_injection))
