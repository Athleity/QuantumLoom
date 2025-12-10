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

# pylint: disable=redefined-outer-name, duplicate-code

import pytest

from loom.interpreter import Syndrome


@pytest.fixture()
def syndrome1_alt_labels():
    """Fixture for a sample syndrome1 object with different labels."""
    return Syndrome(
        stabilizer="stab0",
        measurements=(("c_(1,1,1)", 7),),
        block="block0",
        round=3,
        corrections=(("c_(5, 8, 0)", 0),),
        labels={"space_coordinate": (2, 5, 2), "time_coordinate": (4,), "color": 2},
    )


class TestSyndrome:
    """Tests for the Syndrome class."""

    def test_creation_syndrome(self):
        """Tests the creation of a Syndrome object."""

        syndrome_attributes = {
            "stab_uuid": ["stab0", "stab1", "stab2"],
            "measurements": [(), (("c_(1,1,1)", 7),), (("c_(2,5,2)", 0),)],
            "block": ["block0", "block1", "block2"],
            "corrections": [(), (("c_(5, 8, 0)", 0),), ()],
            "round": [3, 9, 2],
            "labels": [
                {},
                {"space_coordinate": (1, 1, 1)},
                {"space_coordinate": (2, 5, 2), "time_coordinate": (4,), "color": 2},
            ],
        }

        # Loop over the three examples
        for i in range(len(syndrome_attributes["stab_uuid"])):

            syndrome = Syndrome(
                stabilizer=syndrome_attributes["stab_uuid"][i],
                measurements=syndrome_attributes["measurements"][i],
                block=syndrome_attributes["block"][i],
                round=syndrome_attributes["round"][i],
                corrections=syndrome_attributes["corrections"][i],
                labels=syndrome_attributes["labels"][i],
            )

            assert syndrome.stabilizer == syndrome_attributes["stab_uuid"][i]
            assert syndrome.measurements == syndrome_attributes["measurements"][i]
            assert syndrome.block == syndrome_attributes["block"][i]
            assert syndrome.round == syndrome_attributes["round"][i]
            assert syndrome.corrections == syndrome_attributes["corrections"][i]
            assert syndrome.labels == syndrome_attributes["labels"][i]

    def test_syndrome_equality(self, syndrome_sample, syndrome1_alt_labels):
        """Test the equality method"""

        # They are equal despite having different labels
        assert syndrome_sample == syndrome1_alt_labels

        # Test for inequality field by field
        syndrome3 = Syndrome(
            stabilizer="stab1",
            measurements=syndrome_sample.measurements,
            block=syndrome_sample.block,
            round=syndrome_sample.round,
            corrections=syndrome_sample.corrections,
            labels={},
        )

        syndrome4 = Syndrome(
            stabilizer=syndrome_sample.stabilizer,
            measurements=(("c_(1,1,2)", 7),),
            block=syndrome_sample.block,
            round=syndrome_sample.round,
            corrections=syndrome_sample.corrections,
            labels={},
        )

        syndrome5 = Syndrome(
            stabilizer=syndrome_sample.stabilizer,
            measurements=syndrome_sample.measurements,
            block="entropica_labs",
            round=syndrome_sample.round,
            corrections=syndrome_sample.corrections,
            labels={},
        )

        syndrome6 = Syndrome(
            stabilizer=syndrome_sample.stabilizer,
            measurements=syndrome_sample.measurements,
            block=syndrome_sample.block,
            round=334,
            corrections=syndrome_sample.corrections,
            labels={},
        )

        syndrome7 = Syndrome(
            stabilizer=syndrome_sample.stabilizer,
            measurements=syndrome_sample.measurements,
            block=syndrome_sample.block,
            round=syndrome_sample.round,
            corrections=(("c_(5, 8, 0)", 42),),
            labels={},
        )

        wrong_syndromes = [syndrome3, syndrome4, syndrome5, syndrome6, syndrome7]
        for wrong_syndrome in wrong_syndromes:
            assert syndrome_sample != wrong_syndrome

    def test_syndrome_repr(self, syndrome_sample):
        """Test the string representation of the Syndrome object"""

        expected_repr = (
            "Syndrome(Measurements: (('c_(1,1,1)', 7),), "
            "Corrections: (('c_(5, 8, 0)', 0),), Round: 3, "
            "Labels: {'space_coordinate': (1, 1, 1)})"
        )
        assert repr(syndrome_sample) == expected_repr

    def test_syndrome_hash(self, syndrome_sample, syndrome1_alt_labels):
        """Test the proper hashing of the Syndrome object"""

        # Check that two identical syndromes have the same hash
        assert hash(syndrome_sample) == hash(syndrome1_alt_labels)

        # Check that two different syndromes have different hashes
        syndrome3 = Syndrome(
            stabilizer="stab0",
            measurements=(("c_(1,1,1)", 7),),
            block="block0",
            round=8,
            corrections=(("c_(5, 8, 0)", 0),),
            labels={"space_coordinate": (2, 5, 2)},
        )
        assert hash(syndrome_sample) != hash(syndrome3)
