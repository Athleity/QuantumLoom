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

from loom.interpreter import Detector, Syndrome

# pylint: disable=redefined-outer-name, duplicate-code


@pytest.fixture()
def detector_sample_relabeled(detector_sample):
    """Return the detector_sample with different label."""
    return Detector(syndromes=detector_sample.syndromes, labels={"quantum": 42})


@pytest.fixture()
def different_detector_sample():
    """Fixture for a different Detector object."""
    s = Syndrome(
        stabilizer="stab0",
        measurements=(("c_(1,1,1)", 7),),
        block="block0",
        round=8,
        corrections=(("c_(5, 8, 0)", 0),),
        labels={"space_coordinate": (2, 5, 2)},
    )

    return Detector(syndromes=[s], labels={})


class TestDetector:
    """Tests for the Detector class."""

    def test_creation_detector(
        self,
        detector_sample,
        syndrome_sample,
        syndrome_sample_next_round,
        different_detector_sample,
    ):
        """Tests the creation of a Detector object."""

        assert isinstance(detector_sample, Detector)
        assert detector_sample.syndromes == (
            syndrome_sample,
            syndrome_sample_next_round,
        )
        assert detector_sample.labels == syndrome_sample_next_round.labels

        assert isinstance(different_detector_sample, Detector)
        assert different_detector_sample.syndromes == (
            different_detector_sample.syndromes[0],
        )
        assert different_detector_sample.labels == {}

    def test_detector_equality(
        self, detector_sample, detector_sample_relabeled, different_detector_sample
    ):
        """Test the equality method"""
        assert detector_sample == detector_sample_relabeled
        assert detector_sample != different_detector_sample

    def test_detector_rounds(self, detector_sample, different_detector_sample):
        """Test the rounds property of Detector"""
        assert detector_sample.rounds() == (3, 4)
        assert different_detector_sample.rounds() == (8,)

    def test_detector_stabilizer(self, detector_sample, different_detector_sample):
        """Test the stabilizer property of the Detector object"""

        assert detector_sample.stabilizer() == ("stab0", "stab0")
        assert different_detector_sample.stabilizer() == ("stab0",)

    def test_detector_repr(self, detector_sample):
        """Test the string representation of the Detector object"""

        expected_repr = (
            "Detector(Syndromes: (Syndrome(Measurements: (('c_(1,1,1)', 7),), "
            "Corrections: (('c_(5, 8, 0)', 0),), Round: 3, "
            "Labels: {'space_coordinate': (1, 1, 1)}),"
            " Syndrome(Measurements: (('c_(1,1,1)', 8),), "
            "Corrections: (), Round: 4, "
            "Labels: {'time_coordinate': (9,)})), Labels: {'time_coordinate': (9,)})"
        )
        assert repr(detector_sample) == expected_repr

    def test_detector_hash(
        self,
        syndrome_sample,
        detector_sample,
        different_detector_sample,
    ):
        """Test the proper hashing of the Detector object"""

        # Create a syndrome identical to syndrome_sample but with different labels
        syndrome2 = Syndrome(
            syndrome_sample.stabilizer,
            syndrome_sample.measurements,
            block=syndrome_sample.block,
            round=syndrome_sample.round,
            corrections=syndrome_sample.corrections,
            labels={"time_coordinate": (9,)},
        )

        detector1 = Detector(syndromes=[syndrome_sample, syndrome2], labels={})
        detector2 = Detector(
            syndromes=[syndrome_sample, syndrome2], labels={"quantum": 42}
        )

        # Check that two identical syndromes have the same hash
        assert hash(detector1) == hash(detector2)

        # Check that two different syndromes have different hashes
        assert hash(detector_sample) != hash(different_detector_sample)
