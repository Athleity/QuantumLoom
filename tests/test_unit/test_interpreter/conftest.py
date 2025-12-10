"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import pytest

from loom.interpreter import Syndrome, Detector

# pylint: disable=redefined-outer-name, duplicate-code


@pytest.fixture()
def syndrome_sample():
    """Fixture for a sample Syndrome object."""
    return Syndrome(
        stabilizer="stab0",
        measurements=(("c_(1,1,1)", 7),),
        block="block0",
        round=3,
        corrections=(("c_(5, 8, 0)", 0),),
        labels={"space_coordinate": (1, 1, 1)},
    )


@pytest.fixture()
def syndrome_sample_next_round():
    """Fixture for a sample Syndrome object in the next round."""
    return Syndrome(
        stabilizer="stab0",
        measurements=(("c_(1,1,1)", 8),),
        block="block0",
        round=4,
        corrections=(),
        labels={"time_coordinate": (9,)},
    )


@pytest.fixture()
def detector_sample(syndrome_sample, syndrome_sample_next_round):
    """Fixture for a sample Detector object."""
    return Detector(
        syndromes=[syndrome_sample, syndrome_sample_next_round],
        labels=syndrome_sample_next_round.labels,
    )
