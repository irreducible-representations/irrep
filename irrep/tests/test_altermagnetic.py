import numpy as np
from pytest import approx

from irrep.altermagnetic_transformer import find_altermagnetic_symmetry


def test_find_altermagnetic_symmetry():

    a = 4.134
    c = 6.652
    lattice = a * np.array([[1, 0, 0], [-1 / 2, np.sqrt(3) / 2, 0], [0, 0, c / a]])
    positions = np.array(
        [
            [0, 0, 0],
            [0, 0, 1 / 2],
            [1 / 3, 2 / 3, 1 / 4],
            [2 / 3, 1 / 3, 3 / 4],
        ]
    )
    magmom = np.array([1, -1, 0, 0])
    symop = find_altermagnetic_symmetry(lattice=lattice, positions=positions, typat=[1, 1, 2, 2], magmom=magmom)
    assert symop.angle == approx(np.pi / 3), f"Expected angle=np.pi/3, got {symop.angle}"
    assert symop.axis == approx([0, 0, 1]), f"Expected axis=[0, 0, 1], got {symop.axis}"
    assert symop.translation == approx([0, 0, 0.5]), f"Expected translation=[0, 0, 0.5], got {symop.translation}"
    assert symop.inversion is False, f"Expected inversion=False, got {symop.inversion}"
    assert symop.time_reversal is False, f"Expected timereversal=False, got {symop.timereversal}"
