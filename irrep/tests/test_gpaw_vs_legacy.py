
import pytest
from irrep.tests.conftest import CALC_TYPE_LIST
from .test_paw import BandstrGpawDiamond, BandstrGpawBismuth  # noqa: F401
import numpy as np


@pytest.mark.parametrize("bandstructure_fixture", ["BandstrGpawDiamond", "BandstrGpawBismuth"])
def test_bandstructure_gpaw(bandstructure_fixture, request):
    if "LEGACY" not in CALC_TYPE_LIST:
        pytest.skip("Legacy GPAW not available for this version of GPAW.")
    bandstructure_dict = request.getfixturevalue(bandstructure_fixture)
    bandstructure_old = bandstructure_dict["OLD"]
    bandstructure_legacy = bandstructure_dict["LEGACY"]
    assert bandstructure_old.spacegroup.equals(bandstructure_legacy.spacegroup)
    assert np.allclose(bandstructure_old.kplist, bandstructure_legacy.kplist)
    for ik, k in enumerate(bandstructure_old.kplist):
        kp_old = bandstructure_old.kpoints[ik]
        kp_legacy = bandstructure_legacy.kpoints[ik]
        for key in ["k", "spinor", "weight", "num_bands", "RecLattice", "Energy_raw", "ig", "eKG", "WF"]:
            assert np.allclose(getattr(kp_old, key), getattr(kp_legacy, key))
        kp_paw_old = bandstructure_old.kpoints_paw[ik]
        kp_paw_legacy = bandstructure_legacy.kpoints_paw[ik]
        for key in ["k", "spinor", "weight", "num_bands", "RecLattice", "proj", "atom_positions"]:
            try:
                assert np.allclose(getattr(kp_paw_old, key), getattr(kp_paw_legacy, key)), f"Mismatch in {key} for k-point {ik}: {getattr(kp_paw_old, key)} vs {getattr(kp_paw_legacy, key)}"
            except TypeError as e:
                raise TypeError(f"TypeError in comparing {key} for k-point {ik}: {getattr(kp_paw_old, key)} vs {getattr(kp_paw_legacy, key)} : {e}")
        WF_old = kp_paw_old.get_wavefunction()
        WF_legacy = kp_paw_legacy.get_wavefunction()
        assert WF_old.shape == WF_legacy.shape, f"Mismatch in wavefunction shape for k-point {ik}: {WF_old.shape} vs {WF_legacy.shape}"
        assert WF_old.shape[0] == kp_old.num_bands, f"Mismatch in number of bands for k-point {ik}: {WF_old.shape[0]} vs {kp_old.num_bands}"
        # ratio = 1./2.59775567 # np.mean(WF_legacy / WF_old).
        # print(f"Scaling wavefunction by {1./ratio}  the lattice vollume is")
        # WF_old = WF_old * ratio
        for iband in range(kp_old.num_bands):
            diff = np.max(np.abs(WF_old[iband] - WF_legacy[iband]))
            assert diff < 1e-6, (f"Mismatch in wavefunction for k-point {ik}, band {iband}: max difference {diff}\n"
            f" WF_old: ({WF_old[iband].shape}){WF_old[iband]}\n"
                f" WF_legacy ({WF_legacy[iband].shape}): {WF_legacy[iband]}"
                f"ratio: { WF_old[iband] / WF_legacy[iband]}")
