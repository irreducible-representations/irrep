from .conftest import TEST_FILES_PATH
from pytest import approx
from irrep.bandstructure import BandStructure
import pytest
import numpy as np
from gpaw import GPAW
import pickle

import gpaw

from .conftest import CALC_TYPE_LIST, GPAW_VERSION




@pytest.mark.parametrize("calc_type", CALC_TYPE_LIST)
@pytest.mark.parametrize("spinor", [True, False])
def test_gpaw_spinorbit(spinor, calc_type):
    print(f"GPAW version: {gpaw.__version__}, using GPAW_VERSION={GPAW_VERSION}")


    nspinor = 2 if spinor else 1

    if calc_type == "LEGACY":
        kwargs = dict(legacy_gpaw=True)
        calc_type_read = "OLD"
    else:
        kwargs = {}
        calc_type_read = calc_type
    calc = GPAW(f"{TEST_FILES_PATH}/gpaw_{calc_type_read}/Bi-gamma.gpw", **kwargs)


    print("Fermi level", calc.get_fermi_level())
    bandstructure = BandStructure.from_gpaw(calculator_gpaw=calc,
                                  spinor=spinor,
                                  IBstart=10 * nspinor,
                                  IBend=15 * nspinor,
                                  Ecut=50,
                                  degen_thresh=2e-4,
                                  calculate_traces=True,
                                  irreps=True,
                                  read_paw=not spinor,  # for now, read_pae does not work with spinor calculations
                                  search_cell=True,
                                  kplist=[0],
        **kwargs
    )
    output_file = f"{TEST_FILES_PATH}/gpaw_{calc_type_read}/Bi-gamma_bandstructure.pickle"
    bandstructure.pickle(output_file)
    # print ("Bandstructure",bandstructure)
    bandstructure.spacegroup.show()
    bandstructure.identify_irreps(kpnames=["GM"], verbosity=0)

    # Temporary, until we make it valid for isymsep
    bandstructure.write_characters(refcell=False)

    bandstructure.write_trace()
    print("done")


    kGM = bandstructure.kpoints[0]

    if spinor:
        irreps_ref = [{'-GM8': 1},
                      {'-GM9': 1},
                      {'-GM8': 1},
                      {'-GM8': 1},
                      {'-GM4': 1, '-GM5': 1}]
        if calc_type in ["OLD", "LEGACY"]:
            energies_ref = np.array([-7.30428916, -1.83041608, 2.4278237, 5.2004299, 5.499876])
        else:
            energies_ref = np.array([-7.30393384, -1.83000555, 2.42808016, 5.20057551, 5.5001332])
    else:
        irreps_ref = [ {'GM1+': 1},
                       {'GM2-': 1},
                       {'GM1+': 1},
                       {'GM3+': 1},
                        ]
        if calc_type in ["OLD", "LEGACY"]:
            energies_ref = np.array([-7.30392234, -1.82969316, 4.11551033, 4.50561308])
        else:
            energies_ref = np.array([-7.30356775, -1.82928119, 4.11561034, 4.5058931])

    assert kGM.Energy_mean == approx(energies_ref, abs=1e-6), f"calculated energies {kGM.Energy_mean} differ from reference {energies_ref} by {abs(kGM.Energy_mean - energies_ref).max()}"

    for irr, irref, E in zip(kGM.irreps, irreps_ref, energies_ref):
        for k in irref:
            assert k in irr, f"at energy {E}, irrep {k} not found in calculated irreps {irr}"
            assert irr[k] == approx(irref[k], abs=1e-3), f"at energy {E}, irrep {k} has multiplicity {irr[k]}, expected {irref[k]}"
        for k in irr:
            if abs(irr[k]) > 1e-3:
                assert k in irref, f"at energy {E}, extra irrep {k}({irr[k]}) found in calculated irreps {irr}, not in reference {irref}"


@pytest.mark.parametrize("calc_type", CALC_TYPE_LIST)
def test_gpaw_storage(calc_type):
    if calc_type == "LEGACY":
        kwargs = dict(legacy_gpaw=True)
        calc_type_read = "OLD"
    else:
        kwargs = {}
        calc_type_read = calc_type
    calc = GPAW(f"{TEST_FILES_PATH}/gpaw_{calc_type_read}/Bi.gpw", **kwargs)

    print("Fermi level", calc.get_fermi_level())
    bandstructure = BandStructure.from_gpaw(calculator_gpaw=calc,
                                  IBstart=10,
                                  IBend=15,
                                  Ecut=50,
                                  degen_thresh=2e-4,
                                  calculate_traces=True,
                                  irreps=True,
                                  read_paw=True,
                                  search_cell=True,
                                  store_paw=True,
                                  kplist=[0])
    output_file = f"{TEST_FILES_PATH}/gpaw_{calc_type_read}/Bi-gamma_bandstructure_tmp.pickle"
    bandstructure.pickle(output_file)
    bandstructure2 = pickle.load(open(output_file, "rb"))
    assert len(bandstructure.kpoints_paw) == len(bandstructure2.kpoints_paw), "Number of k-points in bandstructure and loaded bandstructure do not match."
    assert len(bandstructure.kpoints) > 0, "No k-points found in bandstructure."
    for kp1, kp2 in zip(bandstructure.kpoints_paw, bandstructure2.kpoints_paw):
        assert np.allclose(kp1.get_wavefunction(), kp2.get_wavefunction()), f"Wavefunctions differ for k-point {kp1.k} and {kp2.k}"
        assert np.allclose(kp1.proj, kp2.proj), f"Projections differ for k-point {kp1.k} and {kp2.k}"
    bandstructure.spacegroup.show()
    bandstructure.identify_irreps(kpnames=["GM"], verbosity=0)
