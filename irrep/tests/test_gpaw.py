from gpaw import GPAW
import numpy as np
import pytest
from irrep.bandstructure import BandStructure
from pytest import approx





@pytest.mark.parametrize("spinor", [True, False])
def test_gpaw_spinorbit(spinor):
    calc = GPAW("../../examples/gpaw/Bi-gamma.gpw")

    nspinor = 2 if spinor else 1

    print("Fermi level", calc.get_fermi_level())
    bandstructure = BandStructure(code="gpaw", calculator_gpaw=calc,
                                  spinor=spinor,
                                  IBstart=10 * nspinor,
                                  IBend=15 * nspinor,
                                  Ecut=50,
                                  degen_thresh=2e-4,
                                  calculate_traces=True,
                                  irreps=True,
                                  search_cell=True,
                                  kplist=[0])
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
        energies_ref = np.array([-7.30428916, -1.83041608, 2.4278237, 5.2004299, 5.499876])
    else:
        irreps_ref = [ {'GM1+': 1},
                       {'GM2-': 1},
                       {'GM1+': 1},
                       {'GM3+': 1},
                        ]
        energies_ref = np.array([-7.30392234, -1.82969316, 4.11551033, 4.50561308])

    assert kGM.Energy_mean == approx(energies_ref, abs=1e-6), f"calculated energies {kGM.Energy_mean} differ from reference {energies_ref} by {abs(kGM.Energy_mean - energies_ref).max()}"

    for irr, irref, E in zip(kGM.irreps, irreps_ref, energies_ref):
        for k in irref:
            assert k in irr, f"at energy {E}, irrep {k} not found in calculated irreps {irr}"
            assert irr[k] == approx(irref[k], abs=1e-3), f"at energy {E}, irrep {k} has multiplicity {irr[k]}, expected {irref[k]}"
        for k in irr:
            if abs(irr[k]) > 1e-3:
                assert k in irref, f"at energy {E}, extra irrep {k}({irr[k]}) found in calculated irreps {irr}, not in reference {irref}"
