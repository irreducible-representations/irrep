
import os
import pytest
from irrep.bandstructure import BandStructure
import numpy as np
from .conftest import TEST_FILES_PATH
from gpaw import GPAW


@pytest.fixture(scope="module", autouse=True)
def BandstrGpawDiamond():
    calc = GPAW(os.path.join(TEST_FILES_PATH, "diamond-gpaw", "diamond-nscf-irred.gpw"))
    bandstructure = BandStructure(calculator_gpaw=calc, read_paw=True, code='gpaw', Ecut=300)
    return bandstructure


@pytest.fixture(scope="module", autouse=True)
def BandstrGpawBismuth():
    calc = GPAW(os.path.join(TEST_FILES_PATH, "gpaw", "Bi.gpw"))
    bandstructure = BandStructure(calculator_gpaw=calc, read_paw=True, code='gpaw', Ecut=300)
    return bandstructure


@pytest.fixture
def check_traces_gpaw():
    def __inner__(bandstructure, include_paw, include_pseudo, ik):
        return_msg = ""
        kp0 = bandstructure.kpoints_paw[ik]
        spacegroup = bandstructure.spacegroup
        kp0g = bandstructure.kpoints[ik]
        kp0g.init_traces(degen_thresh=0.001)
        kp0g.identify_irreps()
        kp0g.write_characters(refcell=False)
        char = kp0g.char

        degeneracies = kp0g.degeneracies
        char_paw = []
        isym_little = []
        for i, symop in enumerate(spacegroup.symmetries):
            # print(f"Symmetry {i}:")
            # symop.show()
            G = symop.transform_k(kp0.k) - kp0.k
            Gint = np.rint(G).astype(int)
            if not np.allclose(G, Gint, atol=1e-5):
                continue
            isym_little.append(i)
            kp0_rot = kp0.get_transformed_copy(symop, k_new=kp0.k)
            o00 = bandstructure.overlap_paw.product(kp0, kp0_rot, include_paw=include_paw, include_pseudo=include_pseudo).diagonal()
            norm = np.diag(bandstructure.overlap_paw.product(kp0, kp0, include_paw=include_paw, include_pseudo=include_pseudo))
            o00 = o00 / norm

            char_paw_isym = []
            start = 0
            for deg in degeneracies:
                end = start + deg
                char_paw_isym.append(np.round(o00[start:end].sum(), 8))
                start = end
            char_paw.append(char_paw_isym)


        char_paw = np.array(char_paw).T

        return_msg += f"kpoint = {kp0.k}, little group size = {len(isym_little)}"
        if np.max(np.abs(char.imag)) < 1e-8:
            char = char.real
        if np.max(np.abs(char_paw.imag)) < 1e-8:
            char_paw = char_paw.real
            return_msg += "Warning: imaginary part of pseudo char is significant!"

        return_msg += f"{char.shape=}, {char_paw.shape=}, {len(isym_little)=}"
        if not np.allclose(char[:-1], char_paw[:-1], atol=1e-4):
            allgood = False
            return_msg += "Discrepancy between PAW and pseudo chars!"
            for isym, sym in enumerate(isym_little):
                symop = spacegroup.symmetries[sym]
                return_msg += "\n\n" + "#" * 80 + "\n"
                symop.show()
                return_msg += f"Symmetry {sym} ({isym} of {len(isym_little)}): transforms {kp0.k} to {symop.transform_k(kp0.k)}\n"

                diff = np.max(abs(char[:-1, isym] - char_paw[:-1, isym]))
                return_msg += f"  Discrepancy in char: {diff}\n"
                if diff > 1e-4:
                    return_msg += "  Warning: significant discrepancy detected!\n"
                    return_msg += f"  Pseudo char: {   char[:,isym]}\n"
                    return_msg += f"  PAW   char: {char_paw[:,isym]}\n"
                    nonzero = np.where(np.abs(char[:, isym]) > 1e-5)[0]
                    ratio = char_paw[nonzero, isym] / char[nonzero, isym]
                    ratio = ratio[:-1]
                    return_msg += f" ratio of nonzero elements:\n absolute {np.abs(ratio)},\n phase {np.round(np.angle(ratio)/np.pi*180,4)} degrees"
        else:
            return_msg += "No significant discrepancies found."
            allgood = True

        return allgood, return_msg
    return __inner__


@pytest.mark.parametrize("incl_paw_pseudo", [(True, True), (True, False), (False, True)], ids=lambda p: f"paw={p[0]}_pseudo={p[1]}")
@pytest.mark.parametrize("ik", range(8), ids=lambda k: f"ik={k}")
def test_traces_gpaw_diamopnd(BandstrGpawDiamond, check_traces_gpaw, incl_paw_pseudo, ik):
    include_paw, include_pseudo = incl_paw_pseudo
    good, msg = check_traces_gpaw(BandstrGpawDiamond, include_paw=include_paw, include_pseudo=include_pseudo, ik=ik)
    assert good, msg


@pytest.mark.parametrize("incl_paw_pseudo", [(True, True), (True, False), (False, True)], ids=lambda p: f"paw={p[0]}_pseudo={p[1]}")
@pytest.mark.parametrize("ik", range(32), ids=lambda k: f"ik={k}")
def test_traces_gpaw_bismuth(BandstrGpawBismuth, check_traces_gpaw, incl_paw_pseudo, ik):
    print(f"Testing ik={ik} (out of {BandstrGpawBismuth.num_k}) with incl_paw_pseudo={incl_paw_pseudo}")
    include_paw, include_pseudo = incl_paw_pseudo
    good, msg = check_traces_gpaw(BandstrGpawBismuth, include_paw=include_paw, include_pseudo=include_pseudo, ik=ik)
    assert good, msg
