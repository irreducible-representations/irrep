import os
import subprocess
from pathlib import Path
from monty.serialization import dumpfn, loadfn
import numpy as np

TEST_FILES_PATH = Path(__file__).parents[2] / "examples"

def test_bi_hoti():

    os.chdir(TEST_FILES_PATH / "Bi-hoti")

    # Test specifying refUC in CLI
    command = [
        "irrep",
        "-spinor",
        "-code=vasp",
        "-kpnames=GM",
        "-kpoints=2",
        "-Ecut=50",
        "-EF=5.2156",
        "-IBstart=5",
        "-IBend=10",
        "-isymsep=3"
    ]
    output = subprocess.run(command, capture_output=True, text=True)
    return_code = output.returncode
    assert return_code == 0, output.stderr

    # Load generated and reference output data
    data_ref = loadfn("ref_output_isymsep.json")
    data_run = loadfn("irrep-output.json")

    # Check SpaceGroup
    sg_ref = data_ref['spacegroup']
    sg_run = data_run['spacegroup']
    assert sg_ref['name'] == sg_run['name']
    assert sg_ref['number'] == sg_run['number']
    assert sg_ref['spinor'] == sg_run['spinor']
    assert sg_ref['num_symmetries'] == sg_run['num_symmetries']
    assert sg_ref['cells_match'] == sg_run['cells_match']
    spinor = sg_ref['spinor']  # used later

    # Todo: implement safe check of symmetries


    # Check properties of separation by eigenvalues of symmetries
    assert data_ref['separated by symmetry'] == data_run['separated by symmetry']
    assert data_ref['separating symmetries'] == data_run['separating symmetries']
    assert len(data_ref['characters_and_irreps']) == len(data_ref['characters_and_irreps'])
    num_cases = len(data_ref['characters_and_irreps'])

    for i in range(num_cases):

        # Check that identified eigenvalues match
        evals_ref = data_ref['characters_and_irreps'][i]['symmetry_eigenvalues']
        evals_run = data_run['characters_and_irreps'][i]['symmetry_eigenvalues']
        assert len(evals_ref) == len(evals_run)
        for val1, val2 in zip(evals_ref, evals_run):
            assert abs(val1 - val2) < 1e-3

        # Check properties of bandstructure
        bs_ref = data_ref['characters_and_irreps'][i]['subspace']
        bs_run = data_run['characters_and_irreps'][i]['subspace']
        assert bs_ref['indirect gap (eV)'] == bs_run['indirect gap (eV)']
        assert bs_ref['Minimal direct gap (eV)'] == bs_run['Minimal direct gap (eV)']
        if spinor:
            assert bs_ref['Z4'] == bs_run['Z4']
            assert bs_ref['number of inversion-odd Kramers pairs'] == bs_run['number of inversion-odd Kramers pairs']
        else:
            assert bs_ref['number of inversion-odd states'] == bs_run['number of inversion-odd states']

        # Check properties at each k-point
        kp_ref = bs_ref['k-points'][0]
        kp_run = bs_run['k-points'][0]
        assert np.allclose(kp_ref['symmetries'], kp_run['symmetries'])
        assert np.allclose(kp_ref['energies'], kp_run['energies'], rtol=0., atol=1e-4)
        assert np.allclose(kp_ref['characters'], kp_run['characters'], rtol=0., atol=1e-4)
        assert kp_ref['characters_refUC_is_the_same'] == kp_run['characters_refUC_is_the_same']
        assert np.allclose(kp_ref['dimensions'], kp_run['dimensions'], rtol=0., atol=1e-4)
        for irrep_ref, irrep_run in zip(kp_ref['irreps'], kp_run['irreps']):
            assert len(irrep_ref) == len(irrep_run)
            for irrepname_ref, irrepname_run in zip(irrep_ref.keys(), irrep_run.keys()):
                assert irrepname_ref == irrepname_run  # compare strings of irreps
                assert np.allclose(irrep_ref[irrepname_ref], irrep_run[irrepname_run], rtol=0, atol=1e-4)  # compare multiplicities

    # Remove output files created during run
    for test_output_file in (
            "irreps.dat",
            "irreptable-template",
            "trace.txt",
            "irrep-output.json"
    ):
        os.remove(test_output_file)
