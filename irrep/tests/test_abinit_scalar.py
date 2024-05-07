import os
import subprocess
from pathlib import Path
from monty.serialization import dumpfn, loadfn
import numpy as np

TEST_FILES_PATH = Path(__file__).parents[2] / "examples"

def test_abinit_scalar_example():

    os.chdir(TEST_FILES_PATH / "abinit_scalar")
    command = [
        "irrep",
        "-Ecut=50",
        "-code=abinit",
        "-fWFK=O_DS2_WFK",
        "-kpoints=1",
        "-IBstart=11",
        "-IBend=15",
        "-kpnames=GM",
    ]
    output = subprocess.run(command, capture_output=True, text=True)
    return_code = output.returncode
    assert return_code == 0, output.stderr

    # Load generated and reference output data
    data_ref = loadfn("ref_output.json")
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

    # Check general properties of the band structure
    bs_ref = data_ref['characters_and_irreps'][0]['subspace']
    bs_run = data_run['characters_and_irreps'][0]['subspace']
    assert abs(bs_ref['indirect gap (eV)'] - bs_run['indirect gap (eV)']) < 1e-4
    assert abs(bs_ref['Minimal direct gap (eV)'] - bs_run['Minimal direct gap (eV)']) < 1e-4
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
            assert np.allclose(irrep_ref[irrepname_ref], irrep_run[irrepname_run])  # compare multiplicities

#    # Remove output files created during run
#    for test_output_file in (
#            "irreps.dat",
#            "irreptable-template",
#            "trace.txt",
#            "irrep-output.json"
#    ):
#        os.remove(test_output_file)
