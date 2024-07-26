import os
import subprocess
from pathlib import Path
from monty.serialization import loadfn
import numpy as np

TEST_FILES_PATH = Path(__file__).parents[2] / "examples"

def test_espresso_spinor_example():

    os.chdir(TEST_FILES_PATH / "espresso_spinor")

    command = [
        "irrep",
        "-spinor",
        "-Ecut=100",
        "-code=espresso",
        "-prefix=Bi",
        "-kpoints=1,2,3,4,5,6",
        "-kpnames=A,GM,M,Y,L,V",
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
    assert sg_ref['num symmetries'] == sg_run['num symmetries']
    assert sg_ref['cells match'] == sg_run['cells match']
    spinor = sg_ref['spinor']  # used later

    # Todo: implement safe check of symmetries

    # Check general properties of the band structure
    bs_ref = data_ref['characters and irreps'][0]['subspace']
    bs_run = data_run['characters and irreps'][0]['subspace']
    assert bs_ref['indirect gap (eV)'] == bs_run['indirect gap (eV)']
    assert bs_ref['Minimal direct gap (eV)'] == bs_run['Minimal direct gap (eV)']
    if spinor:
        assert bs_ref['Z4'] == bs_run['Z4']
        assert bs_ref['number of inversion-odd Kramers pairs'] == bs_run['number of inversion-odd Kramers pairs']
    else:
        assert bs_ref['number of inversion-odd states'] == bs_run['number of inversion-odd states']

    # Check properties at each k-point
    kp_ref = bs_ref['k points'][0]
    kp_run = bs_run['k points'][0]
    assert np.allclose(kp_ref['symmetries'], kp_run['symmetries'])
    assert np.allclose(kp_ref['energies'], kp_run['energies_mean'], rtol=0., atol=1e-4)
    assert np.allclose(kp_ref['characters'], kp_run['characters'], rtol=0., atol=1e-4)
    assert kp_ref['characters refUC is the same'] == kp_run['characters refUC is the same']
    assert np.allclose(kp_ref['dimensions'], kp_run['dimensions'], rtol=0., atol=1e-4)
    for irrep_ref, irrep_run in zip(kp_ref['irreps'], kp_run['irreps']):
        assert len(irrep_ref) == len(irrep_run)
        for irrepname_ref, irrepname_run in zip(irrep_ref.keys(), irrep_run.keys()):
            assert irrepname_ref == irrepname_run  # compare strings of irreps
            assert np.allclose(irrep_ref[irrepname_ref], irrep_run[irrepname_run])  # compare multiplicities

    # Remove output files created during run
    for test_output_file in (
            "irreps.dat",
            "irreptable-template",
            "trace.txt",
            "irrep-output.json"
    ):
        os.remove(test_output_file)

def test_espresso_write_sym():

    os.chdir(TEST_FILES_PATH / "espresso_spinor")

    command = [
        "irrep",
        "-onlysym",
        "-code=espresso",
        "-prefix=Bi",
        "-writesym",
    ]

    output = subprocess.run(command, capture_output=True, text=True)
    return_code = output.returncode
    assert return_code == 0, output.stderr

    compare_sym_files("Bi.sym", "Bi.sym.ref")
    os.remove("Bi.sym")


def test_espresso_read_sym():

    os.chdir(TEST_FILES_PATH / "espresso_spinor")

    command = [
        "irrep",
        "-onlysym",
        "-code=espresso",
        "-prefix=Bi",
        "-writesym",
        "-from_sym_file=Bi.sym.reordered.ref"
    ]

    output = subprocess.run(command, capture_output=True, text=True)
    return_code = output.returncode
    assert return_code == 0, output.stderr

    compare_sym_files("Bi.sym", "Bi.sym.reordered.ref")
    os.remove("Bi.sym")

def readfile(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if len(line)>0 and line[0] != "#"]
    return lines

def compare_sym_files(frun,fref):
    # Load generated and reference output data
    data_ref = readfile(fref)
    data_run = readfile(frun)


    # in future the order of symmetries might change so the test may break..
    for line_ref, line_run in zip(data_ref, data_run):
        a1 = np.array(line_ref.split(), dtype=float)
        a2 = np.array(line_run.split(), dtype=float)
        assert len(a1) == len(a2), f"Reference: {line_ref}, Run: {line_run}"
        assert np.allclose(a1, a2), f"Reference: {line_ref}, Run: {line_run}"

