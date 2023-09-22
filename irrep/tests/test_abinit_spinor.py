import os
import subprocess
from pathlib import Path

TEST_FILES_PATH = Path(__file__).parents[2] / "examples"

def test_abinit_spinor_example():

    os.chdir(TEST_FILES_PATH / "abinit_spinor")

    command = [
        "irrep",
        "-spinor",
        "-Ecut=50",
        "-code=abinit",
        "-fWFK=O_DS2_WFK",
        "-refUC=0,-1,1,1,0,-1,-1,-1,-1",
        "-kpoints=1",
        "-IBend=21",
        "-IBend=32",
        "-kpnames=GM",
    ]

    output = subprocess.run(command, capture_output=True, text=True)

    with open("out") as f:
        reference_out = f.read()

    return_code = output.returncode
    stdout = output.stdout

    assert return_code == 0, output.stderr

    assert "-GM8(1.0)" in stdout, stdout
    assert "-GM9(1.0)" in stdout, stdout

    command = [
        "irrep",
        "-spinor",
        "-Ecut=50",
        "-code=abinit",
        "-fWFK=O_DS2_WFK",
        "-kpoints=1",
        "-IBend=21",
        "-IBend=32",
        "-kpnames=GM",
    ]

    output = subprocess.run(command, capture_output=True, text=True)

    with open("out") as f:
        reference_out = f.read()

    return_code = output.returncode
    stdout = output.stdout

    assert return_code == 0, output.stderr

    assert "-GM8(1.0)" in stdout, stdout
    assert "-GM9(1.0)" in stdout, stdout

    for test_output_file in (
        "irreps.dat",
        "irreptable-template",
        "trace.txt",
    ):
        os.remove(test_output_file)
