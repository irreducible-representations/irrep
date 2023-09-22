import os
import subprocess
from pathlib import Path

TEST_FILES_PATH = Path(__file__).parents[2] / "examples"

def test_wannier_spin_example():

    os.chdir(TEST_FILES_PATH / "wannier_spinor")

    command = [
        "irrep",
        "-code=wannier90",
        "-prefix=NaAs",
        "-kpoints=1,8",
        "-Ecut=50",
        "-kpnames=GM,A"
    ]
    output = subprocess.run(command, capture_output=True, text=True)

    return_code = output.returncode
    stdout = output.stdout

    with open("test_out", "w") as f:
        f.write(stdout)

    assert return_code == 0, output.stderr

    assert "|        2      | -GM7(1.0)  |" in stdout, stdout
    assert "|        2      | -GM9(1.0)  |" in stdout, stdout
    assert "|        2      | -GM9(1.0)  |" in stdout, stdout
    assert "|        2      | -A9(1.0) |" in stdout, stdout
    assert "|        2      | -A8(1.0) |" in stdout, stdout
    assert "|        2      | -A6(1.0) |" in stdout, stdout


    for test_output_file in (
        "irreps.dat",
        "irreptable-template",
        "trace.txt",
    ):
        os.remove(test_output_file)
