import os
import subprocess
from pathlib import Path

TEST_FILES_PATH = Path(__file__).parents[2] / "examples"

def test_vasp_spinor():

    os.chdir(TEST_FILES_PATH / "vasp_spinor")

    command = [
        "irrep",
        "-code=vasp",
        "-spinor",
        "-kpnames=GM",
        "-kpoints=1",
        "-Ecut=50",
    ]
    output = subprocess.run(command, capture_output=True, text=True)

    return_code = output.returncode
    stdout = output.stdout

    with open("test_out", "w") as f:
        f.write(stdout)

    assert return_code == 0, output.stderr

    assert "-36.9423  |        2      | -GM7(1.0)" in stdout, stdout
    assert "-36.9129  |        2      | -GM7(1.0)" in stdout, stdout
    assert "-17.6854  |        2      | -GM9(1.0)" in stdout, stdout
    assert "-17.5432  |        2      | -GM9(1.0)" in stdout, stdout

    for test_output_file in (
            "irreps.dat",
            "irreptable-template",
            "trace.txt",
    ):
        os.remove(test_output_file)
