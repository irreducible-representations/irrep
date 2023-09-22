import os
import subprocess
from pathlib import Path

TEST_FILES_PATH = Path(__file__).parents[2] / "examples"

def test_wannier_scalar_example():

    os.chdir(TEST_FILES_PATH / "wannier_scalar")

    command = [
        "irrep",
        "-code=wannier90",
        "-prefix=wannier90",
        "-kpoints=1,6",
        "-Ecut=50",
        "-IBend=8",
        "-kpnames=GM,M"
    ]
    output = subprocess.run(command, capture_output=True, text=True)

    return_code = output.returncode
    stdout = output.stdout

    with open("test_out", "w") as f:
        f.write(stdout)

    assert return_code == 0, output.stderr

    assert "|        1      | GM1+(1.0)  |" in stdout, stdout
    assert "|        1      | GM1+(1.0)  |" in stdout, stdout
    assert "|        1      | M1+(1.0) |" in stdout, stdout
    assert "|        1      | M4+(1.0) |" in stdout, stdout


    for test_output_file in (
        "irreps.dat",
        "irreptable-template",
        "trace.txt",
    ):
        os.remove(test_output_file)
