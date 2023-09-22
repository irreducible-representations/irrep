import os
import subprocess
from pathlib import Path

TEST_FILES_PATH = Path(__file__).parents[2] / "examples"

def test_vasp_scalar():

    os.chdir(TEST_FILES_PATH / "vasp_scalar")

    command = [
        "irrep",
        "-code=vasp",
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

    assert "-36.8423  |        1      | GM1+(1.0)" in stdout, stdout
    assert "-36.8127  |        1      | GM3+(1.0)" in stdout, stdout
    assert "-15.1605  |        2      | GM5-(1.0)" in stdout, stdout
    assert "-14.8319  |        1      | GM3-(1.0)" in stdout, stdout

    for test_output_file in (
            "irreps.dat",
            "irreptable-template",
            "trace.txt",
    ):
        os.remove(test_output_file)
