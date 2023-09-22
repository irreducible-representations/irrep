import os
import subprocess
from pathlib import Path

TEST_FILES_PATH = Path(__file__).parents[2] / "examples"

def test_abinit_scalar_example():

    os.chdir(TEST_FILES_PATH / "abinit_scalar")
    command = [
        "irrep",
        "-Ecut=50",
        "-code=abinit",
        "-fWFK=O_DS2_WFK",
        "-refUC=0,-1,1,1,0,-1,-1,-1,-1",
        "-kpoints=1",
        "-IBstart=11",
        "-IBend=15",
        "-kpnames=GM",
    ]

    output = subprocess.run(command, capture_output=True, text=True)

    return_code = output.returncode
    stdout = output.stdout

    with open("test_out", "w") as f:
        f.write(stdout)

    assert return_code == 0, output.stderr
    assert "GM1+(1.0)" in stdout, stdout
    assert "GM2-(1.0)" in stdout, stdout
    assert "GM1+(1.0)" in stdout, stdout
    assert "GM3+(1.0)" in stdout, stdout
    
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
    stdout = output.stdout

    with open("test_out", "w") as f:
        f.write(stdout)

    assert return_code == 0, output.stderr
    assert "GM1+(1.0)" in stdout, stdout
    assert "GM2-(1.0)" in stdout, stdout
    assert "GM1+(1.0)" in stdout, stdout
    assert "GM3+(1.0)" in stdout, stdout

    for test_output_file in (
        "irreps.dat",
        "irreptable-template",
        "trace.txt",
    ):
        os.remove(test_output_file)
