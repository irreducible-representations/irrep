import os
import subprocess
from pathlib import Path

TEST_FILES_PATH = Path(__file__).parents[2] / "examples"

def test_bi_hoti():

    os.chdir(TEST_FILES_PATH / "Bi-hoti")

    # Test specifying refUC in CLI
    command = [
        "irrep",
        "-spinor",
        "-code=vasp",
        "-kpnames=T,GM,F,L",
        "-Ecut=50",
        "-refUC=1,-1,0,0,1,-1,1,1,1",
        "-EF=5.2156",
        "-IBstart=5",
        "-IBend=10"
    ]
    output = subprocess.run(command, capture_output=True, text=True)

    return_code = output.returncode
    stdout = output.stdout

    with open("test_out", "w") as f:
        f.write(stdout)

    assert return_code == 0, output.stderr
    assert "-2.7306  |        2      | -GM8(1.0)" in stdout, stdout
    assert "-0.7762  |        2      | -GM8(1.0)" in stdout, stdout
    assert "-0.4961  |        2      | -GM4(1.0), -GM5(1.0)" in stdout, stdout
    assert "-4.8263  |        2      | -F3(1.0), -F4(1.0)" in stdout, stdout
    assert "-3.6784  |        2      | -F5(1.0), -F6(1.0)" in stdout, stdout
    assert "-2.4303  |        2      | -F5(1.0), -F6(1.0)" in stdout, stdout
    assert "-1.7054  |        2      | -L5(1.0), -L6(1.0)" in stdout, stdout
    assert "-1.6885  |        2      | -L3(1.0), -L4(1.0)" in stdout, stdout
    assert "-0.1312  |        2      | -L5(1.0), -L6(1.0)" in stdout, stdout
    assert "-1.5597  |        2      | -T9(1.0)" in stdout, stdout
    assert "-1.2220  |        2      | -T8(1.0)" in stdout, stdout
    assert "0.1460  |        2      | -T6(1.0), -T7(1.0)" in stdout, stdout

    # Test without specifying refUC in CLI
    command = [
        "irrep",
        "-spinor",
        "-code=vasp",
        "-kpnames=T,GM,F,L",
        "-Ecut=50",
        "-EF=5.2156",
        "-IBstart=5",
        "-IBend=10"
    ]
    output = subprocess.run(command, capture_output=True, text=True)

    return_code = output.returncode
    stdout = output.stdout

    with open("test_out", "w") as f:
        f.write(stdout)

    assert return_code == 0, output.stderr
    assert "-2.7306  |        2      | -GM8(1.0)" in stdout, stdout
    assert "-0.7762  |        2      | -GM8(1.0)" in stdout, stdout
    assert "-0.4961  |        2      | -GM4(1.0), -GM5(1.0)" in stdout, stdout
    assert "-4.8263  |        2      | -F3(1.0), -F4(1.0)" in stdout, stdout
    assert "-3.6784  |        2      | -F5(1.0), -F6(1.0)" in stdout, stdout
    assert "-2.4303  |        2      | -F5(1.0), -F6(1.0)" in stdout, stdout
    assert "-1.7054  |        2      | -L5(1.0), -L6(1.0)" in stdout, stdout
    assert "-1.6885  |        2      | -L3(1.0), -L4(1.0)" in stdout, stdout
    assert "-0.1312  |        2      | -L5(1.0), -L6(1.0)" in stdout, stdout
    assert "-1.5597  |        2      | -T9(1.0)" in stdout, stdout
    assert "-1.2220  |        2      | -T8(1.0)" in stdout, stdout
    assert "0.1460  |        2      | -T6(1.0), -T7(1.0)" in stdout, stdout

    for test_output_file in (
            "irreps.dat",
            "irreptable-template",
            "trace.txt",
    ):
        os.remove(test_output_file)
