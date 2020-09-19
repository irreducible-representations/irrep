import os
import subprocess
from pathlib import Path

TEST_FILES_PATH = Path(__file__).parents[2] / "examples"


def test_Bi_scalar_example():

    os.chdir(TEST_FILES_PATH / "Bi-scalar")

    command = [
        "irrep",
        "-Ecut=50",
        "-code=abinit",
        "-fWFK=Bi_WFK",
        "-refUC=0,-1,1,1,0,-1,-1,-1,-1",
        "-kpoints=11",
        "-IBend=5",
        "-kpnames=GM",
    ]

    for test_output_file in (
        "bands-.dat",
        "bands-sym-.dat",
        "irreps.dat",
        "irreptable-template",
        "trace.txt",
    ):
        os.remove(test_output_file)

    output = subprocess.run(command, capture_output=True, text=True)

    with open("out") as f:
        reference_out = f.read()

    assert output.returncode == 0
    assert output.stdout.strip() == reference_out.strip()
