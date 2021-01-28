import os
import subprocess
from pathlib import Path

TEST_FILES_PATH = Path(__file__).parents[2] / "examples"


def test_bi_scalar_example():

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

    output = subprocess.run(command, capture_output=True, text=True)

    return_code = output.returncode
    stdout = output.stdout

    with open("test_out", "w") as f:
        f.write(stdout)

    assert return_code == 0

    assert "GM1+(1.0)" in stdout
    assert "GM2-(1.0)" in stdout
    assert "GM1+(1.0)" in stdout
    assert "GM3+(1.0)" in stdout
    assert "number of inversions-odd states :  1" in stdout

    for test_output_file in (
        "irreps.dat",
        "irreptable-template",
        "trace.txt",
    ):
        os.remove(test_output_file)


def test_bi_spinor_example():

    os.chdir(TEST_FILES_PATH / "Bi-spinor")

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

    output = subprocess.run(command, capture_output=True, text=True)

    with open("out") as f:
        reference_out = f.read()

    return_code = output.returncode
    stdout = output.stdout

    assert return_code == 0

    assert "-GM8(1.0)" in stdout
    assert "-GM9(1.0)" in stdout
    assert "-GM8(0.5)" in stdout
    assert "number of inversions-odd Kramers pairs :  1" in stdout

    for test_output_file in (
        "irreps.dat",
        "irreptable-template",
        "trace.txt",
    ):
        os.remove(test_output_file)


def test_wannier_spin_example():

    os.chdir(TEST_FILES_PATH / "wannier-spin")

    command = [
        "irrep",
        "-code=wannier90",
        "-prefix=PbTe",
        "-kpoints=1,33,37,419",
        "-Ecut=30",
        "-kpnames=GM,L,X,W",
        "-refUC=-1,+1,-1,-1,+1,+1,+1,+1,-1"
    ]
    output = subprocess.run(command, capture_output=True, text=True)

    return_code = output.returncode
    stdout = output.stdout

    with open("test_out", "w") as f:
        f.write(stdout)

    assert return_code == 0

    known_output = """k-point   1 :[0. 0. 0.] 
 number of irreps = 2
   Energy  | multiplicity |   irreps   | sym. operations  
           |              |            |     1        2        3        4        5        6        7        8        9       10       11       12       13       14       15       16       17       18       19       20       21       22       23       24       25       26       27       28       29       30       31       32       33       34       35       36       37       38       39       40       41       42       43       44       45       46       47       48    
   5.2656  |        2     | -GM8(1.0)  |  2.0000  -2.0000   1.4142  -1.4142   0.0000   0.0000   1.4142  -1.4142   0.0000   0.0000  -0.0000   0.0000  -0.0000   0.0000  -0.0000   0.0000   1.0000  -1.0000  -0.0000   0.0000   1.0000  -1.0000   1.4142  -1.4142   1.0000  -1.0000  -0.0000   0.0000   1.0000  -1.0000   1.4142  -1.4142   1.0000  -1.0000   1.4142  -1.4142   1.0000  -1.0000  -0.0000   0.0000   1.0000  -1.0000  -0.0000   0.0000   1.0000  -1.0000   1.4142  -1.4142
   6.3895  |        4     | -GM11(1.0) |  4.0000  -4.0000  -0.0000   0.0000   0.0000   0.0000  -0.0000   0.0000  -0.0000  -0.0000  -0.0000   0.0000   0.0000  -0.0000  -0.0000   0.0000  -1.0000   1.0000  -0.0000   0.0000  -1.0000   1.0000  -0.0000   0.0000  -1.0000   1.0000  -0.0000   0.0000  -1.0000   1.0000  -0.0000   0.0000  -1.0000   1.0000   0.0000  -0.0000  -1.0000   1.0000  -0.0000   0.0000  -1.0000   1.0000  -0.0000   0.0000  -1.0000   1.0000   0.0000  -0.0000
"""

    assert known_output in stdout

    for test_output_file in (
        "irreps.dat",
        "irreptable-template",
        "trace.txt",
    ):
        os.remove(test_output_file)
