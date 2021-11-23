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


def test_bi_spinor_example():

    os.chdir(TEST_FILES_PATH / "Bi-spinor")

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

def test_wannier_scalar_example():

    os.chdir(TEST_FILES_PATH / "wannier-scalar-NaAs")

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

def test_wannier_spin_example():

    os.chdir(TEST_FILES_PATH / "wannier-spinors-NaAs")

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

#def test_wannier_spin_example():
#
#    os.chdir(TEST_FILES_PATH / "wannier-spin")
#
#    command = [
#        "irrep",
#        "-code=wannier90",
#        "-prefix=PbTe",
#        "-kpoints=1,33,37,419",
#        "-Ecut=30",
#        "-kpnames=GM,L,X,W",
#        "-refUC=-1,+1,-1,-1,+1,+1,+1,+1,-1"
#    ]
#    output = subprocess.run(command, capture_output=True, text=True)
#
#    return_code = output.returncode
#    stdout = output.stdout
#
#    with open("test_out", "w") as f:
#        f.write(stdout)
#
#    assert return_code == 0, output.stderr
#
#    assert "5.2656  |        2     | -GM8(1.0)" in stdout, stdout
#    assert "6.3895  |        4     | -GM11(1.0)" in stdout, stdout
#
#    for test_output_file in (
#        "irreps.dat",
#        "irreptable-template",
#        "trace.txt",
#    ):
#        os.remove(test_output_file)


def test_bi_hoti():

    os.chdir(TEST_FILES_PATH / "Bi-hoti")

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


def test_C2B2Gd_vasp_scalar():

    os.chdir(TEST_FILES_PATH / "C2B2Gd-vasp-scalar")

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


def test_C2B2Gd_vasp_spinor():

    os.chdir(TEST_FILES_PATH / "C2B2Gd-vasp-spinor")

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
