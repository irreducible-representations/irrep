import os
import subprocess
from pathlib import Path
from monty.serialization import loadfn
import numpy as np

TEST_FILES_PATH = Path(__file__).parents[2] / "examples"


def test_ebr_decomposition_stable_topology():

    os.chdir(TEST_FILES_PATH / "Bi-hoti")

    command = [
        "irrep",
        "-spinor",
        "-kpnames=T,GM,F,L",
        "-Ecut=100",
        "-IBend=10",
        "--time-reversal",
        "--ebr-decomposition"
    ]
    output = subprocess.run(command, capture_output=True, text=True)
    return_code = output.returncode
    assert return_code == 0, output.stderr

    # Load generated and reference output data
    data_ref = loadfn("ref_output_ebrs.json")
    data_ref = data_ref["characters and irreps"][0]["subspace"]
    data_run = loadfn("irrep-output.json")
    data_run = data_run["characters and irreps"][0]["subspace"]

    assert data_run["classification"] == data_ref["classification"]
    ref_ebrs = data_ref["ebr decomposition"]
    run_ebrs = data_run["ebr decomposition"]
    assert np.allclose(run_ebrs["y"], ref_ebrs["y"])
    assert np.allclose(run_ebrs["y_prime"], ref_ebrs["y_prime"])
    if ref_ebrs["solutions"] is None:
        assert run_ebrs["solutions"] is None
    else:
        assert np.allclose(run_ebrs["solutions"], ref_ebrs["solutions"])

    # Remove output files created during run
    for test_output_file in (
            "irreps.dat",
            "irreptable-template",
            "trace.txt",
            "irrep-output.json"
    ):
        os.remove(test_output_file)
