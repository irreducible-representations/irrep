import os
import subprocess
from pathlib import Path
from monty.serialization import loadfn
import numpy as np

TEST_FILES_PATH = Path(__file__).parents[2] / "examples"

def test_symmetry_indicators_stable_topology():

    os.chdir(TEST_FILES_PATH / "Bi-hoti")

    # Test specifying refUC in CLI
    command = [
        "irrep",
        "-spinor",
        "-kpnames=T,GM,F,L",
        "-Ecut=100",
        "-IBend=10",
        "--symmetry-indicators"
    ]
    output = subprocess.run(command, capture_output=True, text=True)
    return_code = output.returncode
    assert return_code == 0, output.stderr

    # Load generated and reference output data
    data_ref = loadfn("ref_output_indicators.json")
    data_ref = data_ref["characters and irreps"][0]["subspace"]
    data_ref = data_ref["symmetry indicators"]
    data_run = loadfn("irrep-output.json")
    data_run = data_run["characters and irreps"][0]["subspace"]
    data_run = data_run["symmetry indicators"]

    for indicator_run, indicator_ref in zip(data_run, data_ref):
        assert indicator_run == indicator_ref
        assert data_run[indicator_run] == data_ref[indicator_ref]

    # Remove output files created during run
    for test_output_file in (
            "irreps.dat",
            "irreptable-template",
            "trace.txt",
            "irrep-output.json"
    ):
        os.remove(test_output_file)
