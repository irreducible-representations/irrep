from packaging.version import parse, Version
import gpaw
from pathlib import Path
import os

TEST_FILES_PATH = Path(__file__).parents[2] / "examples"
REF_DATA_PATH = Path(__file__).parents[0] / "ref_data"
TMP_DATA_PATH = Path(__file__).parents[0] / "tmp_data"
if not os.path.exists(TMP_DATA_PATH):
    os.makedirs(TMP_DATA_PATH)


if parse(gpaw.__version__) >= Version("26.0.0"):
    GPAW_VERSION = 26
else:
    GPAW_VERSION = 25


CALC_TYPE_LIST = ["OLD"]
if GPAW_VERSION == 26:
    CALC_TYPE_LIST += ["NEW", "LEGACY"]
