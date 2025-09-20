from pathlib import Path
import os

TEST_FILES_PATH = Path(__file__).parents[2] / "examples"
REF_DATA_PATH = Path(__file__).parents[0] / "ref_data"
TMP_DATA_PATH = Path(__file__).parents[0] / "tmp_data"
if not os.path.exists(TMP_DATA_PATH):
    os.makedirs(TMP_DATA_PATH)
