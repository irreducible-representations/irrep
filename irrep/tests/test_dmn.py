import pickle
from pathlib import Path
import numpy as np
from irrep.bandstructure import BandStructure

TEST_FILES_PATH = Path(__file__).parents[2] / "examples"
REF_FILES_PATH = Path(__file__).parent / "ref_data"
TMP_FILES_PATH = Path(__file__).parent / "tmp_data"

def check_Fe_qe(include_TR):
    path = TEST_FILES_PATH / "Fe_qe"

    bandstructure = BandStructure(prefix=str(path/"Fe"),
                                   code="espresso",
                                   degen_thresh=1e-3,
                                   Ecut=50.,
                                   magmom=[[0,0,1]],
                                   include_TR=include_TR)
    # bandstructure.spacegroup.show()
                                   
    data = bandstructure.get_dmn(degen_thresh=1e-3)
    print (f"number of symmetries: {bandstructure.spacegroup.order}")
    for a in data["d_band_blocks"][:1]:
        for b in a:
            for c in b:
                assert np.allclose(c.dot(c.T.conj()), np.eye(c.shape[0])), f"block is not unitary {c}"
    
    fname = f"Fe_qe_dmn_TR={include_TR}.pkl"
    pickle.dump(data, open(TMP_FILES_PATH/fname, "wb"))
    
    data_ref = pickle.load(open(REF_FILES_PATH/fname, "rb"))
    for k in data_ref.keys():
        compare_nested_lists(data[k], data_ref[k], key=k)

# def test_Fe_qe_TR():
#     check_Fe_qe(include_TR=True)

def test_Fe_qe_noTR():
    check_Fe_qe(include_TR=False)   

def compare_nested_lists(a, b, key, depth=0):
    if depth > 3:
        raise ValueError("Too deep, depth={depth}>3")
    try:
        assert np.allclose(a,b), f"Failed for {key} depth={depth}"
    except ValueError:
        for c,d in zip(a,b):
            compare_nested_lists(c,d,key=key, depth=depth+1)