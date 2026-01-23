import os
import json

def load_ebr_data(sg_number, spinor):
    '''
    Load data from file of EBRs

    Parameters
    ----------
    sg_number : int
        Number of the space group
    spinor : bool
        Whether wave functions are spinors (SOC) or not

    Returns
    -------
    dict
        EBR data
    '''

    root = os.path.dirname(__file__)
    filename = f"{sg_number}_ebrs.json"
    ebr_data = json.load(open(root + "/data/ebrs/" + filename, 'r'))
    ebr_data = ebr_data["double" if spinor else "single"]
    return ebr_data