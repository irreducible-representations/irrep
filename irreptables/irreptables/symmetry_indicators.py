import os
import json

def load_si_table(spinor, magnetic):
    '''
    Load table of symmetry indicators

    Returns
    -------
    dict
        Data loaded from the file of symmetry indicators
    '''

    root = os.path.dirname(__file__)
    filename = (
        f"{'double' if spinor else 'single'}_indicators"
        f"{'_magnetic' if magnetic else ''}.json"
    )
    si_table = json.load(open(root + "/data/symmetry_indicators/" + filename, 'r'))
    return si_table