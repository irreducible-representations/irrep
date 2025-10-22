from irrep.spacegroup import SpaceGroup
import numpy as np

from irrep.tests.test_dmn import REF_FILES_PATH, TMP_FILES_PATH

from .conftest import REF_DATA_PATH, TMP_DATA_PATH
from irrep.utility import group_numbers


a = 4.456
c = 5.926
x = 0.24
struct_param_Te = dict(real_lattice=[[a, 0, 0],
                                     [-a / 2, a * (3)**0.5 / 2, 0],
                                     [0, 0, c]],
                       positions=[
    [x, 0, 0],
    [0, x, 1 / 3],
    [-x, -x, 2 / 3]],
    typat=[1, 1, 1],)




def test_spacegroup_Te_noTR():
    spacegroup_Te = SpaceGroup.from_cell(**struct_param_Te,
                                         include_TR=False,)

    np.savez(TMP_DATA_PATH / "spacegroup_Te_noTR.npz", **spacegroup_Te.as_dict())
    assert spacegroup_Te.size == 6
    assert spacegroup_Te.number == 152
    assert spacegroup_Te.name == "P3_121"


def test_spacegroup_Te_TR():
    spacegroup_Te = SpaceGroup.from_cell(**struct_param_Te,
                                         include_TR=True)

    np.savez(TMP_DATA_PATH / "spacegroup_Te_TR.npz", **spacegroup_Te.as_dict())
    assert spacegroup_Te.size == 12
    assert spacegroup_Te.number_str == "152.34"
    assert spacegroup_Te.name == "P3_1211'"



def test_spacegroup_Te_product_noTR():
    spacegroup = SpaceGroup( **np.load(REF_DATA_PATH / "spacegroup_Te_noTR.npz"))
    product, transl_diff, spinor_factors = spacegroup.get_product_table(get_diff=True)
    np.savez(TMP_DATA_PATH / "spacegroup_Te_noTR_product.npz", product=product, transl_diff=transl_diff)
    ref = np.load(REF_DATA_PATH / "spacegroup_Te_noTR_product.npz")
    assert np.all(product == ref["product"])
    assert np.all(transl_diff == ref["transl_diff"])
    assert np.all(spinor_factors == ref["spinor_factors"])


def test_spacegroup_Te_product_TR():
    spacegroup = SpaceGroup( **np.load(REF_DATA_PATH / "spacegroup_Te_TR.npz"))
    product, transl_diff, spinor_factors = spacegroup.get_product_table(get_diff=True)
    np.savez(TMP_FILES_PATH / "spacegroup_Te_TR_product.npz", product=product, transl_diff=transl_diff, spinor_factors=spinor_factors)
    ref = np.load(REF_FILES_PATH / "spacegroup_Te_TR_product.npz")
    assert np.all(product == ref["product"])
    assert np.all(transl_diff == ref["transl_diff"])
    assert np.all(spinor_factors == ref["spinor_factors"])


def test_group_numbers():
    lst = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
           1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    grouped = group_numbers(lst, precision=0.2)
    assert np.allclose(grouped, np.mean(lst))
    lst = [0.09, 0.1, 0.11, 0.2, 0.21, 0.22]
    grouped = group_numbers(lst, precision=0.05)
    assert np.allclose(grouped, [0.1] * 3 + [0.21] * 3)

    lst = [0.21, 0.09, 0.1, 0.2, 0.11, 0.22]
    grouped = group_numbers(lst, precision=0.05)
    assert np.allclose(grouped, [0.21, 0.1, 0.1, 0.21, 0.1, 0.21])
