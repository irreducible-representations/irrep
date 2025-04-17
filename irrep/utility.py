
# ###   ###   #####  ###
# #  #  #  #  #      #  #
# ###   ###   ###    ###
# #  #  #  #  #      #
# #   # #   # #####  #


##################################################################
## This file is distributed as part of                           #
## "IrRep" code and under terms of GNU General Public license v3 #
## see LICENSE file in the                                       #
##                                                               #
##  Written by Stepan Tsirkin                                    #
##  e-mail: stepan.tsirkin@ehu.eus                               #
##################################################################


from fractions import Fraction
import warnings
import numpy as np
from scipy import constants
import fortio
import sys
from typing import Any
BOHR = constants.physical_constants['Bohr radius'][0] / constants.angstrom


class FortranFileR(fortio.FortranFile):
    '''
    Class that implements `syrte/fortio` package to parse long records
    in Abinit WFK file.

    Parameters
    ----------
    filename : str
        Path to the WFK file.
    '''

    def __init__(self, filename):

        print("Using fortio to read")

        try:  # assuming there are not subrecords
            super().__init__(filename,
			     mode='r',
			     header_dtype='uint32',
			     auto_endian=True,
			     check_file=True
			     )
            print("Long records not found in ", filename)
        except ValueError:  # there are subrecords, allow negative markers
            print(("File '{}' contains subrecords - using header_dtype='int32'"
		   .format(filename)
		  ))
            super().__init__(filename,
			     mode='r',
			     header_dtype='int32',
			     auto_endian=True,
			     check_file=True
			     )

def str2list(string):
    """
    Generate `list` from `str`, where elements are separated by '-'.
    Used when parsing parameters set in CLI.

    Parameters
    ----------
    string : str
        `str` to be parsed.

    Returns
    -------
    array

    Notes
    -----
    Ranges can be generated as part of the output `array`. For example, 
    `str2list('1,3-5,7')` will give as ouput `array([1,3,4,5,7])`.
    """
    return np.hstack([np.arange(*(np.array(s.split("-"), dtype=int) + np.array([0, 1])))
                      if "-" in s else np.array([int(s)]) for s in string.split(",")])


def compstr(string):
    """
    Convers `str` to `float` or `complex`.

    Parameters
    ----------
    string : str
        String to convert.
    Returns
    -------
    float or complex
        `float` if `string` does not have imaginary part, `complex` otherwise.
    """
    if "i" in string:
        if "+" in string:
            return float(string.split("+")[0]) + 1j * \
                float(string.split("+")[1].strip("i"))
        elif "-" in string:
            return float(string.split("-")[0]) + 1j * \
                float(string.split("-")[1].strip("i"))
    else:
        return float(string)


def str2list_space(string):
    """
    Generate `list` from `str`, where elements are separated by a space ''. 
    Used when parsing parameters set in CLI.

    Parameters
    ----------
    string : str
        `str` to be parsed.

    Returns
    -------
    array

    Notes
    -----
    Ranges can be generated as part of the output `array`. For example, 
    `str2list('1,3-5,7')` will give as ouput `array([1,3,4,5,7])`.
    """
    res = np.hstack([np.arange(*(np.array(s.split("-"), dtype=int) + np.array([0, 1])))
                     if "-" in s else np.array([int(s)]) for s in string.split()])
    return res


def str2bool(v1):
    """
    Convert `str` to `bool`.

    Parameter
    ---------
    v1 : str
        String to convert.
    
    Returns
    -------
    bool

    Raises
    ------
    RuntimeError
        `v1` does not start with 'F', 'f', 'T' nor 't'.
    """
    v = v1.lower().strip('. ')
    if v[0] == "f":
        return False
    elif v[0] == "t":
        return True
    else:
        raise RuntimeError(
            " unrecognized value of bool parameter :{0}".format(v1))


def str_(x):
    """
    Round `x` to 5 floating points and return as `str`.

    Parameters
    ----------
    x : str

    Returns
    -------
    str
    """
    return str(round(x, 5))


def is_round(A, prec=1e-14):
    """
    Returns `True` if all values in A are integers.

    Parameters
    ----------
    A : array
        `array` for which the check should be done.
    prec : float, default=1e-14 (machine precision).
        Threshold to apply.

    Returns
    -------
    bool
        `True` if all elements are integers, `False` otherwise.
    """
    return(np.linalg.norm(A - np.round(A)) < prec)

    
def short(x, nd=3):
    """
    Format `float` or `complex` number.

    Parameter
    ---------
    x : int, float or complex
        Number to format.
    nd : int, default=3
        Number of decimals.

    Returns
    -------
    str
        Formatted number, with `nd` decimals saved.
    """
    fmt = "{{0:+.{0}f}}".format(nd)
    if abs(x.imag) < 10 ** (-nd):
        return fmt.format(x.real)
    if abs(x.real) < 10 ** (-nd):
        return fmt.format(x.imag) + "j"
    return short(x.real, nd) + short(1j * x.imag)


def split(l):
    """
    Determine symbol used for assignment and split accordingly.

    Parameters
    ---------
    l : str
        Part of a line read from .win file.
    """
    if "=" in l:
        return l.split("=")
    elif ":" in l:
        return l.split(":")
    else:
        return l.split()


def format_matrix(A):
    """
    Format array to print it.

    Parameters
    ----------
    A : array
        Matrix that should be printed.

    Returns
    -------
    str
        Description of the matrix. Ready to be printed.
    """
    return "".join(
        "   ".join("{0:+5.2f} {1:+5.2f}j".format(x.real, x.imag) for x in a)
        + "\n"
        for a in A
    )


def log_message(msg, verbosity, level):
    '''
    Logger to decide if a message is printed or not

    Parameters
    ----------
    msg : str
        Message to print
    verbosity : int
        Verbosity set for the current run of the code
    level : int
        If `verbosity >= level`, the message will be printed
    '''

    if verbosity >= level:
        print(msg)


def orthogonalize(A, warning_threshold=np.inf, error_threshold=np.inf , verbosity=1,
                  debug_msg=""):
    """
    Orthogonalize a square matrix, using SVD
    
    Parameters
    ----------
    A : array( (M,M), dtype=complex)
        Matrix to orthogonalize.
    warning_threshold : float, default=np.inf
        Threshold for warning message. Is some singular values are far from 1
    error_threshold : float, default=np.inf
        Threshold for error message. Is some singular values are far from 1

    Returns
    -------
    array( (M,M), dtype=complex)
        Orthogonalized matrix
    """
    u, s, vh = np.linalg.svd(A)
    if np.any(np.abs(s - 1) > error_threshold):
        raise ValueError(f"Matrix is not orthogonal \n {A} \n {debug_msg}")
    elif np.any(np.abs(s - 1) > warning_threshold):
        log_message(f"Warning: Matrix is not orthogonal \n {A} \n {debug_msg}", verbosity, 1)
    return u @ vh

def sort_vectors(list_of_vectors):
    list_of_vectors = list(list_of_vectors)
    print (list_of_vectors)
    srt = arg_sort_vectors(list_of_vectors)
    print (list_of_vectors, srt)
    return [list_of_vectors[i] for i in srt]

def arg_sort_vectors(list_of_vectors):
    """
    Compare two vectors, 
    First, the longer vector is "larger"
    second, we go element-by-element to compare
    first compare the angle of the complex number (clockwise from the x-axis), then the absolute value
    
    Returns
    -------
    bool
        True if v1>v2, False otherwise
    """
    if all( np.all(abs(np.array(key).imag)<1e-4) for key in list_of_vectors):
        def key(x):
            return np.real(x)
    else:
        def key(x):
            return (np.angle(x)/(2*np.pi)+0.01)%1
    def serialize(x, lenmax):
        return [len(x)]+ [key(y) for y in x] + [0]*(lenmax-len(x))
    lenmax = max([len(x) for x in list_of_vectors])
    sort_key = [serialize(x, lenmax) for x in list_of_vectors]
    srt = np.lexsort(np.array(sort_key).T, axis=0)
    return srt

def get_borders(E, thresh=1e-5, cyclic=False):
    """
    Get the borders of the blocks of degenerate eigenvalues.

    Parameters
    ----------
    E : array
        Eigenvalues.
    thresh : float, default=1e-5
        Threshold for the difference between eigenvalues.
    cyclic : bool, default=False
        If `True`, the first and last eigenvalues are considered to be close. 
        (e.g. complex eigenvalues on the unit circle).

    Returns
    -------
    array(int)
        Borders of the blocks of degenerate eigenvalues.
    """
    if cyclic:
        return np.where(abs(E - np.roll(E, 1)) > thresh)[0]
    else:
        return np.hstack([
                [0],
                np.where(abs(E[1:] - E[:-1]) > thresh)[0] + 1,
                [len(E)],
            ])

def get_block_indices(E, thresh=1e-5, cyclic=False):
    """
    Get the indices of the blocks of degenerate eigenvalues.

    Parameters
    ----------
    E : array
        Eigenvalues.
    thresh : float, default=1e-5
        Threshold for the difference between eigenvalues.
    cyclic : bool, default=False
        If `True`, the first and last eigenvalues are considered to be close. 
        (e.g. complex eigenvalues on the unit circle).

    Returns
    -------
    array((N,2), dtype=int)
        Indices of the blocks of degenerate eigenvalues.
    """
    borders = get_borders(E, thresh=thresh, cyclic=cyclic)
    if cyclic:
        return np.array([borders, np.roll(borders, -1)]).T
    else:
        return np.array([borders[:-1], borders[1:]]).T

def grid_from_kpoints(kpoints, grid=None):
    """
    Given a list of kpoints in fractional coordinates, return a the size of the grid in each direction
    if some k-points are repeated, they are counted only once
    if some k-points are missing, an error is raised

    Parameters
    ----------
    kpoints : np.array((nk, ndim), dtype=float)
        list of kpoints in fractional coordinates

    Returns
    -------
    grid : tuple(int)
        size of the grid in each
    selected_kpoints : list of int
        indices of the selected kpoints

    Raises
    ------
    ValueError
        if some k-points are missing
    """
    if grid is None:
        grid = tuple(np.lcm.reduce([Fraction(k).limit_denominator(100).denominator for k in kp]) for kp in kpoints.T)
    npgrid = np.array(grid)
    print(f"mpgrid = {npgrid}, {len(kpoints)}")
    kpoints_unique = UniqueListMod1()
    selected_kpoints = []
    for i, k in enumerate(kpoints):
        if is_round(k * npgrid, prec=1e-5):
            if k not in kpoints_unique:
                kpoints_unique.append(k)
                selected_kpoints.append(i)
            else:
                warnings.warn(f"k-point {k} is repeated")
    if len(kpoints_unique) < np.prod(grid):
        raise ValueError(f"Some k-points are missing {len(kpoints_unique)}< {np.prod(grid)}")
    if len(kpoints_unique) > np.prod(grid):
        raise RuntimeError("Some k-points are taken twice - this must be a bug")
    if len(kpoints_unique) < len(kpoints):
        warnings.warn("Some k-points are not on the grid or are repeated")
    return grid, selected_kpoints

class UniqueList(list):
    """	
    A list that only allows unique elements.
    uniqueness is determined by the == operator.
    Thus, non-hashable elements are also allowed.
    unlike set, the order of elements is preserved.
    """

    def __init__(self, iterator=[], count=False):
        super().__init__()
        self.do_count = count
        if self.do_count:
            self.counts = []
        for x in iterator:
            self.append(x)

    def append(self, item, count=1):
        for j, i in enumerate(self):
            if i == item:
                if self.do_count:
                    self.counts[self.index(i)] += count
                break
        else:
            super().append(item)
            if self.do_count:
                self.counts.append(1)

    def index(self, value: Any, start=0, stop=sys.maxsize) -> int:
        for i in range(start, stop):
            if self[i] == value:
                return i
        raise ValueError(f"{value} not in list")

    def __contains__(self, item):
        for i in self:
            if i == item:
                return True
        return False

    def remove(self, value: Any, all=False) -> None:
        for i in range(len(self)):
            if self[i] == value:
                if all or not self.do_count:
                    del self[i]
                    del self.counts[i]
                else:
                    self.counts[i] -= 1
                    if self.counts[i] == 0:
                        del self[i]
                        del self.counts[i]
                return


class UniqueListMod1(UniqueList):

    def __init__(self, iterator=[], tol=1e-5):
        self.tol = tol
        self.appended_indices = []
        self.last_try_append = -1
        super().__init__(iterator)

    def append(self, item):
        self.last_try_append += 1
        for i in self:
            if all_close_mod1(i, item, tol=self.tol):
                break
        else:
            list.append(self, item)
            self.appended_indices.append(self.last_try_append)

    def __contains__(self, item):
        for i in self:
            if all_close_mod1(i, item, tol=self.tol):
                return True
        return False

    def index(self, value: Any, start=0, stop=sys.maxsize) -> int:
        stop = min(stop, len(self))
        for i in range(start, stop):
            if all_close_mod1(self[i], value):
                return i
        raise ValueError(f"{value} not in list")


def all_close_mod1(a, b, tol=1e-5):
    """check if two vectors are equal modulo 1"""
    if not np.shape(a) == () and not np.shape(b) == () and (np.shape(a) != np.shape(b)):
        return False
    diff = a - b
    return np.allclose(np.round(diff), diff, atol=tol)

def vector_pprint(vector, fmt=None):
    """
    Format an homogeneous list or array as a vector for printing

    Parameters
    ---------
    vector : array or list
        Vector to format
    fmt : str, default None
        Format of the elements. Numeric types are always sign-padded

    Returns
    -------
    str
        formatted vector string
    """
    if fmt is None:
        fmt = " .5f"
    elif "s" not in fmt:
        fmt = " " + fmt if fmt[0] != " " else fmt

    return ("[" + ("{:{fmt}} " * len(vector)) + "]").format(*vector, fmt=fmt)