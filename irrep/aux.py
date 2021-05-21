
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
##  Written by Stepan Tsirkin, University of Zurich.             #
##  e-mail: stepan.tsirkin@physik.uzh.ch                         #
##################################################################


import numpy as np
from scipy import constants

BOHR = constants.physical_constants['Bohr radius'][0] / constants.angstrom


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
    #    print ("str2list  <{0}> ".format(string))
    res = np.hstack([np.arange(*(np.array(s.split("-"), dtype=int) + np.array([0, 1])))
                     if "-" in s else np.array([int(s)]) for s in string.split()])
#    print ("str2list  <{0}> -> <{1}>".format(string,res))
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
