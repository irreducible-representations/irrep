
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
    return np.hstack([np.arange(*(np.array(s.split("-"), dtype=int) + np.array([0, 1])))
                      if "-" in s else np.array([int(s)]) for s in string.split(",")])


def compstr(string):
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
    #    print ("str2list  <{0}> ".format(string))
    res = np.hstack([np.arange(*(np.array(s.split("-"), dtype=int) + np.array([0, 1])))
                     if "-" in s else np.array([int(s)]) for s in string.split()])
#    print ("str2list  <{0}> -> <{1}>".format(string,res))
    return res


def str2bool(v1):
    v = v1.lower().strip('. ')
    if v[0] == "f":
        return False
    elif v[0] == "t":
        return True
    else:
        raise RuntimeError(
            " unrecognized value of bool parameter :{0}".format(v1))


def str_(x):
    return str(round(x, 5))


def is_round(A, prec=1e-14):
    """ returns true if all values in A are integers, at least within machine precision"""
    return(np.linalg.norm(A - np.round(A)) < prec)
