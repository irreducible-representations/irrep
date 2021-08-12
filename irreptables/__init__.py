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

from ._version import __version__

import copy
import os
import sys
import logging

import numpy as np

from irrep.utility import str2bool, str2list_space, str_

# using a logger to print useful information during debugging,
# set to logging.INFO to disable debug messages
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SymopTable:
    '''
    Parses a `str` that  describes a symmetry operation of the space-group and 
    stores info about it in attributes.

    Parameters
    ----------
    line : str
        Line to be parsed, which describes a symmetry operation.

    Attributes
    ----------
    R : array, shape=(3,3)
        Rotational part, describing the transformation of basis vectors (not 
        cartesian coordinates!).
    t : array, shape=(3,)
        Direct coordinates of the translation vector.
    S : array, shape=(2,2)
        SU(2) matrix describing the transformation of spinor components.
    '''

    def __init__(self, line):
        numbers = line.split()
        self.R = np.array(numbers[:9], dtype=int).reshape(3, 3)
        self.t = np.array(numbers[9:12], dtype=float)
        if len(numbers) > 12:
            self.S = (
                np.array(numbers[12:16], dtype=float)
                * np.exp(1j * np.pi * np.array(numbers[16:20], dtype=float))
            ).reshape(2, 2)
        else:
            self.S = np.eye(2)

    def str(self, spinor=True):
        """
        Create a `str` describing the symmetry operation as implemented in the 
        files included in `IrRep`.

        Parameters
        ----------
        spinor : bool, default=True
            `True` if the matrix describing the transformation of spinor 
            components should be written.

        Returns
        -------
        str
        """
        return (
            "   ".join(" ".join(str(x) for x in r) for r in self.R)
            + "     "
            + " ".join(str_(x) for x in self.t)
            + (
                (
                    "      "
                    + "    ".join(
                        "  ".join(str_(x) for x in X)
                        for X in (
                            np.abs(self.S.reshape(-1)),
                            np.angle(self.S.reshape(-1)) / np.pi,
                        )
                    )
                )
                if spinor
                else ""
            )
        )


class KPoint:
    """
    Organizes the info about a maximal k-point and contains routines to print 
    it. This info is obtained by parsing the parameter `line` or passed 
    directly as `name`, `k` and `isym`.
    
    Parameters
    ----------
    name : str, default=None
        Label of the k-point.
    k : array, default=None
        Direct coordinates of the k-point.
    isym : array, default=None
        Indices of symmetry operations in the little co-group. Indices make 
        reference to the symmetry operations stored in the header of the file 
        and stored in `IrrepTable.symmetries`. 
    line : str, default=None
        Line to be parsed. 
    
    Attributes
    ----------
    name : str
        Label of the k-point. 
    k : array, shape=(3,) 
        Direct coordinates of the k-point.
    isym : array
        Indices of symmetry operations in the little co-group. Indices make 
        reference to the symmetry operations stored in the header of the file 
        and stored in `IrrepTable.symmetries`. 
    """

    def __init__(self, name=None, k=None, isym=None, line=None):

        if line is not None:
            line_ = line.split(":")
            if line_[0].split()[0] != "kpoint":
                raise ValueError
            self.name = line_[0].split()[1]
            self.k = np.array(line_[1].split(), dtype=float)
            self.isym = str2list_space(
                line_[2]
            )  # [ int(x) for x in line_[2].split() ]  #
        else:
            self.name = name
            self.k = k
            self.isym = isym

    def __eq__(self, other):
        """
        Compares the attributes of this class with those of class instance 
        `other`.

        Parameters
        ----------
        other : class
            Instance of class `KPoint`.

        Returns
        -------
        bool
            `True` if all attributes have identical value, `False` otherwise.
        """
        if self.name != other.name:
            return False
        if np.linalg.norm(self.k - other.k) > 1e-8:
            return False
        if self.isym != other.isym:
            return False
        return True

    def show(self):
        """
        Create a `str` containing the values of all attributes.

        Returns
        -------
        str
            Line showing the values of all attributes.
        """
        return "{0} : {1}  symmetries : {2}".format(self.name, self.k, self.isym)

    def str(self):
        '''
        Create a `str` containing the values of all attributes.

        Returns
        -------
        str
            Line that, when parsed, would lead to an instance of class `KPoint` 
            with identical values of attributes.
        '''
        return "{0} : {1}  : {2}".format(
            self.name,
            " ".join(str(x) for x in self.k),
            " ".join(str(x) for x in sorted(self.isym)),
        )


class Irrep:
    """
    Parses the line containing the description of the irrep and stores the info 
    in its attributes. Contains methods print descriptions of the irrep. 

    Parameters
    ----------
    f : file object, default=None 
        It corresponds to the file containing the info about the space-group 
        and its irreps.
    nsym_group : int, default=None
        Number of symmetry operations in the "point-group" of the space-group.
    line : str, default=None
        Line with the description of an irrep, read from the file containing 
        info about the space-group and irreps.
    k_point : class instance, default=None
        Instance of class `KPoint`.

    Attributes
    ----------
    k : array, shape=(3,) 
        Direct coordinates of a k-point.
    kpname : str
        It is the label of a k-point.
    name : str
        Label of the irrep.
    dim : int
        Dimension of the irrep.
    nsym : int
        Number of symmetry operations in the little co-group of the k-point.
    reality : bool
        `True` if traces of all symmetry operations are real, `False` 
        otherwise.
    characters : dict
        Each key is the index of a symmetry operation in the little co-group 
        and the corresponding value is the trace of that symmetry in the irrep.
    """

    def __init__(self, line, k_point):
        logger.debug("reading irrep line <{0}> for KP=<{1}> ".format(line, k_point.str()))
        self.k = k_point.k
        self.kpname = k_point.name
        line = line.split()
        self.name = line[0]
        self.dim = int(line[1])
        self.nsym = len(k_point.isym)
        self.reality = len(line[2:]) == self.nsym
        ch = np.array(line[2 : 2 + self.nsym], dtype=float)
        if not self.reality:
            ch = ch * np.exp(
                1.0j
                * np.pi
                * np.array(line[2 + self.nsym : 2 + 2 * self.nsym], dtype=float)
            )
        self.characters = {k_point.isym[i]: ch[i] for i in range(self.nsym)}
        logger.debug("the irrep {0}  ch= {1}".format(self.name, self.characters))
        assert len(self.characters) == self.nsym

    def show(self):
        """
        Print label of the k-point and info about the irrep.
        """
        print(self.kpname, self.name, self.dim, self.reality)

    def str(self):
        """
        Generate a line describing the irrep and its character.

        Returns
        -------
        str
            Line describing the irrep, as it is written in the table of the
            space-group included in `IrRep`. This line contains the label, 
            dimension and character of the irrep.
        """
        logger.debug(self.characters)
        ch = np.array([self.characters[isym] for isym in sorted(self.characters)])
        if np.abs(np.imag(ch)).max() > 1e-6:
            str_ch = "   " + "  ".join(str_(x) for x in np.abs(ch))
            str_ch += "   " + "  ".join(str_(x) for x in np.angle(ch) / np.pi)
        else:
            str_ch = "   " + "  ".join(str_(x) for x in np.real(ch))
        return self.name + " {} ".format(self.dim) + str_ch


class IrrepTable:
    """
    Parse file corresponding to a space-group, storing the info in attributes. 
    Also contains methods to print and write this info in a file.

    Parameters
    ----------
    SGnumber : int
        Number of the space-group.
    spinor : bool
        `True` if the matrix describing the transformation of spinor components 
        should be read.
    name : str, default=None
        Name of the file from which info about the space-group and irreps 
        should be read. If `None`, the code will try to open a file already 
        included in it.

    Attributes
    ----------
    number : int
        Number of the space-group.
    name : str
        Symbol of the space-group in Hermann-Mauguin notation. 
    spinor : bool
        `True` if wave-functions are spinors (SOC), `False` if they are scalars.
    nsym : int
       Number of symmetry operations in the "point-group" of the space-group. 
    symmetries : list
        Each component is an instance of class `SymopTable` corresponding to a 
        symmetry operation in the "point-group" of the space-group.
    NK : int
        Number of maximal k-points in the Brillouin zone.
    irreps : list
        Each component is an instance of class `IrRep` corresponding to an 
        irrep of the little group of a maximal k-point.
    """

    def __init__(self, SGnumber, spinor, name=None):
        self.number = SGnumber
        self.spinor = spinor
        if name is None:
            name = "{root}/tables/irreps-SG={SG}-{spinor}.dat".format(
                SG=self.number,
                spinor="spin" if self.spinor else "scal",
                root=os.path.dirname(__file__),
            )
            logger.debug("reading from a standard irrep table <{0}>".format(name))
        else:
            logger.debug("reading from a user-defined irrep table <{0}>".format(name))

        lines = open(name).readlines()[-1::-1]
        while len(lines) > 0:
            l = lines.pop().strip().split("=")
            # logger.debug(l,l[0].lower())
            if l[0].lower() == "SG":
                assert int(l[1]) == self.number
            elif l[0].lower() == "name":
                self.name = l[1]
            elif l[0].lower() == "nsym":
                self.nsym = int(l[1])
            elif l[0].lower() == "spinor":
                assert str2bool(l[1]) == self.spinor
            elif l[0].lower() == "symmetries":
                print("Reading symmetries from tables")
                self.symmetries = []
                while len(self.symmetries) < self.nsym:
                    l = lines.pop()
                    # logger.debug(l)
                    try:
                        self.symmetries.append(SymopTable(l))
                    except Exception as err:
                        logger.debug(err)
                        pass
                break

        logger.debug("symmetries are:\n" + "\n".join(s.str() for s in self.symmetries))

        self.irreps = []
        while len(lines) > 0:
            l = lines.pop().strip()
            try:
                kp = KPoint(line=l)
                logger.debug("kpoint successfully read:", kp.str())
            except Exception as err:
                logger.debug("error while reading k-point <{0}>".format(l), err)
                try:
                    self.irreps.append(Irrep(line=l, k_point=kp))
                except Exception as err:
                    logger.debug("error while reading irrep <{0}>".format(l), err)
                    pass

    def show(self):
        '''
        Print info about symmetry operations and irreps.  
        '''
        for i, s in enumerate(self.symmetries):
            print(i + 1, "\n", s.R, "\n", s.t, "\n", s.S, "\n\n")
        for irr in self.irreps:
            irr.show()

