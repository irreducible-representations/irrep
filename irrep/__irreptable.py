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


import copy
import os
import sys
import logging

import numpy as np

from .__aux import str2bool, str2list_space, str_

# using a logger to print useful information during debugging,
# set to logging.INFO to disable debug messages
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SymopTable:
    """
    Docstring to go here,
    """

    def __init__(self, line, from_user=False):
        """
        Docstring to go here.

        :param line: explanation + type of kwarg to go here.
        :param from_user: explanation + type of kwarg to go here.
        """
        if from_user:
            self.__init__from_user(line)
            return
        numbers = line.split()
        self.R = np.array(numbers[:9], dtype=int).reshape(3, 3)
        self.t = np.array(numbers[9:12], dtype=float)
        self.S = (
            np.array(numbers[12::2], dtype=float)
            * np.exp(1j * np.pi * np.array(numbers[13::2], dtype=float))
        ).reshape(2, 2)

    def __init__from_user(self, line):
        """
        Docstring to go here.

        :param line: explanation + type of kwarg to go here.
        :return:
        """
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
        Docstring to go here.

        :param spinor:
        :return:
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


class CharFunction:
    """

    """

    def __init__(self, abcde):
        """

        :param abcde:
        """
        self.abcde = copy.deepcopy(abcde)

    def __call__(self, u=0, v=0, w=0):
        """

        :param u:
        :param v:
        :param w:
        :return:
        """
        return sum(
            aaa[0]
            * np.exp(1j * np.pi * (sum(a * u for a, u in zip(aaa[1:], (1, u, v, w)))))
            for aaa in self.abcde
        )


class KPoint:
    """

    """

    def __init__(self, name=None, k=None, isym=None, line=None):
        """

        :param name:
        :param k:
        :param isym:
        :param line:
        """
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

        :param other:
        :return:
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

        :return:
        """
        return "{0} : {1}  symmetries : {2}".format(self.name, self.k, self.isym)

    def str(self):
        return "{0} : {1}  : {2}".format(
            self.name,
            " ".join(str(x) for x in self.k),
            " ".join(str(x) for x in sorted(self.isym)),
        )


class Irrep:
    """

    """

    def __init__(self, f=None, nsym_group=None, line=None, k_point=None):
        """

        :param f:
        :param nsym_group:
        :param line:
        :param k_point:
        """
        if k_point is not None:
            self.__init__user(line, k_point)
            return
        s = f.readline().split()
        logger.debug(s)
        self.k = np.array(s[:3], dtype=float)
        self.has_rkmk = True if s[3] == "1" else "0" if s[3] == 0 else None
        self.name = s[4]
        self.kpname = s[7]
        self.dim = int(s[5])
        self.nsym = int(int(s[6]) / 2)
        self.reality = int(s[8])
        self.characters = {}
        self.hasuvw = False
        for isym in range(1, nsym_group + 1):
            ism, issym = [int(x) for x in f.readline().split()]
            assert ism == isym
            logger.debug("ism,issym", ism, issym)
            if issym == 0:
                continue
            elif issym != 1:
                raise RuntimeError("issym should be 0 or 1, <{0}> found".format(issym))
            abcde = []
            hasuvw = []
            for i in range(self.dim):
                for j in range(self.dim):
                    l1, l2 = [f.readline() for k in range(2)]
                    if i != j:
                        continue  # we need only diagonal elements
                    l1 = l1.strip()
                    if l1 == "1":
                        hasuvw.append(False)
                    elif l1 == "2":
                        hasuvw.append(True)
                    else:
                        raise RuntimeError(
                            "hasuvw should be 1 or 2. <{0}> found".format(l1)
                        )
                    abcde.append(np.array(l2.split(), dtype=float))
            if any(hasuvw):
                self.hasuvw = True
            if isym <= nsym_group / 2:
                self.characters[isym] = CharFunction(abcde)
        if not self.hasuvw:
            self.characters = {
                isym: self.characters[isym]() for isym in self.characters
            }
        logger.debug("characters are:", self.characters)
        assert len(self.characters) == self.nsym

    def __init__user(self, line, k_point):
        """

        :param line:
        :param k_point:
        :return:
        """
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

    def show(self):
        """

        :return:
        """
        print(self.kpname, self.name, self.dim, self.reality)

    def str(self):
        """

        :return:
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

    """

    def __init__(self, SGnumber, spinor, fromUser=True, name=None):
        """

        :param SGnumber:
        :param spinor:
        :param fromUser:
        :param name:
        """
        if fromUser:
            self.__init__user(SGnumber, spinor, name)
            return
        self.number = SGnumber


        with open(
            os.path.dirname(os.path.realpath(__file__))
            + "/TablesIrrepsLittleGroup/TabIrrepLittle_{0}.txt".format(self.number),
            "r",
        ) as f:
            self.nsym, self.name = f.readline().split()
            self.spinor = spinor
            self.nsym = int(self.nsym)
            self.symmetries = [SymopTable(f.readline()) for i in range(self.nsym)]
            assert f.readline().strip() == "#"
            self.NK = int(f.readline())
            self.irreps = []
            try:
                while True:
                    self.irreps.append(Irrep(f=f, nsym_group=self.nsym))
                    logger.debug("irrep appended:")
                    logger.debug(self.irreps[-1].show())
                    f.readline()
            except EOFError:
                pass
            except IndexError as err:
                logger.debug(err)
                pass

        if self.spinor:
            self.irreps = [s for s in self.irreps if s.name.startswith("-")]
        else:
            self.irreps = [s for s in self.irreps if not s.name.startswith("-")]

        self.nsym = int(self.nsym / 2)
        self.symmetries = self.symmetries[0 : self.nsym]

    def show(self):
        """

        :return:
        """
        for i, s in enumerate(self.symmetries):
            print(i + 1, "\n", s.R, "\n", s.t, "\n", s.S, "\n\n")
        for irr in self.irreps:
            irr.show()

    def save4user(self, name=None):
        """

        :param name:
        :return:
        """
        if name is None:
            name = "irreptables/irreps-SG={SG}-{spinor}.dat".format(
                SG=self.number, spinor="spin" if self.spinor else "scal"
            )
        fout = open(name, "w")
        fout.write(
            "SG={SG}\n name={name} \n nsym= {nsym}\n spinor={spinor}\n".format(
                SG=self.number, name=self.name, nsym=self.nsym, spinor=self.spinor
            )
        )
        fout.write(
            "symmetries=\n"
            + "\n".join(s.str(self.spinor) for s in self.symmetries)
            + "\n\n"
        )

        kpoints = {}

        for irr in self.irreps:
            if not irr.hasuvw:
                kp = KPoint(irr.kpname, irr.k, set(irr.characters.keys()))
                if (
                    len(
                        set(
                            [0.123, 0.313, 1.123, 0.877, 0.427, 0.246, 0.687]
                        ).intersection(list(kp.k))
                    )
                    == 0
                ):
                    try:
                        assert kpoints[kp.name] == kp
                    except KeyError:
                        kpoints[kp.name] = kp

        for kp in kpoints.values():
            fout.write("\n kpoint  " + kp.str() + "\n")
            for irr in self.irreps:
                if irr.kpname == kp.name:
                    fout.write(irr.str() + "\n")
        fout.close()

    def __init__user(self, SG, spinor, name):
        """

        :param SG:
        :param spinor:
        :param name:
        :return:
        """
        self.number = SG
        self.spinor = spinor
        if name is None:
            name = "{root}/irreptables/irreps-SG={SG}-{spinor}.dat".format(
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
                assert int(l[1]) == SG
            elif l[0].lower() == "name":
                self.name = l[1]
            elif l[0].lower() == "nsym":
                self.nsym = int(l[1])
            elif l[0].lower() == "spinor":
                assert str2bool(l[1]) == self.spinor
            elif l[0].lower() == "symmetries":
                print("reading symmetries")
                self.symmetries = []
                while len(self.symmetries) < self.nsym:
                    l = lines.pop()
                    # logger.debug(l)
                    try:
                        self.symmetries.append(SymopTable(l, from_user=True))
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
