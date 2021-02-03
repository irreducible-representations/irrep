
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
from math import pi
from scipy.linalg import expm
import spglib
from irreptables import IrrepTable
from scipy.optimize import minimize
from .__aux import str_

pauli_sigma = np.array(
    [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


class WrongRotation(RuntimeError):
    def __init__(self, rotation):
        super(WrongRotation, self).__init__(
            "strange rotation :\n {0}".format(rotation))


class SymmetryOperation():

    def __init__(self, rot, trans, Lattice, ind=-1, spinor=True):
        self.ind = ind
        self.rotation = rot
        self.Lattice = Lattice
        self.translation = trans % 1
        self.translation[1 - self.translation < 1e-5] = 0
        self.axis, self.angle, self.inversion = self._get_operation_type()
        iangle = (round(self.angle / pi * 6) + 6) % 12 - 6
        if iangle == -6:
            iangle = 6
        self.angle = iangle * pi / 6
        self.angle_str = self.get_angle_str()
        self.spinor = spinor
        self.spinor_rotation = expm(-0.5j * self.angle *
                                    np.einsum('i,ijk->jk', self.axis, pauli_sigma))

    def get_angle_str(self):
        accur = 1e-4
        def is_close_int(x): return abs((x + 0.5) % 1 - 0.5) < accur
        api = self.angle / np.pi
        if abs(api) < 0.01:
            return " 0 "
        for n in 1, 2, 3, 4, 6:
            if is_close_int(api * n):
                return "{0:.0f}{1} pi".format(
                    round(api * n), "" if n == 1 else "/" + str(n))
        raise RuntimeError(
            "{0} pi rotation cannot be in the space group".format(api))

    def _get_operation_type(self):
        rotxyz = self.Lattice.T.dot(
            self.rotation).dot(
            np.linalg.inv(
                self.Lattice).T)
#        print ("rotation in real space:\n",rotxyz)
        E, V = np.linalg.eig(rotxyz)
        if not np.isclose(abs(E), 1).all():
            raise RuntimeError(
                "some eigenvalues of the rotation are not unitary")
        if E.prod() < 0:
            inversion = True
            E *= -1
        else:
            inversion = False
        idx = np.argsort(E.real)
        E = E[idx]
        V = V[:, idx]
        axis = V[:, 2].real
        if np.isclose(E[:2], 1).all():
            angle = 0
        elif np.isclose(E[:2], -1).all():
            angle = np.pi
        else:
            angle = np.angle(E[0])
            v = V[:, 0]
            s = np.real(np.linalg.det([v, v.conj(), axis]) / 1.j)
            if np.isclose(s, -1):
                angle = 2 * np.pi - angle
            elif not np.isclose(s, 1):
                raise RuntimeError("the sign of rotation should be +-1")
        return (axis, angle, inversion)

    def rotation_refUC(self, refUC):
        R = np.linalg.inv(refUC).T.dot(self.rotation).dot(refUC.T)
        R1 = np.array(R.round(), dtype=int)
        if (abs(R - R1).max() > 1e-6):
            raise RuntimeError(
                "the rotation in the reference UC is not integer. Is that OK? \n{0}".format(R))
        return R1

    def translation_refUC(self, refUC, shiftUC):
        return (
            self.translation +
            shiftUC -
            self.rotation.dot(shiftUC)).dot(
            np.linalg.inv(refUC))

    def show(self, refUC=None, shiftUC=np.zeros(3)):
        # refUC - row-vectors, expressing the reference unit cell vectors in
        # terms of the lattice used in calculation
        print(" # ", self.ind)
        rotstr = [s +
                  " ".join("{0:3d}".format(x) for x in row) +
                  t for s, row, t in zip(["rotation : |", " " *
                                          11 +
                                          "|", " " *
                                          11 +
                                          "|"], self.rotation, [" |", " |", " |"])]
        if (refUC is not None) or ((shiftUC is not None)
                                   and np.linalg.norm(shiftUC) > 1e-5):
            if refUC is None:
                refUC = np.eye(3)
            fstr = ("{0:3d}")
            R = self.rotation_refUC(refUC)
            rotstr1 = [" " *
                       5 +
                       s +
                       " ".join(fstr.format(x) for x in row) +
                       t for s, row, t in zip(["in refUC : |", " " *
                                               11 +
                                               "|", " " *
                                               11 +
                                               "|"], R, [" |", " |", " |"])]
            rotstr = [r + r1 for r, r1 in zip(rotstr, rotstr1)]
        print("\n".join(rotstr))

        if self.spinor:
            print("\n".join(s +
                            " ".join("{0:6.3f}{1:+6.3f}j".format(x.real, x.imag) for x in row) +
                            t for s, row, t in zip(["spinor   : |", " " *
                                                    11 +
                                                    "|", " " *
                                                    11 +
                                                    "|"], self.spinor_rotation, [" |", " |", " |"])))
        print(
            " translation : [ " +
            " ".join(
                "{0:10.6f}".format(
                    x %
                    1) for x in self.translation.round(6)) +
            " ] ")
        if refUC is not None:
            print("     in the reference unit cell :")
            print(
                "     translation : [ " +
                " ".join(
                    "{0:10.6f}".format(
                        x %
                        1) for x in self.translation_refUC(
                        refUC,
                        shiftUC).round(6)) +
                " ] ")
        print("axis: {0} ; angle = {1}, inversion : {2} ".format(
            self.axis.round(6), self.angle_str, self.inversion))

    def str(self, refUC=None, shiftUC=np.zeros(3)):
        if refUC is None:
            refUC = np.eye(3, dtype=int)
#       refUC - row-vectors, expressing the reference unit cell vectors in terms of the lattice used in calculation
#        print ( "symmetry # ",self.ind )
        R = self.rotation_refUC(refUC)
        t = self.translation_refUC(refUC, shiftUC)
#        np.savetxt(stdout,np.hstack( (R,t[:,None])),fmt="%8.5f" )
        S = self.spinor_rotation
        return ("   ".join(" ".join(str(x) for x in r) for r in R) + "     " + " ".join(str_(x) for x in t) + ("      " + \
                "    ".join("  ".join(str_(x) for x in X) for X in (np.abs(S.reshape(-1)), np.angle(S.reshape(-1)) / np.pi))))

    def str2(self, refUC=None, shiftUC=np.zeros(3)):
        if refUC is None:
            refUC = np.eye(3, dtype=int)
        if shiftUC is None:
            shiftUC = np.zeros(3, dtype=float)
# this method for Bilbao server
#       refUC - row-vectors, expressing the reference unit cell vectors in terms of the lattice used in calculation
#        print ( "symmetry # ",self.ind )
        R = self.rotation_refUC(refUC)
        t = self.translation_refUC(refUC, shiftUC)
#        np.savetxt(stdout,np.hstack( (R,t[:,None])),fmt="%8.5f" )
        S = self.spinor_rotation
        return ("   ".join(" ".join("{0:2d}".format(x) for x in r) for r in R) + "     " + " ".join("{0:10.6f}".format(x) for x in t) + (
            ("      " + "    ".join("  ".join("{0:10.6f}".format(x) for x in (X.real, X.imag)) for X in S.reshape(-1))) if S is not None else "") + "\n")


class SpaceGroup():

    def __cell_vasp(self, inPOSCAR):
        fpos = (l.strip() for l in open(inPOSCAR))
        title = next(fpos)
        lattice = float(
            next(fpos)) * np.array([next(fpos).split() for i in range(3)], dtype=float)
        try:
            nat = np.array(next(fpos).split(), dtype=int)
        except BaseException:
            nat = np.array(next(fpos).split(), dtype=int)

        numbers = [i + 1 for i in range(len(nat)) for j in range(nat[i])]

        l = next(fpos)
        if l[0] in ['s', 'S']:
            l = next(fpos)
        if not (l[0]) in ['d', 'D']:
            raise RuntimeError(
                'only "direct" atomic coordinates are supproted')
        positions = np.zeros((np.sum(nat), 3))
        i = 0
        for l in fpos:
            if i >= sum(nat):
                break
            try:
                positions[i] = np.array(l.split()[:3])
                i += 1
            except Exception as err:
                print(err)
                pass
        if sum(nat) != i:
            raise RuntimeError(
                "not all atomic positions were read : {0} of {1}".format(
                    i, sum(nat)))
        return lattice, positions, numbers

    def _findsym(self, inPOSCAR, cell):
        if cell is None:
            cell = self.__cell_vasp(inPOSCAR=inPOSCAR)
#    cell1=tuple( [cell.lattice,cell.positions,cell.numbers] )
#    print (cell)
        print('')
        print('\n ----------INFORMATION ABOUT THE UNIT CELL----------- \n')
        print('')
        print(
            'Primitive vectors : \n',
            cell[0],
            '\n Atomic positions: \n',
            cell[1],
            '\n Atom type indices: \n',
            cell[2])
        symmetries = spglib.get_symmetry(cell)
#    print ("symmetriesreturned by spglib : ",symmetries)
        symmetries = [
            SymmetryOperation(
                rot,
                symmetries['translations'][i],
                cell[0],
                ind=i + 1,
                spinor=self.spinor) for i,
            rot in enumerate(
                symmetries['rotations'])]
        nsym = len(symmetries)
        s = spglib.get_spacegroup(cell).split(" ")

        return symmetries, s[0], int(s[1].strip("()")), cell[0]

    def __init__(self, inPOSCAR=None, cell=None, spinor=True):
        self.spinor = spinor
        self.symmetries, self.name, self.number, self.Lattice = self._findsym(
            inPOSCAR, cell)
        self.RecLattice = np.array([np.cross(self.Lattice[(i + 1) %
                                                          3], self.Lattice[(i + 2) %
                                                                           3]) for i in range(3)]) * 2 * np.pi / np.linalg.det(self.Lattice)
        print("\n Reciprocal lattice:\n", self.RecLattice)

    def show(self, refUC=None, shiftUC=np.zeros(3), symmetries=None):
        print('')
        print("\n ---------- INFORMATION ABOUT THE SPACE GROUP ---------- \n")
        print('')
        print(
            "Space group # {0} has {1} symmetry operations  ".format(
                self.number, len(
                    self.symmetries)))
        for symop in self.symmetries:
            if symmetries is None or symop.ind in symmetries:
                symop.show(refUC=refUC, shiftUC=shiftUC)


#  def show2(self,refUC=None,shiftUC=np.zeros(3)):
#    print('')
#    print("\n ---------- INFORMATION ABOUT THE SPACE GROUP ---------- \n")
#    print('')
#    print ("Space group # {0} has {1} symmetry operations  ".format(self.number,len(self.symmetries)))
#    for symop in self.symmetries:
#       symop.show2(refUC=refUC,shiftUC=shiftUC)


    def write_trace(self, refUC=None, shiftUC=np.zeros(3)):
        res = (" {0} \n"  # Number of Symmetry operations
               # In the following lines, one symmetry operation for each operation of the point group in the format: {{R|t}}-> R11,R12,...,R23,R33,t1,t2,t3
               # and, when Spin-orbit=1, the "spinor components" of the symmetry operation in\ the format Re(S11),Im(S11),Re(S12),...,Re(S22),Im(S22)\n"""
               ).format(len(self.symmetries))
        for symop in self.symmetries:
            res += symop.str2(refUC=refUC, shiftUC=shiftUC)
        return(res)

    def str(self, refUC=np.eye(3), shiftUC=np.zeros(3)):
        return (
            "SG={SG}\n name={name} \n nsym= {nsym}\n spinor={spinor}\n".format(
                SG=self.number,
                name=self.name,
                nsym=len(
                    self.symmetries),
                spinor=self.spinor) +
            "symmetries=\n" +
            "\n".join(
                s.str(
                    refUC,
                    shiftUC) for s in self.symmetries) +
            "\n\n")

    def __match_spinor_rotations(self, S1, S2):
        #        for s1,s2 in zip (S1,S2):
        #            np.savetxt(stdout,np.hstack( (s1,s2) ),fmt="%8.5f%+8.5fj "*4)
        n = 2

        def RR(x): return np.array([[x1 + 1j * x2 for x1, x2 in zip(l1, l2)]
                                    for l1, l2 in zip(x[:n * n].reshape((n, n)), x[n * n:].reshape((n, n)))])

        def residue_matrix(r): return sum([min(abs(r.dot(b).dot(
            r.T.conj()) - s * a).sum() for s in (1, -1)) for a, b in zip(S1, S2)])

        def residue(x): return residue_matrix(RR(x)) / len(S1)

        for i in range(11):
            x0 = np.random.random(2 * n * n)
            res = minimize(residue, x0)
            r = res.fun
#            print("accuracy achieved : ",r)
            if r < 1e-4:
                break
        if r > 1e-3:
            raise RuntimeError(
                "the accurcy is only {0}. Is this good?".format(r))

        R1 = RR(res.x)
#        print ("R=")
#        np.savetxt(stdout,np.hstack( (abs(R1),np.angle(R1)/np.pi) ),fmt="%8.5f")

        return np.array([R1.dot(b).dot(R1.T.conj()).dot(np.linalg.inv(
            a)).diagonal().mean().real.round() for a, b in zip(S1, S2)], dtype=int)

    def __gen_refUC():
        nmax = 3

    def get_irreps_from_table(self, refUC, shiftUC, kpname, K):
        #        self.show()
        table = IrrepTable(self.number, self.spinor)
        if self.number != table.number:
            raise RuntimeError(
                "numbers of the symmetry groups do not match : {0} and {1}".format(
                    self.number, SG.number))
#        if self.name!=table.name     : raise RuntimeError(  "names of the symmetry groups do not match : {0} and {1}".format(self.name,table.name) )
        ind = []
        dt = []
        errtxt = ""
        for j, sym in enumerate(self.symmetries):
            R, t = sym.rotation_refUC(
                refUC), sym.translation_refUC(refUC, shiftUC)
            found = False
            for i, sym2 in enumerate(table.symmetries):
                t1 = np.dot(sym2.t - t, refUC) % 1
                # t1=(sym2.t-t)%1
                t1[1 - t1 < 1e-5] = 0
                if np.allclose(R, sym2.R):
                    if np.allclose(t1, [0, 0, 0], atol=1e-6):
                        ind.append(i)
                        dt.append(sym2.t - t)
                        found = True
                        break
                    else:
                        print(
                            'table t=',
                            sym2.t,
                            '\nfound t=',
                            t,
                            "\nt(table)-t(spglib) (mod. lattice translation):",
                            t1)
                        raise RuntimeError(
                            "symmetry {0} with R={1},t={2}, t1={3} was not matched to tables".format(
                                j + 1, R, t, t1))
            if not found:
                raise RuntimeError(
                    "symmetry {0} with R={1},t={2} was not matched to tables".format(
                        j + 1, R, t))

        if (len(set(ind)) != len(self.symmetries)):
            raise RuntimeError(
                "Error in matching symmetries detected by spglib with the symmetries in the tables. Try to modify the refUC and shiftUC parameters")
        if self.spinor:
            S1 = [sym.spinor_rotation for sym in self.symmetries]
            S2 = [table.symmetries[i].S for i in ind]
            signs = self.__match_spinor_rotations(S1, S2)
#            print ("signs = ",signs)
        else:
            signs = np.ones(len(ind), dtype=int)

        tab = {}
        for irr in table.irreps:
            if irr.kpname == kpname:
                k1 = np.round(np.linalg.inv(refUC).dot(irr.k), 5) % 1
                k2 = np.round(K, 5) % 1
                if not all(np.isclose(k1, k2)):
                    raise RuntimeError(
                        "the kpoint {0} does not correspond to the point {1} ({2} in refUC / {3} in primUC) in the table".format(
                            K,
                            kpname,
                            np.round(
                                irr.k,
                                3),
                            k1))
#            print (irr.characters)
                tab[irr.name] = {}
                for j, i in enumerate(ind):
                    try:
                        #                    print (i,j)
                        tab[irr.name][j + 1] = irr.characters[i + 1] * \
                            signs[j] * np.exp(2j * np.pi * dt[j].dot(irr.k))
                    except KeyError as err:
                        pass
#        print (tab)
        if len(tab) == 0:
            raise RuntimeError(
                "the k-point with name {0} is not found in the spacegroup {1}. found only :\n{2}".format(
                    kpname, table.number, "\n ".join(
                        "{0}({1}/{2})".format(
                            irr.kpname, irr.k, np.linalg.inv(refUC).dot(
                                irr.k) %
                            1) for irr in table.irreps)))
#            raise RuntimeError("the k-point with name {0} is not found in the spacegroup {1}. found only {2}".format(kpname,table.number,set([irr.kpname for irr in table.irreps]) ) )
        return tab

#            irr.characters[i]
# return( { irr.name: np.array([irr.characters[i]*signs[j] for j,i in
# enumerate(ind)]) for irr in table.irreps if irr.kpname==kpname})
