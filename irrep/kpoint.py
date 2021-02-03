
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
import numpy.linalg as la
import copy
from .__gvectors import calc_gvectors, symm_eigenvalues, NotSymmetryError, symm_matrix
from .__readfiles import Hartree_eV
from .__readfiles import record_abinit
from .__aux import compstr, is_round
from scipy.io import FortranFile as FF
from lazy_property import LazyProperty

class Kpoint:



    @LazyProperty
    def symmetries(self):
        symmetries = {}
        #        print ("calculating symmetry eigenvalues for E={0}, WF={1} SG={2}".format(self.Energy,self.WF.shape,self.SG) )
        if not (self.SG is None):
            for symop in self.SG.symmetries:
                try:
                    symmetries[symop] = symm_eigenvalues(
                        self.K,
                        self.RecLattice,
                        self.WF,
                        self.ig,
                        spinor=self.spinor,
                        A=symop.rotation,
                        S=symop.spinor_rotation,
                        T=symop.translation,
                    )
                except NotSymmetryError as err:
                    pass  # print  ( err )
        return symmetries

    def __init__(
        self,
        ik,
        NBin,
        IBstart,
        IBend,
        Ecut,
        Ecut0,
        RecLattice,
        SG=None,
        spinor=None,
        code="vasp",
        kpt=None,
        npw_=None,
        fWFK=None,
        WCF=None,
        prefix=None,
        kptxml=None,
        flag=-1,
        usepaw=0,
        eigenval=None,
        spin_channel=None,
        IBstartE=0
    ):
        self.spinor = spinor
        self.ik0 = ik + 1  # the index in the WAVECAR (count start from 1)
        self.Nband = IBend - IBstart
        #        self.n=np.arange(IBstart,IBend)+1
        self.RecLattice = RecLattice

        if code.lower() == "vasp":
            self.WF, self.ig = self.__init_vasp(
                WCF, ik, NBin, IBstart, IBend, Ecut, Ecut0
            )
        elif code.lower() == "abinit":
            self.WF, self.ig = self.__init_abinit(
                fWFK,
                ik,
                NBin,
                IBstart,
                IBend,
                Ecut,
                Ecut0,
                kpt=kpt,
                npw_=npw_,
                flag=flag,
                usepaw=usepaw,
            )
        elif code.lower() == "espresso":
            self.WF, self.ig = self.__init_espresso(
                prefix, ik, IBstart, IBend, Ecut, Ecut0, kptxml=kptxml,
                spin_channel=spin_channel,IBstartE=IBstartE
            )
        elif code.lower() == "wannier":
            self.WF, self.ig = self.__init_wannier(
                NBin, IBstart, IBend, Ecut, kpt=kpt, eigenval=eigenval
            )
        else:
            raise RuntimeError("unknown code : {}".format(code))

        self.WF /= (
            np.sqrt(np.abs(np.einsum("ij,ij->i", self.WF.conj(), self.WF)))
        ).reshape(self.Nband, 1)
        self.SG = SG

    #        self.__calc_sym_eigenvalues()
    #        print("WF=\n",WF)

    def copy_sub(self, E, WF):
        #        print ("making a subspace with E={0}\n WF = {1}".format(E,WF.shape))
        other = copy.copy(self)
        sortE = np.argsort(E)
        other.Energy = E[sortE]
        other.WF = WF[sortE]
        other.Nband = len(E)
        # other.__calc_sym_eigenvalues()
        #        print ( self.Energy,other.Energy)
        #        print ( self.WF.shape, other.WF.shape)
        #        other.write_characters()
        #        print ("self overlap:\n",self.overlap(self))
        return other

    def unfold(self, supercell, kptPBZ, degen_thresh=1e-4):
        """unfolds a kpoint of a supercell onto the point of the primitivecell kptPBZ
           returns an array of 2 (5 in case of spinors) columns:
           E , weight , Sx , Sy , Sz
           """
        if not is_round(kptPBZ.dot(supercell.T) - self.K, prec=1e-5):
            raise RuntimeError(
                "unable to unfold {} to {}, withsupercell={}".format(
                    self.K, kptPBZ, supercell
                )
            )
        g_shift = kptPBZ - self.K.dot(np.linalg.inv(supercell.T))
        #        print ("g_shift={}".format(g_shift))
        selectG = np.array(
            np.where(
                [
                    is_round(dg, prec=1e-4)
                    for dg in (self.ig[:3].T.dot(np.linalg.inv(supercell.T)) - g_shift)
                ]
            )[0]
        )
        #        print ("unfolding {} to {}, selecting {} of {} g-vectors \n".format(self.K,kptPBZ,len(selectG),self.ig.shape[1],selectG,self.ig.T))
        if self.spinor:
            selectG = np.hstack((selectG, selectG + self.NG))
        WF = self.WF[:, selectG]
        result = []
        for b1, b2, E, matrices in self.get_rho_spin(degen_thresh):
            proj = np.array(
                [
                    [WF[i].conj().dot(WF[j]) for j in range(b1, b2)]
                    for i in range(b1, b2)
                ]
            )
            result.append([E,] + [np.trace(proj.dot(M)).real for M in matrices])
        return np.array(result)

    def get_rho_spin(self, degen_thresh=1e-4):
        if not hasattr(self, "rho_spin"):
            self.rho_spin = {}
        if degen_thresh not in self.rho_spin:
            self.rho_spin[degen_thresh] = self.__eval_rho_spin(degen_thresh)
        return self.rho_spin[degen_thresh]

    @property
    def NG(self):
        return self.ig.shape[0]

    def __eval_rho_spin(self, degen_thresh):
        borders = np.hstack(
            [
                [0],
                np.where(self.Energy[1:] - self.Energy[:-1] > degen_thresh)[0] + 1,
                [self.Nband],
            ]
        )
        result = []
        for b1, b2 in zip(borders, borders[1:]):
            E = self.Energy[b1:b2].mean()
            W = np.array(
                [
                    [self.WF[i].conj().dot(self.WF[j]) for j in range(b1, b2)]
                    for i in range(b1, b2)
                ]
            )
            if self.spinor:
                ng = self.NG
                Smatrix = [
                    [
                        np.array(
                            [
                                [
                                    self.WF[i, ng * s : ng * (s + 1)]
                                    .conj()
                                    .dot(self.WF[j, ng * t : ng * (t + 1)])
                                    for j in range(b1, b2)
                                ]
                                for i in range(b1, b2)
                            ]
                        )  # band indices
                        for t in (0, 1)
                    ]
                    for s in (0, 1)
                ]  # spin indices
                Sx = Smatrix[0][1] + Smatrix[1][0]
                Sy = 1j * (-Smatrix[0][1] + Smatrix[1][0])
                Sz = Smatrix[0][0] - Smatrix[1][1]
                result.append((b1, b2, E, (W, Sx, Sy, Sz)))
            else:
                result.append((b1, b2, E, (W,)))
        return result

    def Separate(self, symop, degen_thresh=1e-5, groupKramers=True):
        # separates the bandstructure according to symmetry eigenvalues returns
        # a dictionar of Kpoint objects eigval:Kpoint
        borders = np.hstack(
            [
                [0],
                np.where(self.Energy[1:] - self.Energy[:-1] > degen_thresh)[0] + 1,
                [self.Nband],
            ]
        )
        S = symm_matrix(
            self.K,
            self.RecLattice,
            self.WF,
            self.ig,
            spinor=self.spinor,
            A=symop.rotation,
            S=symop.spinor_rotation,
            T=symop.translation,
        )
        S1 = self.WF.conj().dot(self.WF.T)
        check = np.max(abs(S1 - np.eye(S1.shape[0])))
        if check > 1e-5:
            print(
                "orthogonality (largest of diag. <psi_nk|psi_mk>): {0:7.5} > 1e-5   \n".format(
                    check
                )
            )
        #        print ("symmetry matrix \n",shortS)
        eigenvalues = []
        eigenvectors = []
        Eloc = []

        def short(A):
            return "".join(
                "   ".join("{0:+5.2f} {1:+5.2f}".format(x.real, x.imag) for x in a)
                + "\n"
                for a in A
            )

        Sblock = np.copy(S)
        for b1, b2 in zip(borders, borders[1:]):
            Sblock[b1:b2, b1:b2] = 0
        check = np.max(abs(Sblock))
        if check > 0.1:
            print("WARNING: off-block:  \n", check)
            print(short(Sblock))

        for b1, b2 in zip(borders, borders[1:]):
            W, V = la.eig(S[b1:b2, b1:b2])
            #            print (b1,b2,"symmetry submatrix \n",short(S[b1:b2,b1:b2]))
            for w, v in zip(W, V.T):
                eigenvalues.append(w)
                Eloc.append(self.Energy[b1:b2].mean())
                eigenvectors.append(
                    np.hstack((np.zeros(b1), v, np.zeros(self.Nband - b2)))
                )
        w = np.array(eigenvalues)
        v = np.array(eigenvectors).T
        Eloc = np.array(Eloc)

        #        print ("eigenvalues:",w)
        #        print ("eigenvectors:\n",v)
        #        print ("Eloc:\n",Eloc)
        if np.abs((np.abs(w) - 1.0)).max() > 1e-4:
            print("WARNING : some eigenvalues are not unitary :{0} ".format(w))
        if np.abs((np.abs(w) - 1.0)).max() > 3e-1:
            raise RuntimeError(" some eigenvalues are not unitary :{0} ".format(w))
        w /= np.abs(w)
        nb = len(w)

        subspaces = {}

        if groupKramers:
            w1 = np.argsort(np.real(w))
            w = w[w1]
            v = v[:, w1]
            Eloc = Eloc[w1]
            borders = np.hstack(
                ([0], np.where((w[1:] - w[:-1]) > 0.05)[0] + 1, [len(w)])
            )
            if len(borders) > 0:
                for b1, b2 in zip(borders, borders[1:]):
                    v1 = v[:, b1:b2]
                    subspaces[w[b1:b2].mean()] = self.copy_sub(
                        E=Eloc[b1:b2], WF=v1.T.dot(self.WF)
                    )
            else:
                v1 = v
                subspaces[w.mean()] = self.copy_sub(E=Eloc, WF=v1.T.dot(self.WF))
        else:
            w1 = np.argsort(np.angle(w))
            w = w[w1]
            v = v[:, w1]
            Eloc = Eloc[w1]
            borders = np.where(abs(w - np.roll(w, 1)) > 0.1)[0]
            if len(borders) > 0:
                for b1, b2 in zip(borders, np.roll(borders, -1)):
                    v1 = np.roll(v, -b1, axis=1)[:, : (b2 - b1) % nb]
                    subspaces[np.roll(w, -b1)[: (b2 - b1) % nb].mean()] = self.copy_sub(
                        E=np.roll(Eloc, -b1)[: (b2 - b1) % nb], WF=v1.T.dot(self.WF)
                    )
            else:
                v1 = v
                subspaces[w.mean()] = self.copy_sub(E=Eloc, WF=v1.T.dot(self.WF))

        return subspaces

    def __init_vasp(self, WCF, ik, NBin, IBstart, IBend, Ecut, Ecut0):
        r = WCF.record(2 + ik * (NBin + 1))
        # get the number of planewave coefficients. It should be even for spinor wavefunctions
        #    print (r)
        npw = int(r[0])
        if self.spinor:
            if npw != int(npw / 2) * 2:
                raise RuntimeError(
                    "odd number of coefs {0} for spinor wavefunctions".format(npw)
                )
        self.K = r[1:4]
        eigen = np.array(r[4 : 4 + NBin * 3]).reshape(NBin, 3)[:, 0]
        self.Energy = eigen[IBstart:IBend]
        try:
            self.upper = eigen[IBend]
        except BaseException:
            self.upper = np.NaN

        ig = calc_gvectors(
            self.K, self.RecLattice, Ecut0, npw, Ecut, spinor=self.spinor
        )
        selectG = np.hstack((ig[3], ig[3] + int(npw / 2))) if self.spinor else ig[3]
        WF = np.array(
            [
                WCF.record(3 + ik * (NBin + 1) + ib, npw, np.complex64)[selectG]
                for ib in range(IBstart, IBend)
            ]
        )
        return WF, ig

    def __init_abinit(
        self,
        fWFK,
        ik,
        NBin,
        IBstart,
        IBend,
        Ecut,
        Ecut0,
        kpt,
        npw_,
        thresh=1e-4,
        flag=-1,
        usepaw=0,
    ):

        assert not (kpt is None)
        self.K = kpt
        print("reading k-point", ik)
        # we need to skip lines in fWFK until we reach the lines of ik
        while flag < ik:
            record = record_abinit(fWFK, "3i4")  # [0]
            npw, nspinor_loc, nband_loc = record
            kg = record_abinit(fWFK, "({npw},3)i4".format(npw=npw))  # [0]
            eigen, occ = fWFK.read_record(
                "{nband}f8,{nband}f8".format(nband=nband_loc)
            )[0]
            nspinor = 2 if self.spinor else 1
            CG = np.zeros((IBend - IBstart, npw * nspinor), dtype=complex)
            for iband in range(nband_loc):
                cg_tmp = record_abinit(fWFK, "{0}f8".format(2 * npw * nspinor))  # [0]
                if iband >= IBstart and iband < IBend:
                    CG[iband - IBstart] = cg_tmp[0::2] + 1.0j * cg_tmp[1::2]
            flag += 1

        # now, we have kept in npw,nspinor_loc,naband_loc,eigen,occ,cg_tmp the
        # info of the k-point labeled by ik
        assert npw == npw_
        assert nband_loc == NBin
        assert (nspinor_loc == 2 and self.spinor) or (
            nspinor_loc == 1 and not self.spinor
        )

        if usepaw == 0:
            assert (
                np.max(np.abs(CG.conj().dot(CG.T) - np.eye(IBend - IBstart))) < 1e-10
            )  # check orthonormality

        self.Energy = eigen[IBstart:IBend] * Hartree_eV
        try:
            self.upper = eigen[IBend] * Hartree_eV
        except BaseException:
            self.upper = np.NaN

        return self.__sortIG(kg, kpt, CG, self.RecLattice, Ecut0, Ecut, thresh=thresh)

    def __init_wannier(self, NBin, IBstart, IBend, Ecut, kpt, eigenval, thresh=1e-4):
        self.K = np.array(kpt, dtype=float)
        self.Energy = eigenval[IBstart:IBend]
        fname = "UNK{:05d}.{}".format(self.ik0, "NC" if self.spinor else "1")
        fUNK = FF(fname, "r")
        ngx, ngy, ngz, ik, nbnd = record_abinit(fUNK, "i4,i4,i4,i4,i4")[0]
        ngtot = ngx * ngy * ngz
        if ik != self.ik0:
            raise RuntimeError(
                "file {} contains point number {}, expected {}".format(
                    fname, ik, self.ik0
                )
            )
        if nbnd != NBin:
            raise RuntimeError(
                "file {} contains {} bands , expected {}".format(fname, nbnd, NBin)
            )
        nspinor = 2 if self.spinor else 1

        try:
            self.upper = eigenval[IBend]
        except BaseException:
            self.upper = np.NaN

        ig = calc_gvectors(
            self.K,
            self.RecLattice,
            Ecut,
            spinor=self.spinor,
            nplanemax=np.max([ngx, ngy, ngz]) // 2,
        )

        selectG = tuple(ig[0:3])

        def _readWF_1(skip=False):
            cg_tmp = record_abinit(fUNK, "{}f8".format(ngtot * 2))
            if skip:
                return np.array([0], dtype=complex)
            cg_tmp = (cg_tmp[0::2] + 1.0j * cg_tmp[1::2]).reshape(
                (ngx, ngy, ngz), order="F"
            )
            cg_tmp = np.fft.fftn(cg_tmp)
            return cg_tmp[selectG]

        def _readWF(skip=False):
            return np.hstack([_readWF_1(skip) for i in range(nspinor)])

        for ib in range(IBstart):
            _readWF(skip=True)
        WF = np.array([_readWF(skip=False) for ib in range(IBend - IBstart)])
        return WF, ig

    def __init_espresso(
        self, prefix, ik, IBstart, IBend, Ecut, Ecut0, kptxml, thresh=1e-4,
           spin_channel=None,IBstartE=0
    ):
        self.K = np.array(kptxml.find("k_point").text.split(), dtype=float)

        eigen = np.array(kptxml.find("eigenvalues").text.split(), dtype=float)

        self.Energy=eigen[IBstartE+IBstart:IBstartE+IBend]*Hartree_eV
        try:
            self.upper=eigen[IBstartE+IBend]*Hartree_eV
        except:
            self.upper = np.NaN


        npw = int(kptxml.find("npw").text)
        #        kg= np.random.randint(100,size=(npw,3))-50
        npwtot = npw * (2 if self.spinor else 1)
        CG = np.zeros((IBend - IBstart, npwtot), dtype=complex)
        wfcname="wfc{}{}".format({None:"","dw":"dw","up":"up"}[spin_channel],ik+1)
        try:
            fWFC=FF("{}.save/{}.dat".format(prefix,wfcname.lower()),"r")
        except FileNotFoundError:
            fWFC=FF("{}.save/{}.dat".format(prefix,wfcname.upper()),"r")

        rec = record_abinit(fWFC, "i4,3f8,i4,i4,f8")[0]
        ik, xk, ispin, gamma_only, scalef = rec
        #        xk/=bohr
        #        xk=xk.dot(np.linalg.inv(RecLattice))

        rec = record_abinit(fWFC, "4i4")
        #        print ('rec=',rec)
        ngw, igwx, npol, nbnd = rec

        rec = record_abinit(fWFC, "(3,3)f8")
        #        print ('rec=',rec)
        B = np.array(rec)
        #        print (np.mean(B/RecLattice))
        self.K = xk.dot(np.linalg.inv(B))

        rec = record_abinit(fWFC, "({},3)i4".format(igwx))
        #        print ('rec=',rec)
        kg = np.array(rec)
        #        print (np.mean(B/RecLattice))
        #        print ("k-point {0}: {1}/{2}={3}".format(ik, self.K,xk,self.K/xk))
        #        print ("k-point {0}: {1}".format(ik,self.K ))

        for ib in range(IBend):
            cg_tmp = record_abinit(fWFC, "{}f8".format(npwtot * 2))
            if ib >= IBstart:
                CG[ib - IBstart] = cg_tmp[0::2] + 1.0j * cg_tmp[1::2]

        return self.__sortIG(kg, self.K, CG, B, Ecut0, Ecut, thresh=thresh)

    def __sortIG(self, kg, kpt, CG, RecLattice, Ecut0, Ecut, thresh=1e-4):
        KG = (kg + kpt).dot(RecLattice)
        npw = kg.shape[0]
        eKG = Hartree_eV * (la.norm(KG, axis=1) ** 2) / 2
        print(
            "Found cutoff: {0:12.6f} eV   Largest plane wave energy in K-point {1:4d}: {2:12.6f} eV".format(
                Ecut0, self.ik0, np.max(eKG)
            )
        )
        assert Ecut0 * 1.000000001 > np.max(eKG)
        sel = np.where(eKG < Ecut)[0]
        npw1 = sel.shape[0]

        KG = KG[sel]
        kg = kg[sel]
        eKG = eKG[sel]
        srt = np.argsort(eKG)
        eKG = eKG[srt]
        igall = np.zeros((6, len(sel)), dtype=int)
        igall[:3, :] = kg[srt].T
        igall[3, :] = srt
        wall = (
            [0] + list(np.where(eKG[1:] - eKG[:-1] > thresh)[0] + 1) + [igall.shape[1]]
        )
        for i in range(len(wall) - 1):
            igall[4, wall[i] : wall[i + 1]] = wall[i]
            igall[5, wall[i] : wall[i + 1]] = wall[i + 1]

        if self.spinor:
            CG = CG[:, np.hstack((sel[srt], sel[srt] + npw))]
        else:
            CG = CG[:, sel[srt]]

        return CG, igall

    def write_characters(
        self,
        degen_thresh=1e-8,
        irreptable=None,
        symmetries=None,
        preline="",
        efermi=0.0,
        plotFile=None,
        kpl="",
    ):
        if symmetries is None:
            sym = {s.ind: s for s in self.symmetries}
        else:
            sym = {s.ind: s for s in self.symmetries if s.ind in symmetries}

        char = np.vstack([self.symmetries[sym[i]] for i in sorted(sym)])
        borders = np.hstack(
            [
                [0],
                np.where(self.Energy[1:] - self.Energy[:-1] > degen_thresh)[0] + 1,
                [self.Nband],
            ]
        )
        Nirrep = np.linalg.norm(char.sum(axis=1)) ** 2 / char.shape[0]
        if abs(Nirrep - round(Nirrep)) > 1e-2:
            print("WARNING - non-integer number of irreps : {0}".format(Nirrep))
        Nirrep = int(round(Nirrep))
        char = np.array(
            [char[:, start:end].sum(axis=1) for start, end in zip(borders, borders[1:])]
        )

        #        print(" char ",char.shape,"\n",char)
        writeimaginary = np.abs(char.imag).max() > 1e-4

        s1 = " " * 4 if writeimaginary else ""

        E = np.array(
            [self.Energy[start:end].mean() for start, end in zip(borders, borders[1:])]
        )
        dim = np.array([end - start for start, end in zip(borders, borders[1:])])
        if irreptable is None:
            irreps = ["None"] * (len(borders) - 1)
        else:
            try:
                irreps = [
                    {
                        ir: np.array(
                            [irreptable[ir][sym.ind] for sym in self.symmetries]
                        ).dot(ch.conj())
                        / len(ch)
                        for ir in irreptable
                    }
                    for ch in char
                ]
            except KeyError as ke:
                print(ke)
                print("irreptable:", irreptable)
                print([sym.ind for sym in self.symmetries])
                raise ke
            irreps = [
                ", ".join(
                    ir
                    + "({0:.5}".format(irr[ir].real)
                    + (
                        "{0:+.5f}i".format(irr[ir].imag)
                        if abs(irr[ir].imag) > 1e-4
                        else ""
                    )
                    + ")"
                    for ir in irr
                    if abs(irr[ir]) > 1e-3
                )
                for irr in irreps
            ]
        #            irreps=[ "None" ]*(len(borders)-1)
        irreplen = max(len(irr) for irr in irreps)
        if irreplen % 2 == 1:
            irreplen += 1
        s2 = " " * int(irreplen / 2 - 3)
        print(
            "\n\nk-point {0:3d} :{1} \n number of irreps = {2}".format(
                self.ik0, self.K, Nirrep
            )
        )
        print("   Energy  | multiplicity |{0} irreps {0}| sym. operations  ".format(s2))
        print(
            "           |              |{0}        {0}| ".format(s2),
            " ".join(s1 + "{0:4d}    ".format(i) + s1 for i in sorted(sym)),
        )
        print(
            "\n".join(
                (" {0:8.4f}  |    {1:5d}     | {2:" + str(irreplen) + "s} |").format(
                    e - efermi, d, ir
                )
                + " ".join(
                    "{0:8.4f}".format(c.real)
                    + ("{0:+7.4f}j".format(c.imag) if writeimaginary else "")
                    for c in ch
                )
                for e, d, ir, ch in zip(E, dim, irreps, char)
            )
        )

        if plotFile is not None:
            plotFile.write(
                (
                    "\n".join(
                        ("{2:8.4f}   {0:8.4f}      {1:5d}   ").format(
                            e - efermi, d, kpl
                        )
                        + " ".join(
                            "{0:8.4f}".format(c.real)
                            + ("{0:+7.4f}j".format(c.imag) if writeimaginary else "")
                            for c in ch
                        )
                        for e, d, ch in zip(E, dim, char)
                    )
                )
                + "\n\n"
            )

        isyminv = None
        for s in sym:
            if (
                sum(abs(sym[s].translation)) < 1e-6
                and abs(sym[s].rotation + np.eye(3)).sum() < 1e-6
            ):
                isyminv = s
        if isyminv is None:
            print("no inversion")
            NBANDINV = 0
        else:
            print("inversion is #", isyminv)
            NBANDINV = int(round(sum(1 - self.symmetries[sym[isyminv]].real) / 2))
            if self.spinor:
                print("number of inversions-odd Kramers pairs : ", int(NBANDINV / 2))
            else:
                print("number of inversions-odd states : ", NBANDINV)
            print("Gap with upper bands : ", self.upper - self.Energy[-1])

        firrep = open("irreps.dat", "a")
        for e, ir in zip(E, irreps):
            for irrep in ir.split(","):
                try:
                    weight = abs(compstr(irrep.split("(")[1].strip(")")))
                    if weight > 0.3:
                        firrep.write(
                            preline
                            + " {0:10s} ".format(irrep.split("(")[0])
                            + "  {0:10.5f}\n".format(e - efermi)
                        )
                except IndexError:
                    pass

        return NBANDINV, self.Energy[-1], self.upper

    def write_trace(self, degen_thresh=1e-8, symmetries=None, efermi=0.0):
        if symmetries is None:
            sym = {s.ind: s for s in self.symmetries}
        else:
            sym = {s.ind: s for s in self.symmetries if s.ind in symmetries}

        res = (
            "{0} \n"
            + " {1} \n"  # Number of symmetry operations of the little co-group of the 1st maximal k-vec. In the next line the position of each element of the point group in the list above.
            # For each band introduce a row with the followind data: (1) 1+number of bands below, (2) dimension (degeneracy) of the band,
            # (3) energy and eigenvalues (real part, imaginary part) for each symmetry operation of the little group (listed above).
        ).format(len(sym.keys()), "  ".join(str(x) for x in sym))

        char = np.vstack([self.symmetries[sym[i]] for i in sorted(sym)])
        borders = np.hstack(
            [
                [0],
                np.where(self.Energy[1:] - self.Energy[:-1] > degen_thresh)[0] + 1,
                [self.Nband],
            ]
        )
        char = np.array(
            [char[:, start:end].sum(axis=1) for start, end in zip(borders, borders[1:])]
        )

        E = np.array(
            [self.Energy[start:end].mean() for start, end in zip(borders, borders[1:])]
        )
        dim = np.array([end - start for start, end in zip(borders, borders[1:])])
        IB = np.cumsum(np.hstack(([0], dim[:-1]))) + 1
        res += (
            "\n".join(
                (" {ib:8d}  {d:8d}   {E:8.4f} ").format(E=e - efermi, d=d, ib=ib)
                + "  ".join("{0:10.6f}   {1:10.6f} ".format(c.real, c.imag) for c in ch)
                for e, d, ib, ch in zip(E, dim, IB, char)
            )
            + "\n"
        )

        return res

    def write_trace_all(self, degen_thresh=1e-8, symmetries=None, efermi=0.0, kpline=0):
        preline = "{0:10.6f}     {1:10.6f}  {2:10.6f}  {3:10.6f}  ".format(
            kpline, *tuple(self.K)
        )
        if symmetries is None:
            sym = {s.ind: s for s in self.symmetries}
        else:
            sym = {s.ind: s for s in self.symmetries if s.ind in symmetries}

        char0 = {i: self.symmetries[sym[i]] for i in sym}
        borders = np.hstack(
            [
                [0],
                np.where(self.Energy[1:] - self.Energy[:-1] > degen_thresh)[0] + 1,
                [self.Nband],
            ]
        )
        char = {
            i: np.array(
                [char0[i][start:end].sum() for start, end in zip(borders, borders[1:])]
            )
            for i in char0
        }
        E = np.array(
            [self.Energy[start:end].mean() for start, end in zip(borders, borders[1:])]
        )
        dim = np.array([end - start for start, end in zip(borders, borders[1:])])
        IB = np.cumsum(np.hstack(([0], dim[:-1]))) + 1
        res = (
            "\n".join(
                preline
                + (" {ib:8d}  {d:8d}   {E:8.4f} ").format(E=e - efermi, d=d, ib=ib)
                + "     ".join(
                    (
                        "{0:10.6f} {1:10.6f}".format(char[i][j].real, char[i][j].imag)
                        if i in char
                        else (" " * 7 + "X" * 3 + " " * 8 + "X" * 3)
                    )
                    for i in range(1, len(self.SG.symmetries) + 1)
                )
                for e, d, ib, j in zip(E, dim, IB, np.arange(len(dim)))
            )
            + "\n"
        )
        return res

    def overlap(self, other):
        """ Calculates the overlap matrix <u_m(k) | u_n(k'+g) > """
        g = np.array((self.K - other.K).round(), dtype=int)
        igall = np.hstack((self.ig[:3], other.ig[:3] - g[:, None]))
        igmax = igall.max(axis=1)
        igmin = igall.min(axis=1)
        igsize = igmax - igmin + 1
        #        print (self.ig.T)
        #        print (igsize)
        res = np.zeros((self.Nband, other.Nband), dtype=complex)
        for s in [0, 1] if self.spinor else [0]:
            WF1 = np.zeros((self.Nband, igsize[0], igsize[1], igsize[2]), dtype=complex)
            WF2 = np.zeros(
                (other.Nband, igsize[0], igsize[1], igsize[2]), dtype=complex
            )
            for i, ig in enumerate(self.ig.T):
                WF1[:, ig[0] - igmin[0], ig[1] - igmin[1], ig[2] - igmin[2]] = self.WF[
                    :, i + s * self.ig.shape[1]
                ]
            for i, ig in enumerate(other.ig[:3].T - g[None, :]):
                WF2[:, ig[0] - igmin[0], ig[1] - igmin[1], ig[2] - igmin[2]] = other.WF[
                    :, i + s * other.ig.shape[1]
                ]
            res += np.einsum("mabc,nabc->mn", WF1.conj(), WF2)
        #        return np.einsum("mabc,nabc->mn",WF1.conj(),WF2)
        #        return np.einsum("ma,na->mn",self.WF.conj(),other.WF)
        return res

    def getloc1(self, loc):
        gmax = abs(self.ig[:3]).max(axis=1)
        grid = [np.linspace(0.0, 1.0, 2 * gm + 1, False) for gm in gmax]
        print("grid:", grid)
        loc_grid = loc(
            grid[0][:, None, None], grid[1][None, :, None], grid[2][None, None, :]
        )
        print("loc=", loc, "loc_grid=\n", loc_grid)
        #        FFTgrid=np.zeros( (self.Nband,*(2*gmax+1)),dtype=complex )
        res = np.zeros(self.Nband)
        for s in [0, 1] if self.spinor else [0]:
            WF1 = np.zeros((self.Nband, *(2 * gmax + 1)), dtype=complex)
            for i, ig in enumerate(self.ig.T):
                WF1[:, ig[0], ig[1], ig[2]] = self.WF[:, i + s * self.ig.shape[1]]
            #            print ("wfsum",WF1.sum()," shape ",WF1.shape,loc_grid.shape)
            res += np.array(
                [
                    np.sum(np.abs(np.fft.ifftn(WF1[ib])) ** 2 * loc_grid).real
                    for ib in range(self.Nband)
                ]
            )
        print("    ", loc_grid.shape)
        return res * (np.prod(loc_grid.shape))

    def getloc(self, locs):
        return np.array([self.getloc1(loc) for loc in locs])
