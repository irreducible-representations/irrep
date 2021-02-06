
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
##################################################################{0:8.4f}   {1:8.4f}  ".format(


import copy
import functools

import numpy as np
import numpy.linalg as la

from .__aux import str2bool, BOHR
from .__readfiles import AbinitHeader, Hartree_eV
from .__readfiles import WAVECARFILE
from .kpoint import Kpoint
from .spacegroup import SpaceGroup


class BandStructure:
    def __init__(
        self,
        fWAV=None,
        fWFK=None,
        prefix=None,
        fPOS=None,
        Ecut=None,
        IBstart=None,
        IBend=None,
        kplist=None,
        spinor=None,
        code="vasp",
        EF=None,
        onlysym=False,
        spin_channel=None
    ):
        code = code.lower()

        if spin_channel is not None:
            spin_channel=spin_channel.lower()
        if spin_channel=='down' : spin_channel='dw'

        if code == "vasp":
            self.__init_vasp(
                fWAV, fPOS, Ecut, IBstart, IBend, kplist, spinor, EF=EF, onlysym=onlysym
            )
        elif code == "abinit":
            self.__init_abinit(
                fWFK, Ecut, IBstart, IBend, kplist, EF=EF, onlysym=onlysym
            )
        elif code == "espresso":
            self.__init_espresso(
                prefix, Ecut, IBstart, IBend, kplist, EF=EF, onlysym=onlysym, spin_channel=spin_channel
            )
        elif code == "wannier90":
            self.__init_wannier(
                prefix, Ecut, IBstart, IBend, kplist, EF=EF, onlysym=onlysym
            )
        else:
            raise RuntimeError("Unknown/unsupported code :{}".format(code))

    def __init_vasp(
        self,
        fWAV,
        fPOS,
        Ecut=None,
        IBstart=None,
        IBend=None,
        kplist=None,
        spinor=None,
        EF=None,
        onlysym=False,
    ):
        if spinor is None:
            raise RuntimeError(
                "spinor should be specified in the command line for VASP bandstructure"
            )
        self.spacegroup = SpaceGroup(inPOSCAR=fPOS, spinor=spinor)
        self.spinor = spinor
        if onlysym:
            return
        self.efermi = 0.0 if EF is None else EF
        print("Efermi = ", self.efermi, EF)
        WCF = WAVECARFILE(fWAV)
        # RECLENGTH=3 # the length of a record in WAVECAR. It is defined in the
        # first record, so let it be 3 fo far"
        WCF.rl, ispin, iprec = [int(x) for x in WCF.record(0)]
        if iprec != 45200:
            raise RuntimeError("double precision WAVECAR is not supported")
        if ispin != 1:
            raise RuntimeError(
                "WAVECAR contains spin-polarized non-spinor wavefunctions. "
                + "ISPIN={0}  this is not supported yet".format(ispin)
            )

        tmp = WCF.record(1)
        NK = int(tmp[0])
        NBin = int(tmp[1])
        Ecut0 = tmp[2]

        IBstart = 0 if (IBstart is None or IBstart <= 0) else IBstart - 1
        if IBend is None or IBend <= 0 or IBend > NBin:
            IBend = NBin
        NBout = IBend - IBstart
        if NBout <= 0:
            raise RuntimeError("No bands to calculate")
        if Ecut is None or Ecut > Ecut0 or Ecut <= 0:
            Ecut = Ecut0
        self.Ecut = Ecut
        self.Ecut0 = Ecut0

        self.Lattice = np.array(tmp[3:12]).reshape(3, 3)
        self.RecLattice = (
            np.array(
                [
                    np.cross(self.Lattice[(i + 1) % 3], self.Lattice[(i + 2) % 3])
                    for i in range(3)
                ]
            )
            * 2
            * np.pi
            / np.linalg.det(self.Lattice)
        )

        print(
            "WAVECAR contains {0} k-points and {1} bands.\n Saving {2} bands starting from {3} in the output".format(
                NK, NBin, NBout, IBstart + 1
            )
        )
        print("Energy cutoff in WAVECAR : ", Ecut0)
        print("Energy cutoff reduced to : ", Ecut)
        #        print (kplist,NK)

        if kplist is None:
            kplist = range(NK)
        else:
            kplist -= 1
            kplist = np.array([k for k in kplist if k >= 0 and k < NK])
        #        print (kplist)
        self.kpoints = [
            Kpoint(
                ik,
                NBin,
                IBstart,
                IBend,
                Ecut,
                Ecut0,
                self.RecLattice,
                SG=self.spacegroup,
                spinor=self.spinor,
                WCF=WCF,
            )
            for ik in kplist
        ]

    def __init_abinit(
        self,
        WFKname,
        Ecut=None,
        IBstart=None,
        IBend=None,
        kplist=None,
        EF=None,
        onlysym=False,
    ):

        header = AbinitHeader(WFKname)
        usepaw = header.usepaw
        self.spinor = header.spinor
        self.spacegroup = SpaceGroup(
            cell=(header.rprimd, header.xred, header.typat), spinor=self.spinor
        )
        if onlysym:
            return
        self.efermi = header.efermi * Hartree_eV if EF is None else EF
        #        self.spacegroup.show()

        #        global fWFK
        fWFK = header.fWFK
        Ecut0 = header.ecut
        NBin = header.nband.min()
        NK = header.nkpt
        IBstart = 0 if (IBstart is None or IBstart <= 0) else IBstart - 1
        if IBend is None or IBend <= 0 or IBend > NBin:
            IBend = NBin
        NBout = IBend - IBstart
        if NBout <= 0:
            raise RuntimeError("No bands to calculate")
        if Ecut is None or Ecut > Ecut0 or Ecut <= 0:
            Ecut = Ecut0
        self.Ecut = Ecut
        self.Ecut0 = Ecut0

        self.Lattice = header.rprimd
        print("lattice vectors:\n", self.Lattice)
        self.RecLattice = (
            np.array(
                [
                    np.cross(self.Lattice[(i + 1) % 3], self.Lattice[(i + 2) % 3])
                    for i in range(3)
                ]
            )
            * 2
            * np.pi
            / np.linalg.det(self.Lattice)
        )

        print(
            "WFK contains {0} k-points and {1} bands.\n Saving {2} bands starting from {3} in the output".format(
                NK, NBin, NBout, IBstart + 1
            )
        )
        print("Energy cutoff in WFK file : ", Ecut0)
        print("Energy cutoff reduced to : ", Ecut)
        if kplist is None:
            kplist = range(NK)
        else:
            kplist -= 1
            kplist = np.array([k for k in kplist if k >= 0 and k < NK])
        #        print ("kplist",kplist)
        self.kpoints = []
        flag = -1
        for ik in kplist:
            kp = Kpoint(
                ik,
                header.nband[ik],
                IBstart,
                IBend,
                Ecut,
                Ecut0,
                self.RecLattice,
                SG=self.spacegroup,
                spinor=self.spinor,
                code="abinit",
                kpt=header.kpt[ik],
                npw_=header.npwarr[ik],
                fWFK=fWFK,
                flag=flag,
                usepaw=usepaw,
            )
            self.kpoints.append(kp)
            flag = ik

    def __init_wannier(
        self,
        prefix,
        Ecut=None,
        IBstart=None,
        IBend=None,
        kplist=None,
        EF=None,
        onlysym=False,
    ):

        if Ecut is None:
            raise RuntimeError("Ecut mandatory for Wannier90")

        fwin = [l.strip().lower() for l in open(prefix + ".win").readlines()]

        def split(l):
            if "=" in l:
                return l.split("=")
            elif ":" in l:
                return l.split(":")
            else:
                return l.split()

        fwin = [
            [s.strip() for s in split(l)]
            for l in fwin
            if len(l) > 0 and l[0] not in ("!", "#")
        ]
        ind = np.array([l[0] for l in fwin])
        # print(fwin) #com

        def get_param(key, tp, default=None, join=False):
            i = np.where(ind == key)[0]
            if len(i) == 0:
                if default is None:
                    raise RuntimeError(
                        "parameter {} was not found in {}.win".format(key, prefix)
                    )
                else:
                    return default
            if len(i) > 1:
                raise RuntimeError(
                    "parameter {} was found {} times in {}.win".format(
                        key, len(i), prefix
                    )
                )
            x = fwin[i[0]][1:]  # mp_grid should work
            if len(x) > 1:
                if join:
                    x = " ".join(x)
                else:
                    raise RuntimeError(
                        "length {} found for parameter {}, rather than lenght 1 in {}.win".format(
                            len(x), key, prefix
                        )
                    )
            else:
                x = fwin[i[0]][1]
            return tp(x)

        NBin = get_param("num_bands", int)
        #        print ("nbands=",NBin)
        self.spinor = str2bool(get_param("spinors", str))
        if EF is None:
            try:
                self.efermi = (
                    get_param("fermi_energy", float, 0.0) if EF is None else EF
                )
            except Exception as err:
                print(
                    "WARNING : fermi_energy not found.Setting as zero : `{}`".format(
                        err
                    )
                )
                self.efermi = 0
        else:
            self.efermi = EF

        NK = np.prod(np.array(get_param("mp_grid", str).split(), dtype=int))

        self.Lattice = None
        kpred = None

        if kplist is None:
            kplist = np.arange(NK) + 1
        else:
            # kplist-=1 #files start from 1 in W90
            kplist = np.array([k for k in kplist if k >= 0 and k < NK])
        found_atoms = False
        # todo : use an iterator to avoid double looping over lines between
        # "begin" and "end"
        iterwin = iter(fwin)

        def check_end(name):
            s = next(iterwin)
            if " ".join(s) != "end " + name:
                raise RuntimeError(
                    "expected 'end {}, found {}'".format(name, " ".join(s))
                )

        for l in iterwin:
            if l[0].startswith("begin"):
                if l[1] == "unit_cell_cart":
                    if self.Lattice is not None:
                        raise RuntimeError(
                            "'begin unit_cell_cart' found more then once  in {}.win".format(
                                prefix
                            )
                        )
                    j = 0
                    l1 = next(iterwin)
                    if l1[0] in ("bohr", "ang"):
                        units = l1[0]
                        L = [next(iterwin) for i in range(3)]
                    else:
                        units = "ang"
                        L = l1 + [next(iterwin)[0] for i in range(2)]
                    self.Lattice = np.array(L, dtype=float)
                    if units == "bohr":
                        self.Lattice *= BOHR
                    check_end("unit_cell_cart")
                elif l[1] == "kpoints":
                    if kpred is not None:
                        raise RuntimeError(
                            "'begin kpoints' found more then once  in {}.win".format(
                                prefix
                            )
                        )
                    kpred = np.array(
                        [next(iterwin)[:3] for i in range(NK)], dtype=float
                    )
                    #                    kpred=np.array([kpred[j].split()[:3] for j in kplist],dtype=float)
                    check_end("kpoints")
                elif l[1].startswith("atoms_"):
                    if l[1][6:10] not in ("cart", "frac"):
                        raise RuntimeError("unrecognised block :  '{}' ".format(l[0]))
                    if found_atoms:
                        raise RuntimeError(
                            "'begin atoms_***' found more then once  in {}.win".format(
                                prefix
                            )
                        )
                    found_atoms = True
                    xred = []
                    nameat = []
                    while True:
                        l1 = next(iterwin)
                        if l1[0] == "end":
                            # if l1[1]!=l[0].split()[1]:
                            if l1[1] != l[1]:
                                #                                print (l1[1],l[0].split()[1])
                                raise RuntimeError(
                                    "'{}' ended with 'end {}'".format(
                                        " ".join(l), l1[1]
                                    )
                                )
                            break
                        nameat.append(l1[0])
                        xred.append(l1[1:4])
                    typatdic = {n: i + 1 for i, n in enumerate(set(nameat))}
                    typat = [typatdic[n] for n in nameat]
                    xred = np.array(xred)
                    if l[1][6:10] == "cart":
                        xred = xred.dot(np.linalg.inv(self.Lattice))

        #        print ("lattice vectors:\n",self.Lattice)
        self.RecLattice = (
            np.array(
                [
                    np.cross(self.Lattice[(i + 1) % 3], self.Lattice[(i + 2) % 3])
                    for i in range(3)
                ]
            )
            * 2
            * np.pi
            / np.linalg.det(self.Lattice)
        )

        self.spacegroup = SpaceGroup(
            cell=(self.Lattice, xred, typat), spinor=self.spinor
        )
        if onlysym:
            return

        feig = prefix + ".eig"
        eigenval = np.loadtxt(prefix + ".eig")
        try:
            if eigenval.shape[0] != NBin * NK:
                raise RuntimeError("wrong number of entries ")
            ik = np.array(eigenval[:, 1]).reshape(NK, NBin)
            if not np.all(
                ik == np.arange(1, NK + 1)[:, None] * np.ones(NBin, dtype=int)[None, :]
            ):
                raise RuntimeError("wrong k-point indices")
            ib = np.array(eigenval[:, 0]).reshape(NK, NBin)
            if not np.all(
                ib == np.arange(1, NBin + 1)[None, :] * np.ones(NK, dtype=int)[:, None]
            ):
                raise RuntimeError("wrong band indices")
            eigenval = eigenval[:, 2].reshape(NK, NBin)
        except Exception as err:
            raise RuntimeError(" error reading {} : {}".format(feig,err))
  

        IBstart = 0 if (IBstart is None or IBstart <= 0) else IBstart - 1
        if IBend is None or IBend <= 0 or IBend > NBin:
            IBend = NBin
        NBout = IBend - IBstart
        if NBout <= 0:
            raise RuntimeError("No bands to calculate")

        #        print ("eigenvalues are : ",eigenval)
        self.kpoints = [
            Kpoint(
                ik - 1,
                NBin,
                IBstart,
                IBend,
                Ecut,
                None,
                self.RecLattice,
                SG=self.spacegroup,
                spinor=self.spinor,
                code="wannier",
                eigenval=eigenval[ik - 1],
                kpt=kpred[ik - 1],
            )
            for ik in kplist
        ]

    def __init_espresso(
        self,
        prefix,
        Ecut=None,
        IBstart=None,
        IBend=None,
        kplist=None,
        EF=None,
        onlysym=False,
        spin_channel=None
    ):
        import xml.etree.ElementTree as ET

        mytree = ET.parse(prefix + ".save/data-file-schema.xml")
        myroot = mytree.getroot()

        inp = myroot.find("input")
        outp = myroot.find("output")
        bandstr = outp.find("band_structure")
        ntyp = int(inp.find("atomic_species").attrib["ntyp"])
        atnames = [
            sp.attrib["name"] for sp in inp.find("atomic_species").findall("species")
        ]
        assert len(atnames) == ntyp
        atnumbers = {atn: i + 1 for i, atn in enumerate(atnames)}
        self.spinor = str2bool(bandstr.find("noncolin").text)
        #        print ('spinor=',self.spinor)
        struct = inp.find("atomic_structure")
        nat = struct.attrib["nat"]
        self.Lattice = BOHR * np.array(
            [
                struct.find("cell").find("a{}".format(i + 1)).text.strip().split()
                for i in range(3)
            ],
            dtype=float,
        )
        xcart = []
        typat = []
        for at in struct.find("atomic_positions").findall("atom"):
            typat.append(atnumbers[at.attrib["name"]])
            xcart.append(at.text.split())
        xred = (np.array(xcart, dtype=float) * BOHR).dot(np.linalg.inv(self.Lattice))
        #        print ("xred=",xred)
        self.spacegroup = SpaceGroup(
            cell=(self.Lattice, xred, typat), spinor=self.spinor
        )
        if onlysym:
            return
        Ecut0 = float(inp.find("basis").find("ecutwfc").text) * Hartree_eV

        IBstartE=0
        try:
            NBin_dw=int(bandstr.find('nbnd_dw').text)
            NBin_up=int(bandstr.find('nbnd_up').text)
            spinpol=True
            print ("spin-polarised bandstructure composed of {} up and {} dw states".format(NBin_dw,NBin_up))
            NBin_dw+NBin_up
        except AttributeError as err: 
            spinpol=False
            NBin=int(bandstr.find('nbnd').text)


        IBstartE=0
        if self.spinor and spinpol:
            raise RuntimeError("bandstructure cannot be both noncollinear and spin-polarised. Smth is wrong with the 'data-file-schema.xml'")
        if spinpol :
            if spin_channel is None:
                raise ValueError("Need to select a spin channel for spin-polarised calculations set  'up' or 'dw'")
            assert (spin_channel in ['dw','up'])
            if spin_channel=='dw':
                IBstartE=NBin_up
                NBin=NBin_dw
            else : 
                NBin=NBin_up
        else :
            if spin_channel is not None:
                raise ValueError("Found a non-polarized bandstructure, but spin channel is set to {}".format(spin_channel))


        IBstart = 0 if (IBstart is None or IBstart <= 0) else IBstart - 1
        if IBend is None or IBend <= 0 or IBend > NBin:
            IBend = NBin
        NBout = IBend - IBstart
        if NBout <= 0:
            raise RuntimeError("No bands to calculate")
        if Ecut is None or Ecut > Ecut0 or Ecut <= 0:
            Ecut = Ecut0

        self.Ecut = Ecut
        self.RecLattice = (
            np.array(
                [
                    np.cross(self.Lattice[(i + 1) % 3], self.Lattice[(i + 2) % 3])
                    for i in range(3)
                ]
            )
            * 2
            * np.pi
            / np.linalg.det(self.Lattice)
        )
        try:
            self.efermi = float(bandstr.find("fermi_energy").text) * Hartree_eV
        except Exception as err:
            print("WARNING : fermi_energy not found.Setting as zero : `{}`".format(err))
            self.efermi = 0
        kpall = bandstr.findall("ks_energies")
        NK = len(kpall)
        if kplist is None:
            kplist = np.arange(NK)
        else:
            kplist -= 1
            kplist = np.array([k for k in kplist if k >= 0 and k < NK])
        #        print ("kplist",kplist)
        #        for kp in kpall:
        #            print(kp.find('k_point').text)
        self.kpoints = []
        flag = -1
        for ik in kplist:
            kp = Kpoint(
                ik,
                NBin,
                IBstart,
                IBend,
                Ecut,
                Ecut0,
                self.RecLattice,
                SG=self.spacegroup,
                spinor=self.spinor,
                code="espresso",
                kptxml=kpall[ik],
                prefix=prefix,
                spin_channel=spin_channel,
                IBstartE=IBstartE
            )
            self.kpoints.append(kp)
            flag = ik

    #        tagname= mytree.getElementsByTagName('item')[0]
    #        print(tagname)
    # If I try to fetch the first ele
    #        myroot = mytree.getroot()
    #        print (myroot)
    #        exit()

    def getNK():
        return len(self.kpoints)

    NK = property(getNK)

    def write_characters(
        self,
        degen_thresh=0,
        refUC=None,
        shiftUC=np.zeros(3),
        kpnames=None,
        symmetries=None,
        preline="",
        plotFile=None,
    ):
        #        if refUC is not None:
        #        self.spacegroup.show(refUC=refUC,shiftUC=shiftUC)
        #        self.spacegroup.show2(refUC=refUC)
        kpline = self.KPOINTSline()
        try:
            pFile = open(plotFile, "w")
        except BaseException:
            pFile = None
        NBANDINV = 0
        GAP = np.Inf
        Low = -np.Inf
        Up = np.inf
        if kpnames is not None and refUC is not None:
            for kpname, KP in zip(kpnames, self.kpoints):
                irreps = self.spacegroup.get_irreps_from_table(
                    refUC, shiftUC, kpname, KP.K
                )
                ninv, low, up = KP.write_characters(
                    degen_thresh,
                    irreptable=irreps,
                    symmetries=symmetries,
                    preline=preline,
                    efermi=self.efermi,
                    plotFile=pFile,
                    kpl=kpline,
                )
                NBANDINV += ninv
                GAP = min(GAP, up - low)
                Up = min(Up, up)
                Low = max(Low, low)
        else:
            for KP, kpl in zip(self.kpoints, kpline):
                ninv, low, up = KP.write_characters(
                    degen_thresh,
                    symmetries=symmetries,
                    preline=preline,
                    efermi=self.efermi,
                    plotFile=pFile,
                    kpl=kpl,
                )
                NBANDINV += ninv
                GAP = min(GAP, up - low)
                Up = min(Up, up)
                Low = max(Low, low)

        if self.spinor:
            print(
                "number of inversions-odd Kramers pairs IN THE LISTED KPOINTS: ",
                int(NBANDINV / 2),
                "  Z4= ",
                int(NBANDINV / 2) % 4,
            )
        else:
            print("number of inversions-odd states : ", NBANDINV)

        #        print ("Total number of inversion-odd Kramers pairs IN THE LISTED KPOINTS: ",NBANDINV,"  Z4= ",NBANDINV%4)
        print("Minimal direct gap:", GAP, " eV")
        print("indirect  gap:", Up - Low, " eV")

    def getNbands(self):
        # returns the number of bands (if equal over all k-points) of
        # RuntimeError otherwise
        nbarray = [k.Nband for k in self.kpoints]
        if len(set(nbarray)) > 1:
            raise RuntimeError(
                "the numbers of bands differs over k-points:{0} \n cannot write tracce.txt \n".format(
                    nbarray
                )
            )
        if len(nbarray) == 0:
            raise RuntimeError(
                "do we have any k-points??? NB={0} \n cannot write tracce.txt \n".format(
                    nbarray
                )
            )
        return nbarray[0]

    def write_trace(
        self,
        degen_thresh=0,
        refUC=None,
        shiftUC=np.zeros(3),
        kpnames=None,
        symmetries=None,
    ):
        f = open("trace.txt", "w")
        f.write(
            (
                " {0}  \n"
                + " {1}  \n"  # Number of bands below the Fermi level  # Spin-orbit coupling. No: 0, Yes: 1
            ).format(self.getNbands(), 1 if self.spinor else 0)
        )

        f.write(self.spacegroup.write_trace(refUC=refUC, shiftUC=shiftUC))
        # Number of maximal k-vectors in the space group. In the next files
        # introduce the components of the maximal k-vectors))
        f.write("  {0}  \n".format(len(self.kpoints)))
        for KP in self.kpoints:
            f.write(
                "   ".join(
                    "{0:10.6f}".format(x)
                    for x in (refUC.dot(KP.K) if refUC is not None else KP.K)
                )
                + "\n"
            )
        for KP in self.kpoints:
            f.write(
                KP.write_trace(degen_thresh, symmetries=symmetries, efermi=self.efermi)
            )

    def Separate(self, isymop, degen_thresh=1e-5, groupKramers=True):
        if isymop == 1:
            return {1: self}
        symop = self.spacegroup.symmetries[isymop - 1]
        print("Separating by symmetry operation # ", isymop)
        symop.show()
        kpseparated = [
            kp.Separate(symop, degen_thresh=degen_thresh, groupKramers=groupKramers)
            for kp in self.kpoints
        ]
        allvalues = np.array(sum((list(kps.keys()) for kps in kpseparated), []))
        #        print (allvalues)
        #        for kps in kpseparated :
        #            allvalues=allvalues | set( kps.keys())
        #        allvalues=np.array(allavalues)
        if groupKramers:
            allvalues = allvalues[np.argsort(np.real(allvalues))].real
            borders = np.hstack(
                (
                    [0],
                    np.where(abs(allvalues[1:] - allvalues[:-1]) > 0.01)[0] + 1,
                    [len(allvalues)],
                )
            )
            #            nv=len(allvalues)
            if len(borders) > 2:
                allvalues = set(
                    [allvalues[b1:b2].mean() for b1, b2 in zip(borders, borders[1:])]
                )
                subspaces = {}
                for v in allvalues:
                    other = copy.copy(self)
                    other.kpoints = []
                    for K in kpseparated:
                        vk = list(K.keys())
                        vk0 = vk[np.argmin(np.abs(v - vk))]
                        if abs(vk0 - v) < 0.05:
                            other.kpoints.append(K[vk0])
                        subspaces[v] = other
                return subspaces
            else:
                return dict({allvalues.mean(): self})
        else:
            allvalues = allvalues[np.argsort(np.angle(allvalues))]
            print("allvalues:", allvalues)
            borders = np.where(abs(allvalues - np.roll(allvalues, 1)) > 0.01)[0]
            nv = len(allvalues)
            if len(borders) > 0:
                allvalues = set(
                    [
                        np.roll(allvalues, -b1)[: (b2 - b1) % nv].mean()
                        for b1, b2 in zip(borders, np.roll(borders, -1))
                    ]
                )
                print("distinct values:", allvalues)
                subspaces = {}
                for v in allvalues:
                    other = copy.copy(self)
                    other.kpoints = []
                    for K in kpseparated:
                        vk = list(K.keys())
                        vk0 = vk[np.argmin(np.abs(v - vk))]
                        #                    print ("v,vk",v,vk)
                        #                    print ("v,vk",v,vk[np.argmin(np.abs(v-vk))])
                        if abs(vk0 - v) < 0.05:
                            other.kpoints.append(K[vk0])
                        subspaces[v] = other
                return subspaces
            else:
                return dict({allvalues.mean(): self})

    def zakphase(self):
        overlaps = [
            x.overlap(y)
            for x, y in zip(self.kpoints, self.kpoints[1:] + [self.kpoints[0]])
        ]
        print("overlaps")
        for O in overlaps:
            print(np.abs(O[0, 0]), np.angle(O[0, 0]))
        print("   sum  ", np.sum(np.angle(O[0, 0]) for O in overlaps) / np.pi)
        #        overlaps.append(self.kpoints[-1].overlap(self.kpoints[0],g=np.array( (self.kpoints[-1].K-self.kpoints[0].K).round(),dtype=int )  )  )
        nmax = np.min([o.shape for o in overlaps])
        z = np.angle(
            [[la.det(O[:n, :n]) for n in range(1, nmax + 1)] for O in overlaps]
        ).sum(axis=0) % (2 * np.pi)
        #        print (np.array([k.Energy[1:] for k in self.kpoints] ))
        #        print (np.min([k.Energy[1:] for k in self.kpoints],axis=0) )
        emin = np.hstack(
            (np.min([k.Energy[1:nmax] for k in self.kpoints], axis=0), [np.Inf])
        )
        emax = np.max([k.Energy[:nmax] for k in self.kpoints], axis=0)
        locgap = np.hstack(
            (
                np.min(
                    [k.Energy[1:nmax] - k.Energy[0 : nmax - 1] for k in self.kpoints],
                    axis=0,
                ),
                [np.Inf],
            )
        )
        return z, emin - emax, (emin + emax) / 2, locgap

    def wcc(self):
        overlaps = [
            x.overlap(y)
            for x, y in zip(self.kpoints, self.kpoints[1:] + [self.kpoints[0]])
        ]
        nmax = np.min([o.shape for o in overlaps])
        wilson = functools.reduce(
            np.dot,
            [functools.reduce(np.dot, np.linalg.svd(O)[0:3:2]) for O in overlaps],
        )
        return np.sort((np.angle(np.linalg.eig(wilson)) / (2 * np.pi)) % 1)

    def write_bands(self, locs=None):
        #        print (locs)
        kpline = self.KPOINTSline()
        nbmax = max(k.Nband for k in self.kpoints)
        EN = np.zeros((nbmax, len(kpline)))
        EN[:, :] = np.Inf
        for i, k in enumerate(self.kpoints):
            EN[: k.Nband, i] = k.Energy - self.efermi
        if locs is not None:
            loc = np.zeros((nbmax, len(kpline), len(locs)))
            for i, k in enumerate(self.kpoints):
                loc[: k.Nband, i, :] = k.getloc(locs).T
            return "\n\n\n".join(
                "\n".join(
                    (
                        "{0:8.4f}   {1:8.4f}  ".format(k, e)
                        + "  ".join("{0:8.4f}".format(l) for l in L)
                    )
                    for k, e, L in zip(kpline, E, LC)
                )
                for E, LC in zip(EN, loc)
            )
        else:
            return "\n\n\n".join(
                "\n".join(
                    ("{0:8.4f}   {1:8.4f}  ".format(k, e)) for k, e in zip(kpline, E)
                )
                for E in EN
            )

    def write_trace_all(
        self,
        degen_thresh=0,
        refUC=None,
        shiftUC=np.zeros(3),
        symmetries=None,
        fname="trace_all.dat",
    ):
        f = open(fname, "w")
        kpline = self.KPOINTSline()
        f.write(
            (
                "# {0}  # Number of bands below the Fermi level\n"
                + "# {1}  # Spin-orbit coupling. No: 0, Yes: 1\n"  #
            ).format(self.getNbands(), 1 if self.spinor else 0)
        )
        f.write(
            "\n".join(
                ("#" + l)
                for l in self.spacegroup.write_trace(
                    refUC=refUC, shiftUC=shiftUC
                ).split("\n")
            )
            + "\n\n"
        )
        for KP, KPL in zip(self.kpoints, kpline):
            f.write(
                KP.write_trace_all(
                    degen_thresh, symmetries=symmetries, efermi=self.efermi, kpline=KPL
                )
            )

    def KPOINTSline(self, kpred=None, breakTHRESH=0.1):
        if kpred is None:
            kpred = [k.K for k in self.kpoints]
        KPcart = np.array(kpred).dot(self.RecLattice)
        K = np.zeros(KPcart.shape[0])
        k = np.linalg.norm(KPcart[1:, :] - KPcart[:-1, :], axis=1)
        k[k > breakTHRESH] = 0.0
        K[1:] = np.cumsum(k)
        return K
