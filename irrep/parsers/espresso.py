import os
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.io import FortranFile as FF

from ..gvectors import Hartree_eV
from ..utility import BOHR, log_message, str2bool
from .common import ParserCommon


class ParserEspresso(ParserCommon):
    """Parser for Quantum Espresso outputs."""

    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix
        mytree = ET.parse(prefix + ".save/data-file-schema.xml")
        myroot = mytree.getroot()

        self.input = myroot.find("input")
        outp = myroot.find("output")
        self.bandstr = outp.find("band_structure")
        self.spinor = str2bool(self.bandstr.find("noncolin").text)
        recip_lattice_alat = outp.find("basis_set").find("reciprocal_lattice")
        recip_lattice_alat = np.array([recip_lattice_alat.find(f"b{i}").text.split() for i in range(1, 4)], dtype=float)
        self.recip_lattice_alat_inv = np.linalg.inv(recip_lattice_alat)

    def parse_header(self, spin_channel=None):
        Ecut0 = float(self.input.find("basis").find("ecutwfc").text)
        Ecut0 *= Hartree_eV
        NK = len(self.bandstr.findall("ks_energies"))

        self.spin_channel = spin_channel
        try:
            NBin_dw = int(self.bandstr.find("nbnd_dw").text)
            NBin_up = int(self.bandstr.find("nbnd_up").text)
            self.spinpol = True
            print(f"spin-polarised bandstructure composed of {NBin_up} up and {NBin_dw} dw states")
        except AttributeError:
            self.spinpol = False
            NBin = int(self.bandstr.find("nbnd").text)

        try:
            EF = float(self.bandstr.find("fermi_energy").text) * Hartree_eV
        except Exception:
            EF = None

        if self.spinpol:
            self.NBin_list = [NBin_up, NBin_dw]
        else:
            self.NBin_list = [NBin]

        if self.spinor and self.spinpol:
            raise RuntimeError(
                "bandstructure cannot be both noncollinear and spin-polarised. "
                "Smth is wrong with the 'data-file-schema.xml'"
            )
        elif self.spinpol:
            if spin_channel is None:
                raise ValueError("Need to select a spin channel for spin-polarised calculations set  'up' or 'dw'")
            assert spin_channel in ["dw", "up"]
            if spin_channel == "dw":
                NBin = self.NBin_list[1]
            else:
                NBin = self.NBin_list[0]
        else:
            NBin = self.NBin_list[0]
            if spin_channel is not None:
                raise ValueError(f"Found a non-polarized bandstructure, but spin channel is set to {spin_channel}")

        return self.spinpol, Ecut0, EF, NK, NBin

    def parse_lattice(self):
        ntyp = int(self.input.find("atomic_species").attrib["ntyp"])
        struct = self.input.find("atomic_structure")
        nat = int(struct.attrib["nat"])
        alat = float(struct.attrib["alat"])
        del nat

        lattice = []
        for i in range(3):
            lattice.append(struct.find("cell").find(f"a{i + 1}").text.strip().split())
        lattice = BOHR * np.array(lattice, dtype=float)

        positions = []
        for at in struct.find("atomic_positions").findall("atom"):
            positions.append(at.text.split())
        positions = np.dot(np.array(positions, dtype=float) * BOHR, np.linalg.inv(lattice))

        atnames = []
        for sp in self.input.find("atomic_species").findall("species"):
            atnames.append(sp.attrib["name"])
        if len(atnames) != ntyp:
            raise RuntimeError(
                "Error in the assigment of atom species. Probably a bug in Quantum Espresso, but your DFT input files"
            )
        atnumbers = {atn: i + 1 for i, atn in enumerate(atnames)}
        typat = []
        for at in struct.find("atomic_positions").findall("atom"):
            typat.append(atnumbers[at.attrib["name"]])

        return lattice, positions, typat, alat

    def get_kpt_coord(self, ik):
        kptxml = self.bandstr.findall("ks_energies")[ik]
        kpt = np.array(kptxml.find("k_point").text.split(), dtype=float)
        kpt = kpt.dot(self.recip_lattice_alat_inv)
        return kpt

    def parse_kpoint(self, ik, verbosity=0, getE=True, getWF=True):
        kptxml = self.bandstr.findall("ks_energies")[ik]
        if self.spinpol:
            if self.spin_channel == "up":
                NB_skip = 0
                NBin = self.NBin_list[0]
            else:
                NB_skip = self.NBin_list[0]
                NBin = self.NBin_list[1]
        else:
            NB_skip = 0
            NBin = self.NBin_list[0]

        if getE:
            Energy = np.array(kptxml.find("eigenvalues").text.split(), dtype=float)[NB_skip: NB_skip + NBin]
            Energy *= Hartree_eV
        else:
            Energy = None
        npw = int(kptxml.find("npw").text)
        nspinor = 2 if self.spinor else 1
        npwtot = npw * nspinor

        if getWF:
            wfcname = f"wfc{'' if self.spin_channel is None else self.spin_channel}{ik + 1}"
            fWFC = None
            checked_files = []
            for extension in ["hdf5", "dat"]:
                for strcase in [str.lower, str.upper]:
                    print(f"{ik=}, {wfcname=}, {extension=}, {strcase=}")
                    filename = f"{self.prefix}.save/{strcase(wfcname)}.{extension}"
                    if os.path.exists(filename):
                        if extension == "hdf5":
                            fWFC = h5py.File(filename, "r")
                            attrnames = [
                                "gamma_only",
                                "igwx",
                                "ik",
                                "ispin",
                                "nbnd",
                                "ngw",
                                "npol",
                                "scale_factor",
                                "xk",
                            ]
                            attributes = []
                            for atr in attrnames:
                                attributes.append(fWFC.attrs[atr])

                            _gamma_only, _igwx, ik_read, _ispin, _nbnd, _ngw, _npol, _scale_factor, xk = attributes
                            assert ik_read == ik + 1, f"Found ik={ik_read} in the wavefunction file {filename}, but expected ik={ik + 1}"
                            kpt = np.array(xk)
                            Miller_Indices = fWFC["MillerIndices"]
                            B = np.array([Miller_Indices.attrs[f"bg{i}"] for i in range(1, 4)])
                            kg = np.array(Miller_Indices[::])
                            kpt = kpt.dot(np.linalg.inv(B))
                            evc = np.array(fWFC["evc"], dtype=float)
                            WF = evc[:, 0::2] + 1.0j * evc[:, 1::2]
                        else:
                            fWFC = FF(filename, "r")
                            rec = fWFC.read_record("i4,3f8,i4,i4,f8")[0]
                            kpt = rec[1]

                            rec = fWFC.read_record("4i4")
                            igwx = rec[1]

                            rec = fWFC.read_record("(3,3)f8")
                            B = np.array(rec)
                            rec = fWFC.read_record(f"({igwx},3)i4")
                            kg = np.array(rec)
                            log_message(f"npwtot: {npwtot}, igwx: {igwx}", verbosity, 2)
                            kpt = kpt.dot(np.linalg.inv(B))
                            WF = np.zeros((NBin, npwtot), dtype=complex)
                            for ib in range(NBin):
                                rec = fWFC.read_record(f"{npwtot * 2}f8")
                                WF[ib] = rec[0::2] + 1.0j * rec[1::2]
                        assert np.allclose(kpt, self.get_kpt_coord(ik)), f"Found kpt[{ik}] {kpt} in the wavefunction file {filename}, but expected {self.get_kpt_coord(ik)}"
                        WF = WF.reshape((NBin, npw, nspinor), order="F")
                        return WF, Energy, kg, kpt
                    checked_files.append(filename)
        raise RuntimeError(f"Wavefunction file not found. Tried files: {checked_files}")
