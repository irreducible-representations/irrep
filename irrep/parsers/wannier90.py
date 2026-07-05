import os

import numpy as np
from scipy.io import FortranFile as FF

from ..utility import BOHR, split, str2bool
from .common import ParserCommon


class ParserW90(ParserCommon):
    """Parser for Wannier90 inputs and UNK wavefunction files."""

    spin_channels = {"up": 1, "dw": 2, None: None, 1: 1, 2: 2}

    def __init__(self, prefix, unk_formatted=False, spin_channel=None):
        super().__init__()
        self.prefix = prefix
        spin_channel = self.spin_channels[spin_channel]

        self.spin_channel = spin_channel
        self.path = os.path.dirname(prefix)
        self.fwin = [l.strip().lower() for l in open(prefix + ".win").readlines()]
        self.fwin = [
            [s.strip() for s in split(l)]
            for l in self.fwin
            if len(l) > 0 and l[0] not in ("!", "#")
        ]
        self.ind = np.array([l[0] for l in self.fwin])
        self.iterwin = iter(self.fwin)
        self.unk_formatted = unk_formatted

    def parse_header(self):
        self.NBin = self.get_param("num_bands", int)
        self.spinor = str2bool(self.get_param("spinors", str))
        try:
            EF = self.get_param("fermi_energy", float, None)
        except RuntimeError:
            EF = None
        self.NK = np.prod(np.array(self.get_param("mp_grid", str).split(), dtype=int))

        return self.NK, self.NBin, self.spinor, EF

    def parse_lattice(self):
        lattice = None
        kpred = None
        found_atoms = False

        for l in self.iterwin:
            if l[0].startswith("begin"):
                if l[1] == "unit_cell_cart":
                    if lattice is not None:
                        raise RuntimeError(f"'begin unit_cell_cart' found more than once in {self.prefix}.win")
                    l1 = next(self.iterwin)
                    if l1[0] in ("bohr", "ang"):
                        units = l1[0]
                        L = [next(self.iterwin) for i in range(3)]
                    else:
                        units = "ang"
                        L = [l1] + [next(self.iterwin) for i in range(2)]
                    lattice = np.array(L, dtype=float)
                    if units == "bohr":
                        lattice *= BOHR
                    self.check_end("unit_cell_cart")

                elif l[1] == "kpoints":
                    if kpred is not None:
                        raise RuntimeError(f"'begin kpoints' found more then once in {self.prefix}.win")
                    kpred = np.zeros((self.NK, 3), dtype=float)
                    for i in range(self.NK):
                        kpred[i] = next(self.iterwin)[:3]
                    self.check_end("kpoints")

                elif l[1].startswith("atoms_"):
                    if l[1][6:10] not in ("cart", "frac"):
                        raise RuntimeError(f"unrecognised block :  '{l[0]}' ")
                    if found_atoms:
                        raise RuntimeError(f"'begin atoms_***' found more then once  in {self.prefix}.win")
                    found_atoms = True
                    positions = []
                    nameat = []
                    while True:
                        l1 = next(self.iterwin)
                        if l1[0] == "end":
                            if l1[1] != l[1]:
                                raise RuntimeError(f"'{' '.join(l)}' ended with 'end {l1[1]}'")
                            else:
                                break
                        nameat.append(l1[0])
                        positions.append(l1[1:4])
                    typatdic = {n: i + 1 for i, n in enumerate(set(nameat))}
                    typat = [typatdic[n] for n in nameat]
                    positions = np.array(positions, dtype=float)
                    if l[1][6:10] == "cart":
                        positions = positions.dot(np.linalg.inv(lattice))

        return lattice, positions, typat, kpred

    def parse_energies(self):
        feig = self.prefix + ".eig"
        Energy = np.loadtxt(self.prefix + ".eig")
        try:
            if Energy.shape[0] != self.NBin * self.NK:
                raise RuntimeError("wrong number of entries ")
            ik = np.array(Energy[:, 1]).reshape(self.NK, self.NBin)
            if not np.all(ik == np.arange(1, self.NK + 1)[:, None] * np.ones(self.NBin, dtype=int)[None, :]):
                raise RuntimeError("wrong k-point indices")
            ib = np.array(Energy[:, 0]).reshape(self.NK, self.NBin)
            if not np.all(ib == np.arange(1, self.NBin + 1)[None, :] * np.ones(self.NK, dtype=int)[:, None]):
                raise RuntimeError("wrong band indices")
            Energy = Energy[:, 2].reshape(self.NK, self.NBin)
        except Exception as err:
            raise RuntimeError(f" error reading {feig} : {err}")
        return Energy

    def parse_kpoint(self, ik, selectG):
        if self.unk_formatted:
            return self.parse_kpoint_formatted(ik, selectG)
        else:
            return self.parse_kpoint_unformatted(ik, selectG)

    def get_UNK_name(self, ik):
        if self.spin_channel is None:
            spin_channel_loc = "NC" if self.spinor else "1"
        else:
            spin_channel_loc = str(self.spin_channel).upper()
        return os.path.join(self.path, f"UNK{ik:05d}.{spin_channel_loc}")

    def parse_kpoint_formatted(self, ik, selectG):
        fname = self.get_UNK_name(ik)
        print(f"parse_kpoint_formatted: {fname}")
        fUNK = open(fname, "r")
        ngx, ngy, ngz, ik_in, nbnd = (int(x) for x in fUNK.readline().split())
        ngtot = ngx * ngy * ngz
        nspinor = 2 if self.spinor else 1
        self.check_ik_nb(ik_in, ik, nbnd, fname)
        WF_in = np.loadtxt(fUNK, dtype=complex)
        WF_in = WF_in[:, 0] + 1.0j * WF_in[:, 1]
        fUNK.close()
        ng_loc = selectG[0].shape[0]
        WF = np.zeros((self.NBin, ng_loc, nspinor), dtype=complex)
        for ib in range(self.NBin):
            for i in range(nspinor):
                i_start = ib * nspinor * ngtot + i * ngtot
                cg_tmp = WF_in[i_start: i_start + ngtot]
                cg_tmp = cg_tmp.reshape((ngx, ngy, ngz), order="F")
                cg_tmp = np.fft.fftn(cg_tmp)
                WF[ib, :, i] = cg_tmp[selectG]
        return WF

    def check_ik_nb(self, ik_in, ik, nbnd, fname):
        if ik_in != ik:
            raise RuntimeError(f"file {fname} contains point number {ik_in}, expected {ik}")
        if nbnd != self.NBin:
            raise RuntimeError(f"file {fname} contains {nbnd} bands, expected {self.NBin}")

    def parse_kpoint_unformatted(self, ik, selectG):
        fname = self.get_UNK_name(ik)
        fUNK = FF(fname, "r")
        ngx, ngy, ngz, ik_in, nbnd = fUNK.read_record("i4,i4,i4,i4,i4")[0]
        ngtot = ngx * ngy * ngz
        nspinor = 2 if self.spinor else 1

        self.check_ik_nb(ik_in, ik, nbnd, fname)

        ng_loc = selectG[0].shape[0]
        WF = np.zeros((self.NBin, ng_loc, nspinor), dtype=complex)
        for ib in range(self.NBin):
            for i in range(nspinor):
                cg_tmp = fUNK.read_record(f"{ngtot * 2}f8")
                cg_tmp = (cg_tmp[0::2] + 1.0j * cg_tmp[1::2]).reshape((ngx, ngy, ngz), order="F")
                cg_tmp = np.fft.fftn(cg_tmp)
                WF[ib, :, i] = cg_tmp[selectG]
        return WF

    def parse_grid(self, ik):
        fname = self.get_UNK_name(ik)
        if self.unk_formatted:
            fUNK = open(fname, "r")
            ngx, ngy, ngz, ik_in, nbnd = (int(x) for x in fUNK.readline().split())
            fUNK.close()
        else:
            fUNK = FF(fname, "r")
            ngx, ngy, ngz, ik_in, nbnd = fUNK.read_record("i4,i4,i4,i4,i4")[0]
            fUNK.close()
        return ngx, ngy, ngz

    def check_end(self, name):
        s = next(self.iterwin)
        if " ".join(s) != "end " + name:
            raise RuntimeError(f"expected 'end {name}', found {' '.join(s)}")

    def get_param(self, key, tp, default=None, join=False):
        i = np.where(self.ind == key)[0]
        if len(i) == 0:
            if default is None:
                raise RuntimeError(f"parameter {key} was not found in {self.prefix}.win")
            else:
                return default
        if len(i) > 1:
            raise RuntimeError(f"parameter {key} was found {len(i)} times in {self.prefix}.win")

        x = self.fwin[i[0]][1:]
        if len(x) > 1:
            if join:
                x = " ".join(x)
            else:
                raise RuntimeError(
                    f"length {len(x)} found for parameter {key}, rather than length 1 in {self.prefix}.win"
                )
        else:
            x = self.fwin[i[0]][1]
        return tp(x)
