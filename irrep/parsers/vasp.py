import numpy as np

from irrep.parsers.common import ParserCommon

from ..utility import log_message


class WAVECARFILE:
    """Low-level reader for VASP WAVECAR records."""

    def __init__(self, filename, RL=3, verbosity=0):
        self.verbosity = verbosity
        self.f = open(filename, "rb")
        self.rl = RL
        self.rl, ispin, iprec = [int(x) for x in self.record(0)]
        self.iprec = iprec
        log_message(f"iprec tag = {iprec}, record_length = {self.rl} bytes", self.verbosity, 1)
        if iprec not in (45200, 53300):
            raise RuntimeError(
                f"invalid iprec tag found: {iprec}, probably not a single-precision file. "
                "Double-precision is not supported"
            )
        if ispin != 1:
            raise RuntimeError(
                "WAVECAR contains spin-polarized non-spinor wavefunctions."
                f"ISPIN={ispin}  this is not supported yet"
            )
        self.nrec_enocc = None
        self.nrec_kpoint = None
        self.nrec_header = 2

    def set_nrec_kpoint(self, NBin):
        size_enocc = (4 + 3 * NBin) * 8
        self.nrec_enocc = (size_enocc + self.rl - 1) // self.rl
        log_message(
            f"number of records for energies and occupancies : {self.nrec_enocc}",
            self.verbosity,
            1,
        )
        if self.iprec in (42200, 42210):
            assert self.nrec_enocc == 1, (
                f"energies and occupancies for tag {self.iprec} should fit in one record. However, "
                f"the record length is {self.rl} bytes, which does not fit 4 + 3*{NBin} = "
                f"{(4 + 3 * NBin)}*8 = {size_enocc} bytes for {NBin} bands"
            )
        self.nrec_kpoint = NBin + self.nrec_enocc

    def record(self, irec, cnt=np.inf, dtype=float):
        """Read a WAVECAR record."""
        self.f.seek(irec * self.rl)
        return np.fromfile(self.f, dtype=dtype, count=min(self.rl, cnt))

    def irec_start_k(self, ik):
        return self.nrec_header + ik * self.nrec_kpoint

    def record_k_header(self, ik):
        rec_start = self.irec_start_k(ik)
        return np.hstack([self.record(rec_start + i) for i in range(self.nrec_enocc)])

    def record_k_band(self, ik, ib, cnt=np.inf):
        irec = self.irec_start_k(ik) + self.nrec_enocc + ib
        return self.record(irec, cnt=cnt, dtype=np.complex64)


class ParserVasp(ParserCommon):
    """Parser for VASP POSCAR and WAVECAR files."""

    def __init__(self, fPOS, fWAV, onlysym=False, verbosity=0):
        super().__init__()
        self.verbosity = verbosity
        self.fPOS = fPOS
        if not onlysym:
            self.fWAV = WAVECARFILE(fWAV, verbosity=self.verbosity)

    def parse_poscar(self):
        log_message(f"Reading POSCAR: {self.fPOS}", self.verbosity, 1)
        fpos = (l.strip() for l in open(self.fPOS))
        title = next(fpos)
        del title
        lattice = float(next(fpos)) * np.array([next(fpos).split() for i in range(3)], dtype=float)
        try:
            nat = np.array(next(fpos).split(), dtype=int)
        except BaseException:
            nat = np.array(next(fpos).split(), dtype=int)

        typat = [i + 1 for i in range(len(nat)) for j in range(nat[i])]

        l = next(fpos)
        if l[0] in ["s", "S"]:
            l = next(fpos)
        cartesian = False
        if l[0].lower() == "c":
            cartesian = True
        elif l[0].lower() != "d":
            raise RuntimeError('only "direct" or "cartesian"atomic coordinates are supproted')
        positions = np.zeros((np.sum(nat), 3))
        i = 0
        for l in fpos:
            if i >= sum(nat):
                break
            try:
                positions[i] = np.array(l.split()[:3])
                i += 1
            except Exception as err:
                log_message(err, self.verbosity, 1)
                pass
        if sum(nat) != i:
            raise RuntimeError(f"not all atomic positions were read : {i} of {sum(nat)}")
        if cartesian:
            positions = positions.dot(np.linalg.inv(lattice))
        return lattice, positions, typat

    def parse_header(self):
        tmp = self.fWAV.record(1)
        NK = int(tmp[0])
        NBin = int(tmp[1])
        self.fWAV.set_nrec_kpoint(NBin=NBin)
        Ecut0 = tmp[2]
        lattice = np.array(tmp[3:12]).reshape(3, 3)
        return NK, NBin, Ecut0, lattice

    def parse_kpoint(self, ik, NBin, spinor):
        r = self.fWAV.record_k_header(ik)
        nspinor = 2 if spinor else 1
        npw = int(r[0])
        log_message(f"npw = {npw}, nspinor = {nspinor}, NBin = {NBin}", self.verbosity, 2)
        if spinor:
            assert npw % 2 == 0, f"odd number of coefs {npw} for spinor wavefunctions"
        npw //= nspinor
        kpt = r[1:4]
        Energy = np.array(r[4: 4 + NBin * 3]).reshape(NBin, 3)[:, 0]
        WF = np.zeros((NBin, npw, nspinor), dtype=np.complex64)
        for ib in range(NBin):
            WF[ib] = self.fWAV.record_k_band(ik=ik, ib=ib, cnt=npw * nspinor).reshape((npw, nspinor), order="F")
        return WF, Energy, kpt, npw
