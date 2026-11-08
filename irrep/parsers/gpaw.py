import numpy as np

from irrep.parsers.common import ParserCommon

from ..gvectors import calc_gvectors
from ..utility import log_message


class ParserGPAW(ParserCommon):
    """Parser for GPAW interface."""

    spin_channels = {"up": 0, "dw": 1, None: 0, 0: 0, 1: 1}

    def __init__(self, calculator, spinor=False, spin_channel=None, verbosity=0):
        super().__init__()
        spin_channel = self.spin_channels[spin_channel]
        from gpaw import GPAW

        if isinstance(calculator, str):
            calculator = GPAW(calculator)
        self.calculator = calculator
        self.nband = self.calculator.get_number_of_bands()
        self.spinor = spinor
        self.verbosity = verbosity

        if self.spinor:
            nspins = self.calculator.get_number_of_spins()
            self.spin_channels = np.arange(nspins)
        else:
            self.spin_channels = [spin_channel]

    def parse_header(self):
        kpred = self.calculator.get_ibz_k_points()
        Lattice = self.calculator.atoms.cell
        typat = self.calculator.atoms.get_atomic_numbers()
        positions = self.calculator.atoms.get_scaled_positions()
        EF_in = self.calculator.get_fermi_level()
        return (
            self.nband * (1 + int(self.spinor)),
            kpred,
            Lattice,
            self.spinor,
            typat,
            positions,
            EF_in,
        )

    def get_kpt_coord(self, ik):
        return self.calculator.get_ibz_k_points()[ik]

    def parse_kpoint(self, ik, RecLattice, Ecut):
        WFupdw = [
            np.array(
                [
                    self.calculator.get_pseudo_wave_function(kpt=ik, band=ib, periodic=True, spin=ispin)
                    for ib in range(self.nband)
                ]
            )
            for ispin in self.spin_channels
        ]
        Eupdw = [self.calculator.get_eigenvalues(kpt=ik, spin=ispin) for ispin in self.spin_channels]

        log_message(f"shapes of WFupdw: {[wf.shape for wf in WFupdw]}", self.verbosity, 2)
        ngx, ngy, ngz = WFupdw[0].shape[1:]
        kpt = self.calculator.get_ibz_k_points()[ik]
        assert np.allclose(kpt, self.get_kpt_coord(ik)), f"Found kpt[{ik}] {kpt} in the wavefunction file, but expected {self.get_kpt_coord(ik)}"
        kg, eKG = calc_gvectors(
            kpt,
            RecLattice,
            Ecut,
            spinor=False,
            verbosity=self.verbosity,
        )
        selectG = (kg[:, 0:3].T).copy()
        selectG[0, :] = selectG[0, :] % ngx
        selectG[1, :] = selectG[1, :] % ngy
        selectG[2, :] = selectG[2, :] % ngz
        selectG = tuple(selectG)

        for i in range(len(WFupdw)):
            WFupdw[i] = np.fft.fftn(WFupdw[i], axes=(1, 2, 3))
            WFupdw[i] = np.array([wf[selectG] for wf in WFupdw[i]])
        if self.spinor:
            if len(WFupdw) == 1:
                WFupdw = WFupdw * 2
                Eupdw = Eupdw * 2
            WF = WFupdw[0]
            WFspinor = np.zeros((2 * WF.shape[0], WF.shape[1], 2), dtype=complex)
            H = get_soc_gpaw(self.calculator, ik, flatten=True)
            rng = np.arange(0, 2 * self.nband, 2)
            for s in range(2):
                H[rng + s, rng + s] += Eupdw[s]
            E, V = np.linalg.eigh(H)
            V = V.T
            for s in range(2):
                WFspinor[:, :, s] = V[:, s::2] @ WFupdw[s]
            energies = E
            WFout = WFspinor
        else:
            WFout = WFupdw[0][:, :, None]
            energies = self.calculator.get_eigenvalues(kpt=ik, spin=self.spin_channels[0])

        return energies, WFout, kg, kpt, eKG


def get_soc_gpaw(calc, ik, flatten=True, theta=0, phi=0, legacy_gpaw=None):
    """Calculate SOC Hamiltonian in KS basis for a given k-point."""
    from ase.units import Hartree
    from gpaw.spinorbit import soc

    legacy_gpaw = check_legacy_gpaw(calc, legacy_gpaw)
    nspin = calc.get_number_of_spins()

    C_ss = np.array(
        [
            [
                np.cos(theta / 2) * np.exp(-1.0j * phi / 2),
                -np.sin(theta / 2) * np.exp(-1.0j * phi / 2),
            ],
            [
                np.sin(theta / 2) * np.exp(1.0j * phi / 2),
                np.cos(theta / 2) * np.exp(1.0j * phi / 2),
            ],
        ]
    )

    dVL_avii = {
        a: soc(calc.wfs.setups[a], calc.hamiltonian.xc, D_sp)
        for a, D_sp in calc.density.D_asp.items()
    }

    m = calc.get_number_of_bands()
    nk = len(calc.get_ibz_k_points())
    assert ik < nk, f"ik = {ik} >= nk = {nk}"

    h_soc = np.zeros((2, 2, m, m), complex)

    H_a = {}
    for a, dVL_vii in dVL_avii.items():
        ni = dVL_vii.shape[1]
        H_ssii = np.zeros((2, 2, ni, ni), complex)
        H_ssii[0, 0] = dVL_vii[2]
        H_ssii[0, 1] = dVL_vii[0] - 1j * dVL_vii[1]
        H_ssii[1, 0] = dVL_vii[0] + 1j * dVL_vii[1]
        H_ssii[1, 1] = -dVL_vii[2]
        H_a[a] = H_ssii

    q = ik
    for a, H_ssii in H_a.items():
        h_ssii = np.einsum("ab,bcij,cd->adij", C_ss.T.conj(), H_ssii, C_ss, optimize=True)
        for s1 in range(2):
            for s2 in range(2):
                h_ii = h_ssii[s1, s2]
                if legacy_gpaw:
                    if nspin == 2:
                        P1_mi = calc.wfs.kpt_qs[q][s1].P_ani[a]
                        P2_mi = calc.wfs.kpt_qs[q][s2].P_ani[a]
                    else:
                        P1_mi = calc.wfs.kpt_qs[q][0].P_ani[a]
                        P2_mi = P1_mi
                else:
                    if nspin == 2:
                        # maybe need to use get_wfs_on_master for parallel runs, but needs testing
                        P1_mi = calc.dft.ibzwfs._get_wfs(kpt=q, spin=s1).P_ani[a]
                        P2_mi = calc.dft.ibzwfs._get_wfs(kpt=q, spin=s2).P_ani[a]
                    else:
                        P1_mi = calc.dft.ibzwfs._get_wfs(kpt=q, spin=0).P_ani[a]
                        P2_mi = P1_mi

                h_soc[s1, s2] += np.dot(np.dot(P1_mi.conj(), h_ii), P2_mi.T)
    if flatten:
        h_soc_old = h_soc.copy()
        h_soc = np.zeros((2 * m, 2 * m), complex)
        for s1 in range(2):
            for s2 in range(2):
                h_soc[s1::2, s2::2] = h_soc_old[s1, s2]
    return h_soc * Hartree


def check_legacy_gpaw(calc, legacy_gpaw=None):
    if legacy_gpaw is not None:
        return legacy_gpaw
    else:
        return not is_new_gpaw(calc)


def is_new_gpaw(calc):
    # "New GPAW" (default in 26): ASECalculator exposing .dft
    # Legacy calculator (25.x, or 26 with legacy_gpaw=True): real calc.wfs
    if hasattr(calc, '_wfs_dft'):
        return True
    if hasattr(calc, 'get_realspace_wfs'):
        return False
    raise ValueError("Cannot determine if GPAW is new or legacy. Please check the GPAW version and calculator type.")



def is_collinear(calc):
    if hasattr(calc, 'dft'):                 # new GPAW (26 default)
        return calc.dft.ibzwfs.collinear
    return calc.wfs.collinear


def get_magnetic_moments(calc, project_collinear_axis=(0, 0, 1), legacy_gpaw=None):
    legacy_gpaw = check_legacy_gpaw(calc, legacy_gpaw)
    if legacy_gpaw:
        magmoms = calc.get_magnetic_moments()
    else:
        magmoms = calc.dft.magmoms()[1]
        if is_collinear(calc) and project_collinear_axis:
            magmoms = np.dot(magmoms, project_collinear_axis)
    return magmoms


@staticmethod
def read_gpaw_wf(calc, ibz_index, ispin, IBstart, IBend, RecLattice, legacy_gpaw=None):
    legacy_gpaw = check_legacy_gpaw(calc, legacy_gpaw)
    k = calc.get_ibz_k_points()[ibz_index]

    if legacy_gpaw:
        P_ani = calc.wfs.kpt_qs[ibz_index][ispin].P_ani
        if RecLattice is None:
            RecLattice = calc.wfs.gd.reciprocal_lattice
        wavefunction = np.array([
            calc.wfs.get_wave_function_array(n, ibz_index, ispin, periodic=True)
            for n in range(IBstart, IBend)])
    else:
        wfs = calc.dft.ibzwfs._get_wfs(kpt=ibz_index, spin=ispin)
        P_ani = wfs.P_ani.to_cpu()              # AtomArrays; P_ani[a] -> (nbands, nproj_a)
        wavefunction = calc.dft.wave_functions(n1=IBstart, n2=IBend, kpt=ibz_index, spin=ispin, periodic=True).data
        from ase.units import Bohr
        wavefunction *= Bohr ** 1.5

        if RecLattice is None:
            RecLattice = 2 * np.pi * np.linalg.inv(calc.atoms.cell.array).T  # check units vs 25.7

    proj = np.hstack([P_ani[a][IBstart:IBend] for a in range(len(calc.atoms))])

    atom_positions = calc.atoms.get_scaled_positions()

    return k, proj, wavefunction, RecLattice, atom_positions
