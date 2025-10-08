import numpy as np
from irrep.utility import cached_einsum


class OverlapPAW:

    def __init__(self, wfs):
        self.dO_aii = {}
        for a in wfs.kpt_u[0].projections.map:
            self.dO_aii[a] = wfs.setups[a].dO_ii
        self.dv = wfs.gd.dv

    def product(self, KP1, KP2, 
                include_paw=True, 
                include_pseudo=True,
                normalize=False):
        wf1 = KP1.wavefunction
        proj1 = KP1.proj
        wf2 = KP2.wavefunction
        proj2 = KP2.proj
        assert wf1.ndim == 4
        assert wf2.ndim == 4
        assert wf1.shape[1:] == wf2.shape[1:], f"wavefunction grids do not match: {wf1.shape[1:]} vs {wf2.shape[1:]}"
        prod = np.zeros((wf1.shape[0], wf2.shape[0]), dtype=complex)
        if include_pseudo:
            prod += cached_einsum('aijk,bijk->ab', wf1.conj(), wf2) * self.dv
        if include_paw:
            for a, dO_ii in self.dO_aii.items():
                prod += (proj1[a].conj() @ dO_ii @ (proj2[a].T))
        if normalize:
            prod/= np.sqrt(abs(self.product(KP1, KP1, include_paw=include_paw, include_pseudo=include_pseudo).diagonal()[:, None]))
            prod/= np.sqrt(abs(self.product(KP2, KP2, include_paw=include_paw, include_pseudo=include_pseudo).diagonal()[None, :]))
        return prod
