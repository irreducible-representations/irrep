from irrep.utility import cached_einsum


class OverlapPAW:

    def __init__(self, wfs):
        self.dO_aii = {}
        for a in wfs.kpt_u[0].projections.map:
            self.dO_aii[a] = wfs.setups[a].dO_ii
        self.dv = wfs.gd.dv

    def product(self, KP1, KP2):
        wf1 = KP1.wavefunction
        proj1 = KP1.proj
        wf2 = KP2.wavefunction
        proj2 = KP2.proj
        prod = cached_einsum('aijk,bijk->ab', wf1.conj(), wf2) * self.dv
        for a, dO_ii in self.dO_aii.items():
            prod += (proj1[a].conj() @ dO_ii @ (proj2[a].T))
        return prod
