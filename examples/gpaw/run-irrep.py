from gpaw import GPAW
from irrep.bandstructure import BandStructure
calc = GPAW("Bi-gamma.gpw")

spinor = True
nspinor = 2 if spinor else 1

print("Fermi level", calc.get_fermi_level())
bandstructure = BandStructure(code="gpaw", calculator_gpaw=calc,
                                spinor=spinor,
                                IBstart=10*nspinor+1,
                                IBend=15*nspinor,
                                Ecut=50, 
                                degen_thresh=2e-4,
                                calculate_traces=True,
                                irreps=True,
                                search_cell=True,
                                kplist=[1])
bandstructure.spacegroup.show()
bandstructure.identify_irreps(kpnames=["GM"], verbosity=0)
bandstructure.write_characters(refcell=False)

bandstructure.write_trace()
