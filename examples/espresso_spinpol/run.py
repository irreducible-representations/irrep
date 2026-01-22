import irrep
import irrep.bandstructure
bandstructure  = irrep.bandstructure.BandStructure(prefix="Fe",
                                                       code="espresso",
                                                       onlysym=True,
                                                         spin_channel='dw'
                                                       )
bandstructure.spacegroup.write_sym_file("Fe.sym")