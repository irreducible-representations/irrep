import irrep
import irrep.bandstructure
bandstructure  = irrep.bandstructure.BandStructure.from_espresso(prefix="Fe",
                                                       onlysym=True,
                                                         spin_channel='dw'
                                                       )
bandstructure.spacegroup.write_sym_file("Fe.sym")