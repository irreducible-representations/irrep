from ..bandstructure import BandStructure


def parse_files(
    code='vasp',
    fWAV=None,
    fWFK=None,
    prefix=None,
    calculator_gpaw=None,
    fPOS=None,
    Ecut=None,
    IBstart=None,
    IBend=None,
    kplist=None,
    spinor=None,
    EF='0.0',
    onlysym=False,
    spin_channel=None,
    verbosity=0,
    alat=None,
    irreps=False,
    from_sym_file=None,
    unk_formatted=False,
    spacegroup=None,
    select_grid=None,
    irreducible=False,
    read_paw=False,
    **kwargs
):
    code = code.lower()
    if code == "vasp":
        return BandStructure.from_vasp(
            fWAV=fWAV,
            fPOS=fPOS,
            Ecut=Ecut,
            IBstart=IBstart,
            IBend=IBend,
            kplist=kplist,
            spinor=spinor,
            EF=EF,
            onlysym=onlysym,
            spin_channel=spin_channel,
            verbosity=verbosity,
            irreps=irreps,
            spacegroup=spacegroup,
            select_grid=select_grid,
            irreducible=irreducible,
            **kwargs
        )

    elif code == "abinit":
        return BandStructure.from_abinit(
            fWFK=fWFK,
            Ecut=Ecut,
            IBstart=IBstart,
            IBend=IBend,
            kplist=kplist,
            EF=EF,
            onlysym=onlysym,
            spin_channel=spin_channel,
            verbosity=verbosity,
            irreps=irreps,
            spacegroup=spacegroup,
            select_grid=select_grid,
            irreducible=irreducible,
            **kwargs
        )
    elif code == "espresso":
        return BandStructure.from_espresso(
            prefix=prefix,
            Ecut=Ecut,
            IBstart=IBstart,
            IBend=IBend,
            kplist=kplist,
            EF=EF,
            onlysym=onlysym,
            spin_channel=spin_channel,
            verbosity=verbosity,
            alat=alat,
            irreps=irreps,
            from_sym_file=from_sym_file,
            spacegroup=spacegroup,
            select_grid=select_grid,
            irreducible=irreducible,
            **kwargs
        )
    elif code == "wannier90":
        return BandStructure.from_wannier90(
            prefix=prefix,
            Ecut=Ecut,
            IBstart=IBstart,
            IBend=IBend,
            kplist=kplist,
            spinor=spinor,
            EF=EF,
            onlysym=onlysym,
            spin_channel=spin_channel,
            verbosity=verbosity,
            irreps=irreps,
            unk_formatted=unk_formatted,
            spacegroup=spacegroup,
            select_grid=select_grid,
            irreducible=irreducible,
            **kwargs
        )
    elif code == "gpaw":
        return BandStructure.from_gpaw(
            calculator_gpaw=calculator_gpaw,
            Ecut=Ecut,
            IBstart=IBstart,
            IBend=IBend,
            kplist=kplist,
            spinor=spinor,
            EF=EF,
            onlysym=onlysym,
            spin_channel=spin_channel,
            verbosity=verbosity,
            irreps=irreps,
            spacegroup=spacegroup,
            select_grid=select_grid,
            irreducible=irreducible,
            read_paw=read_paw,
            **kwargs
        )
    else:
        raise ValueError(f"code {code} not recognized, must be one of 'vasp', 'abinit', 'espresso', 'wannier90' or 'gpaw'")
