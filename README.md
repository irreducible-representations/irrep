# Irrep

This is a code to determine symmetry eigenvalues of electronic states obtained by DFT codes, as well as irreducible representations, 
wannier charge centers (1D) and many more

also one can get the trace.txt file for [Check Topological Mat](http://www.cryst.ehu.es/cgi-bin/cryst/programs/topological.pl) routine 
of [Bilbao Crystallographic Server](https://www.cryst.ehu.es).

Help on usage can be obtained by typing

```
irrep -h
```

Help is under construction and is far from being complete. If you have interest in the code and not sure how to use it, 
feel free to contact the author.

Also, have a look at the examples provided.

An example of using: 

```
irrep Ecut=50 code=abinit fWFK=Bi_WFK refUC=0,-1,1,1,0,-1,-1,-1,-1  kpoints=11 IBend=5  kpnames="GM"
```


## Installation

Install using `pip`:

```
pip install irrep
```

Currently the code is interfaced VASP, ABINIT and QuantumEspresso, but if interface with other code is needed, please contact te author.

The code relies on [spglib](https://github.com/atztogo/spglib) library to determine the symmetry of the crystal
and the tables of the characters of irreducible representations, obtained from the [Bilbao Crystallographic Server (BCS)](http://www.cryst.ehu.es/)
If you use this code to determine irreps for a scientific publication, please acknowledge BCS and
cite:

> L. Elcoro, B. Bradlyn, Z. Wang, M. G. Vergniory, J. Cano, C. Felser, B. A. Bernevig, D. Orobengoa, G. de la Flor and M. I. Aroyo
"Double crystallographic groups and their representations on the Bilbao Crystallographic Server"
J. of Appl. Cryst. (2017). 50, 1457-1477. doi:10.1107/S1600576717011712

Before releasing this public repository on github on 22th of June 2019, 
the code is mainly written by:

Stepan S. Tsirkin 
University of Zurich
stepan.tsirkin@physik.uzh.ch

after a fruitful discussion of formalism with Maia G. Vergniory (Donostia International Physics Center/IKERBASQUE, Basque Country, Spain) 

I also acknowledge contributions to the code from Mikel Iraola (Donostia International Physics Center, Basque Country, Spain) 

Further contributions from other authors may be tracked on GitHub [contributors list](https://github.com/stepan-tsirkin/irrep/graphs/contributors). 
