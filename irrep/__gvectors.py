
            # ###   ###   #####  ###
            # #  #  #  #  #      #  #
            # ###   ###   ###    ###
            # #  #  #  #  #      #
            # #   # #   # #####  #


##################################################################
## This file is distributed as part of                           #
## "IrRep" code and under terms of GNU General Public license v3 #
## see LICENSE file in the                                       #
##                                                               #
##  Written by Stepan Tsirkin, University of Zurich.             #
##  e-mail: stepan.tsirkin@physik.uzh.ch                         #
##################################################################


import numpy as np
from numpy.linalg import det, norm


class NotSymmetryError(RuntimeError):
    pass


#!!   constant  below is 2m/hbar**2 in units of 1/eV Ang^2 (value is
#!!   adjusted in final decimal places to agree with VASP value; program
#!!   checks for discrepancy of any results between this and VASP values)
twomhbar2 = 0.262465831

correction = 0
twomhbar2 *= 1 + correction


# This function is a python translation of a part of WaveTrans Code
def calc_gvectors(
    K,
    RecLattice,
    Ecut,
    nplane=np.Inf,
    Ecut1=-1,
    thresh=1e-3,
    spinor=True,
    nplanemax=10000,
):
    """ calculating g-vectors for a point K with energy cutoff Ecut
        nplane is used to check the correctness
        optionally one may provide a lower cutoff Ecut1, to exclude higher g-vectors
        returns an integer array of column-vectors igall[6,ncnt1],
        thresh - threshould for defining the g-vectors with the same energy
        where
            first three components are the reciprocal lattice coordinates
            the fourth - is the index of the g-vectors (with the reduced Ecut1)
            the 5th and the 6th - the start and end of the group of g-vectors with the same energy
        in the original list (with Ecut from WAVECAR)
"""
    if Ecut1 <= 0:
        Ecut1 = Ecut
    B = RecLattice

    igp = np.zeros(3)
    igall = []
    Eg = []
    for N in range(nplanemax):
        if N % 10 == 0:
            print(N, len(igall))
        if len(igall) >= nplane / (2 if spinor else 1):
            break
        for ig3 in range(-N, N + 1):
            for ig2 in range(-(N - abs(ig3)), N - abs(ig3) + 1):
                for ig1 in set([-(N - abs(ig3) - abs(ig2)), N - abs(ig3) - abs(ig2)]):
                    igp = (ig1, ig2, ig3)
                    etot = norm((K + np.array(igp)).dot(B)) ** 2 / twomhbar2
                    if etot < Ecut:
                        igall.append(igp)
                        Eg.append(etot)

    ncnt = len(igall)
    #    print ("\n".join("{0:+4d}  {1:4d} {2:4d}  |  {3:6d}".format(ig[0],ig[1],ig[2],np.abs(ig).sum()) for ig in igall) )
    #    print (len(igall),len(set(igall)))
    if nplane < np.Inf:
        if spinor:
            if 2 * ncnt != nplane:
                raise RuntimeError(
                    "*** error - computed 2*ncnt={0} != input nplane={1}".format(
                        2 * ncnt, nplane
                    )
                )
        else:
            if ncnt != nplane:
                raise RuntimeError(
                    "*** error - computed ncnt={0} != input nplane={1}".format(
                        ncnt, nplane
                    )
                )
    igall = np.array(igall, dtype=int)
    ng = igall.max(axis=0) - igall.min(axis=0)
    igall1 = igall % ng[None, :]
    #    print ("ng=",ng)
    #    print ("igall1=",igall1)
    igallsrt = np.argsort((igall1[:, 2] * ng[1] + igall1[:, 1]) * ng[0] + igall1[:, 0])
    #    print (igallsrt)
    igall1 = igall[igallsrt]
    Eg = np.array(Eg)[igallsrt]
    #    print (igall1)
    igall = np.zeros((ncnt, 6), dtype=int)
    igall[:, :3] = igall1
    #    print (igall)
    igall[:, 3] = np.arange(ncnt)
    igall = igall[Eg <= Ecut1]
    Eg = Eg[Eg <= Ecut1]
    srt = np.argsort(Eg)
    Eg = Eg[srt]
    igall = igall[srt, :].T
    wall = [0] + list(np.where(Eg[1:] - Eg[:-1] > thresh)[0] + 1) + [igall.shape[1]]
    for i in range(len(wall) - 1):
        igall[4, wall[i] : wall[i + 1]] = wall[i]
        igall[5, wall[i] : wall[i + 1]] = wall[i + 1]
    #    print ("K={0}\n E={1}\nigall=\n{2}".format(K,Eg,igall.T))
    return igall


def transformed_g(kpt, ig, RecLattice, A):
    """This function calculates how the transformation matrix A reorders the reciprocal
lattice vectors ig[6,ng], returns an array rotind, defined as
rotind[i]=j if A*ig[:,i]==ig[:,j]  (where igTr is the array of transformed g-vectors)
RecLattice -- rows are the vectors of the reciprocal lattice
"""
    #    Btrr=RecLattice.dot(A).dot(np.linalg.inv(RecLattice))
    #    Btr=np.array(np.round(Btrr),dtype=int) # The transformed rec. lattice expressed in the basis of the original rec. lattice
    #    if np.sum(np.abs(Btr-Btrr))>1e-6:
    #        raise NotSymmetryError("The lattice is not invariant under transformation \n {0}".format(A))
    B = np.linalg.inv(A).T
    kpt_ = B.dot(kpt)
    dkpt = np.array(np.round(kpt_ - kpt), dtype=int)
    #    print ("Transformation\n",A)
    #    print ("kpt ={0} -> {1}".format(kpt,kpt_))
    if not np.isclose(dkpt, kpt_ - kpt).all():
        raise NotSymmetryError(
            "The k-point {0} is transformed to non-equivalent point {1}  under transformation\n {2}".format(
                kpt, kpt_, A
            )
        )

    igTr = B.dot(ig[:3, :]) + dkpt[:, None]  # the transformed
    igTr = np.array(np.round(igTr), dtype=int)
    #    print ("the original g-vectors :\n",ig)
    #    print ("the transformed g-vectors :\n",igTr)
    ng = ig.shape[1]
    rotind = -np.ones(ng, dtype=int)
    for i in range(ng):
        for j in range(ig[4, i], ig[5, i]):
            if (igTr[:, i] == ig[:3, j]).all():
                rotind[i] = j
                break
        if rotind[i] == -1:
            raise RuntimeError(
                "no pair found for the transformed g-vector igTr[{i}]={igtr}  ig[{i}]={ig} in the original g-vectors set (kpoint{kp}). Other g-vectors with same energy:\n{other} ".format(
                    i=i,
                    ig=ig[:3, i],
                    igtr=igTr[:, i],
                    kp=kpt,
                    other=ig[:3, ig[4, i] : ig[5, i]],
                )
            )
    return rotind


def symm_eigenvalues(
    K, RecLattice, WF, igall, A=np.eye(3), S=np.eye(2), T=np.zeros(3), spinor=True
):
    npw1 = igall.shape[1]
    multZ = np.exp(
        -1.0j * (2 * np.pi * np.linalg.inv(A).dot(T).dot(igall[:3, :] + K[:, None]))
    )
    igrot = transformed_g(K, igall, RecLattice, A)
    if spinor:
        part1 = WF[:, igrot].conj() * WF[:, :npw1] * S[0, 0]
        part2 = (
            WF[:, igrot + npw1].conj() * WF[:, npw1:] * S[1, 1]
            + WF[:, igrot].conj() * WF[:, npw1:] * S[0, 1]
            + WF[:, igrot + npw1].conj() * WF[:, :npw1] * S[1, 0]
        )
        return np.dot(part1 + part2, multZ)
    else:
        return np.dot(WF[:, igrot].conj() * WF[:, :], multZ)


def symm_matrix(
    K, RecLattice, WF, igall, A=np.eye(3), S=np.eye(2), T=np.zeros(3), spinor=True
):
    # computes the matrix S_mn = <Psi_m|T|Psi_n>
    npw1 = igall.shape[1]
    multZ = np.exp(-1.0j * (2 * np.pi * A.dot(T).dot(igall[:3, :] + K[:, None])))
    igrot = transformed_g(K, igall, RecLattice, A)
    if spinor:
        WF1 = np.stack([WF[:, igrot], WF[:, igrot + npw1]], axis=2).conj()
        WF2 = np.stack([WF[:, :npw1], WF[:, npw1:]], axis=2)
        #        print (WF1.shape,WF2.shape,multZ.shape,S.shape)
        return np.einsum("mgs,ngt,g,st->mn", WF1, WF2, multZ, S)
    else:
        return np.einsum("mg,ng,g->mn", WF[:, igrot].conj(), WF, multZ)
