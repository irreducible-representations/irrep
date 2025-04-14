"""Module to compute EBR decompositions.
"""
import numpy as np
import os
import json

# Actual EBR decomposition requires OR-Tool's SAT problem solver.
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ModuleNotFoundError:
    ORTOOLS_AVAILABLE = False


def get_ebr_matrix(ebr_data):
    """
    Gets the EBR matrix from a dictionary of EBR data.

    Parameters
    ----------
    ebr_data : dict
        Dictionary containing the EBR data as saved in the package files

    Returns
    -------
    array
        EBR matrix with dimensions Nirreps x Nebrs
    """
    ebr_matrix = np.array([x["vector"] for x in ebr_data["ebrs"]], dtype=int).T

    return ebr_matrix

def get_smith_form(ebr_data, return_all=True):
    """
    Returns the Smith normal form from EBR data

    Parameters
    ----------
    ebr_data : dict
        Dictionary with EBR data as saved in the package files.
    return_all : bool, optional
        Whether to return all the matrices or only the diagonal, by default True

    Returns
    -------
    array or tuple of arrays
        Matrices involved in the Smith normal form

    Notes
    -----
    Notation for the Smith decomposition:

    .. math::

        EBR = U^{-1} \cdot R \cdot V^{-1}
    """
    #U^{-1}RV^{-1}
    u = np.array(ebr_data["smith_form"]["u"], dtype=int)
    v = np.array(ebr_data["smith_form"]["v"], dtype=int)
    r = np.array(ebr_data["smith_form"]["r"], dtype=int)

    if return_all:
        return u,r,v
    else:
        return r

def get_ebr_names_and_positions(ebr_data):
    """
    Get the EBR labels and Wyckoff position from the EBR data

    Parameters
    ----------
    ebr_data : dict
        Dictionary with EBR data as save in the package files.

    Returns
    -------
    list
        list of tuples with EBR label and WP
    """
    return [(x["ebr_name"], x["wyckoff_position"]) for x in ebr_data["ebrs"]]


def create_symmetry_vector(irrep_counts, basis_labels):
    """Computes the symmetry vector given an ordered list of basis irrep labels
    and the irrep counts.

    Parameters
    ----------
    irrep_counts : dict
        Keys are labels of irreps, values are their multiplicities
    basis_labels : list
        basis irrep labels

    Returns
    -------
    np.ndarray
        symmetry vector. Elements are multiplicities of the irreps in the 
        table of EBRs and are sorted accordingly
    """
    basis_index = {name : i for i, name in enumerate(basis_labels)}

    vec = np.zeros(len(basis_index), dtype=int)

    for label, multi in irrep_counts.items():
        if label in basis_index:
            vec[basis_index[label]] = multi

    return vec 



def compute_topological_classification_vector(irrep_counts, ebr_data):
    """Computes relevant quantities in the problem of identifying topology from
    the Smith decomposition of the EBR matrix

    Parameters
    ----------
    irrep_counts : dict
        Keys are labels of irreps, values are their multiplicities
    ebr_data : dict
        EBR data loaded from files

    Returns
    -------
    y : np.ndarray
        Symmetry-data vector with multiplicities of irreps sorted as in EBRs' 
        tables
    y_prime : np.ndarray
        Symmetry-data vector labeled as :math:`y'`
    nontrivial : bool
        Whether the bands host a nontrivial phase

    Notes
    -----
    The Smith decomposition follows this notation:

    .. math::

        EBR \cdot x = y, \\
        EBR = U^{-1} \cdot R \cdot R^{-1},\\
        R \cdot Y = C, \\
        x' = V^{-1} \cdot x,\\
        y' = U \cdot y.
    """

    y = create_symmetry_vector(irrep_counts, ebr_data["basis"]["irrep_labels"])

    u, r, _ = get_smith_form(ebr_data)
    d = r.diagonal()
    d_pos = d[d > 0]

    y_prime = u @ y
    # check if the entries of vec_prime divide the elementary divisors
    nontrivial = ((y_prime[:len(d_pos)] % d_pos != 0)).any()

    return y, y_prime, nontrivial




def compute_ebr_decomposition(ebr_data, y):
    from .or_solutions_obtainer import varArraySolutionObtainer 
    """
    Compute the decomposition of the symmetry vector into EBRs

    Parameters
    ----------
    ebr_data : dict
        Dictionary with EBR data loaded from the package files.
    y : array
        symmetry vector

    Returns
    -------
    solutions : list
        Solutions found for the EBR decomposition. `None` if no solution was 
        found
    is_positive : bool
        Whether solutions involve only positive coefficients of EBRs. If no 
        solution was found, it is also returns `False`

    Notes
    -----
    The Smith decomposition follows this notation:

    .. math::

        EBR \cdot x = y, \\
        EBR = U^{-1} \cdot R \cdot R^{-1},\\
        R \cdot Y = C, \\
        x' = V^{-1} \cdot x,\\
        y' = U \cdot y.
    """

    def get_solutions(bounds=(0,15), n_smallest=5):
        """
        Solve the decomposition problem with some coefficient bounds and return
        some solutions, starting by the combinations with smallest coefficients

        Parameters
        ----------
        bounds : tuple, optional
            bounds for the coefficients, by default (0,15)
        n_smallest : int, optional
            how many solutions to, by default 5

        Returns
        -------
        list    
            list of solutions in form of lists of integers
        str
            status of the search of solutions
        """

        lb, ub = bounds
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()

        # Add the coefficients of EBRs as variables
        x = [model.NewIntVar(lb, ub, f"x{i}") for i in range(n_ebr)]

        # Construct a callback that will be called each time a new solution is 
        # found
        solution_obtainer = varArraySolutionObtainer(x)

        # Add constraint: multiplicities of irreps in the solution must match 
        # the multiplicities in the symmetry-data vector y
        for i in range(n_ir):
            model.Add(EBR[i] @ x == y[i])

        solver.SearchForAllSolutions(model, solution_obtainer)

        return solution_obtainer.n_smallest_solutions(n_smallest), solver.status_name()

    EBR = get_ebr_matrix(ebr_data)

    n_ir, n_ebr = EBR.shape

    # first check positive coefficients only
    is_positive = True
    solutions, status = get_solutions(bounds=(0,50))

    # if positive solutions were not found
    if status not in ["OPTIMAL", "FEASIBLE"]:
        is_positive = False
        
        # try with negative solutions
        solutions, status = get_solutions(bounds=(-50,50))

        # if negative solutions are not found, something's wrong
        if status not in ["OPTIMAL", "FEASIBLE"]:
            return None, is_positive
        # else return negative + positive solutions
        else:
            return solutions, is_positive
    else:
        return solutions, is_positive


def compose_irrep_string(irrep_counts):
    """
    Creates a string with the direct sum of irreps from a list of (repeated)
    irrep labels

    Parameters
    ----------
    irrep_counts : dict
        Keys are irrep labels and values are multiplicities

    Returns
    -------
    str
        String with direct sum of irreps with multiplicities.
    """

    terms = [f"{multi} x {name}" for name, multi in irrep_counts.items() if multi != 0]
    s = " + ".join(terms)

    return s

def compose_ebr_string(vec, ebrs):
    """
    Create a string with the direct sum of EBRS from a decomposition vector

    Parameters
    ----------
    vec : array
        Vector with coefficients for each irrep
    ebrs : list
        List of tuples with EBR label and Wyckoff position

    Returns
    -------
    str
        string representing the EBR decomposition in readable form
    """

    terms = [
        f"{multi} x [ {label} @ {wp} ]" for (label, wp), multi in zip(ebrs, vec)
        if multi != 0
    ]
    s = " + ".join(terms)

    return s

def load_ebr_data(sg_number, spinor):
    '''
    Load data from file of EBRs

    Parameters
    ----------
    sg_number : int
        Number of the space group
    spinor : bool
        Whether wave functions are spinors (SOC) or not

    Returns
    -------
    dict
        EBR data
    '''

    root = os.path.dirname(__file__)
    filename = f"{sg_number}_ebrs.json"
    ebr_data = json.load(open(root + "/data/ebrs/" + filename, 'r'))
    ebr_data = ebr_data["double" if spinor else "single"]
    return ebr_data
