from ortools.sat.python import cp_model
import numpy as np

class varArraySolutionObtainer(cp_model.CpSolverSolutionCallback):
    """A class representing a solution callback for the solver.

    Parameters
    ----------
    variables : list
        List of variables of type `cp_model.IntVar` defined in the model

    Attributes
    ----------
    __variables : list
        List of variables of type `cp_model.IntVar` defined in the model
    __x : list
        Each element is list of values for the variables in a solution
    __solutionCount : int
        Number of discovered solutions
    """

    def __init__(self, variables: cp_model.IntVar) -> None:
        """Initiate a solution initial state.

        Args:
            variables (ortools.sat.python.cp_model.IntVar): values of the variables.
        """
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solutionCount = 0
        self.__x = []

    def on_solution_callback(self) -> None:
        """Save the current solution in a list and add `1` to the solution count."""
        self.__solutionCount += 1
        self.__x.append([self.Value(v) for v in self.__variables])  # type:ignore

    def solutionCount(self) -> int:
        """Returns the total number of solutions.

        Returns:
            int: Total number of solutions
        """
        return self.__solutionCount

    def solutions(self) -> np.ndarray:
        """Returns all solutions of the model in a matrix form, where each row is a particular solution.

        Returns
        -------
        numpy.ndarray
            Each row contains a solution
        """
        return np.array(self.__x)
    
    def n_smallest_solutions(self, n=5):
        '''
        Get the first `n` solutions

        Parameters
        ----------
        n : int
            Number of solutions to be retrieved

        Returns
        -------
        list : 
            Contains `n` elements, and each element is a solution
        '''
        solutions = [(x, np.linalg.norm(x)) for x in self.__x]
        solutions = sorted(solutions, key=lambda x: x[1])

        return [x[0] for x in solutions[:n]]
