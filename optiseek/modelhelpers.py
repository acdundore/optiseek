import itertools
import numpy as np

class parameter_grid_search:
    """
    Creates a grid of parameter permutations based on user input and runs a selected algorithm with those parameters to find the best set.

    Attributes
    ----------
    param_permutations : list of dicts
        A list of dictionaries that represent the parameter permutations used for each iteration of the grid search. In the
        dicts, the parameter name as a string is the key and the parameter value is the value.

    permutation_positions : list of ndarrays
        A list containing the optimal positions found for each permutation of parameters. This corresponds with the permutations
        in the param_permutations attribute.

    permutation_values : list of floats
        A list containing the optimal values found for each permutation of parameters. This corresponds with the permutations
        in the param_permutations attribute.

    best_parameters : dict
        A dictionary containing the best performing set of parameters. The parameter names as strings are stored as keys and the
        corresponding values are stored as values.

    best_position : list or ndarray
        The most optimal position that was found using the best performing parameters.

    best_value : float
        The most optimal function value that was found using the best performing parameters.

    Methods
    -------
    solve()
        Executes the parameter grid search process and stores the results in the class attributes.
    """
    def __init__(self, algorithm, input_function, param_options, show_progress=False):
        """
        Constructs the necessary attributes for the parameter grid search.

        Parameters
        ----------
        algorithm : class
            Class of the algorithm that you would like to use.

        input_function : function
            Function object for the algorithm to optimize.

        param_options : dict
            Dictionary containing the grid of parameters to be explored with the parameter names (strings) as keys
            and a list of parameter values as values. All permutations of parameters in this dict will be tested.
            For inputs that would normally be in a list (like the search bounds on a 2+ dimensional function, for example),
            place that list inside another list. For any parameters not specified, the default will be used.

            Example of possible param_options dict for a 2D function using the simulated annealing algorithm:
            param_options = {
                "initial_guess": [[8, 4]],
                "b_lower": [[-10, -10]],
                "b_upper": [[10, 10]],
                "sigma_coeff": [0.05, 0.1, 0.2, 0.3],
                "max_iter": [50, 100, 500],
                "start_temperature": [20, 10, 5, 2],
                "alpha": [0.85, 0.93, 0.99],
                "neighbor_dim_changes": [1, 2]
                }

            Note that the initial_guess, b_lower, and b_upper parameters will use the same values for every permutation,
            but must be double bracketed due to the function being 2D.

        show_progress : bool, default = False
            Boolean to indicate whether the grid search will print progress to the console as the solve continues.
            The number of permutations increases exponentially with respect to parameter inputs, so for high numbers
            of parameter inputs, this can be useful to see how much longer the solver has left.
        """
        self.algorithm = algorithm # instance of an algorithm object to be investigated
        self.param_options = param_options # dictionary of parameter options used to create parameter dictionary
        self.input_function = input_function # input function to be optimized
        self.show_progress = show_progress

        # creating all permutations of parameters possible from user's choices
        keys, values = zip(*self.param_options.items())
        self.param_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # setting lengths of output lists
        self.permutation_positions = [None] * len(self.param_permutations)
        self.permutation_values = [None] * len(self.param_permutations)

    # TODO: create internal method to check that parameters all exist

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        self._algorithm = value

    @property
    def param_options(self):
        return self._param_options

    @param_options.setter
    def param_options(self, value):
        self._param_options = value

    @property
    def input_function(self):
        return self._input_function

    @input_function.setter
    def input_function(self, value):
        self._input_function = value

    @property
    def show_progress(self):
        return self._show_progress

    @show_progress.setter
    def show_progress(self, value):
        self._show_progress = value

    @property
    def param_permutations(self):
        return self._param_permutations

    @param_permutations.setter
    def param_permutations(self, value):
        self._param_permutations = value

    @property
    def permutation_values(self):
        return self._permutation_values

    @permutation_values.setter
    def permutation_values(self, value):
        self._permutation_values = value

    @property
    def permutation_positions(self):
        return self._permutation_positions

    @permutation_positions.setter
    def permutation_positions(self, value):
        self._permutation_positions = value

    @property
    def best_parameters(self):
        return self._best_parameters

    @best_parameters.setter
    def best_parameters(self, value):
        self._best_parameters = value

    @property
    def best_position(self):
        return self._best_position

    @best_position.setter
    def best_position(self, value):
        self._best_position = value

    @property
    def best_value(self):
        return self._best_value

    @best_value.setter
    def best_value(self, value):
        self._best_value = value

    def solve(self):
        n_permutations = len(self.param_permutations)
        for i in range(n_permutations):
            # get current keyword arguments and apply them to the algorithm
            current_kwargs = self.param_permutations[i].copy()
            current_alg = self.algorithm(self.input_function, **current_kwargs)

            # setting initial best values and for first iteration
            if i == 0:
                if current_alg.find_minimum == True:
                    self.best_value = np.Inf
                else:
                    self.best_value = -np.Inf

            # solve the algorithm with the current keyword arguments
            current_alg.solve()

            # store the best position and value for this permutation of keyword arguments
            self.permutation_values[i] = current_alg.best_value
            self.permutation_positions[i] = current_alg.best_position

            # replacing best performing parameter values if necessary
            if (current_alg.find_minimum == True and current_alg.best_value < self.best_value) or (current_alg.find_minimum == False and current_alg.best_value > self.best_value):
                self.best_parameters = current_kwargs.copy()
                self.best_value = current_alg.best_value
                self.best_position = current_alg.best_position

            # printing progress if necessary
            if self.show_progress:
                current_progress = "Iteration %d of %d complete." % (i+1, n_permutations)
                print(current_progress)


def penalty_constraints(input_function, constraint_dict, find_minimum=True, p_quadratic=1, p_count=0):
    """
    Applies constraints to an objective function as penalties and returns a new penalized function to be optimized.

    Parameters
    ----------
    input_function : function
        Function object for the algorithm to optimize.

    constraint_dict : dict with constraint functions as keys and constraint types as values
        A dictionary that contains any number of constraint equations to be applied to the input function.
        The dictionary is structured like {constraint function: constraint type} where the constraints are
        compared to zero with a mathematical operator: g1(x) = 0, g2(x) < 0, etc.
        The constraint function must share the same arguments in the same order as the objective function.
        The constraint type must be one of the following strings: "<", "<=", ">", ">=", "=".

        Example:
        Constraint 1: 2x > 5y  ->  create constraint function g1(x, y) that returns 2x-5y
        Constraint 2: 3x-y <= 4  ->  create constraint function g2(x, y) that returns 3x-y-4
        Then create the constraint dictionary: {g1: ">", g2: "<="}

    find_minimum : bool, default = True
        Indicates whether the optimum of interest is a minimum or maximum. If false, looks for maximum.

    p_quadratic : float, default = 1
        Penalty multiplier for the quadratic penalty in [0, inf]. A value of zero will result in no
        quadratic penalty to the objective function. A nonzero value smoothly penalizes the function according to
        the magnitude that the constraint is broken.

    p_count : float, default = 0
        Penalty multiplier for the count penalty in [0, inf]. A value of zero will result in no
        count penalty to the objective function. A nonzero value creates a sharp discontinuity where the
        constraint is broken.

    Returns
    -------
    A function representing the input objective function with the constraints applied as penalties to the function value.
    """
    def penalized_function(*args):
        # initialize the output as the plain function output
        output = input_function(*args)

        # include a sign change depending on whether the function is to be minimized or maximized
        penalty_sign = 1
        if find_minimum == False:
            penalty_sign = -1

        # iterate through constraints and penalize the function output accordingly
        for g, g_type in constraint_dict.items():
            # get current constraint function output and constraint type
            g_output = g(*args)

            # evaluate penalty value and add it to the main functions output
            if g_type == "<":
                output += penalty_sign * (p_quadratic * max(g_output, 0) ** 2 + p_count * (g_output < 0))
            elif g_type == "<=":
                output += penalty_sign * (p_quadratic * max(g_output, 0) ** 2 + p_count * (g_output <= 0))
            elif g_type == ">":
                output += penalty_sign * (p_quadratic * min(g_output, 0) ** 2 + p_count * (g_output > 0))
            elif g_type == ">=":
                output += penalty_sign * (p_quadratic * min(g_output, 0) ** 2 + p_count * (g_output >= 0))
            else:
                output += penalty_sign * (p_quadratic * g_output ** 2)

        return output

    return penalized_function



























