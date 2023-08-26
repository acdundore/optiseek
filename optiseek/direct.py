import types
import numpy as np

class _direct_method:
    def __init__(self, input_function, initial_guess, find_minimum=True, max_iter=100, sol_threshold=None, store_results=False):
        # initializing variables
        self.input_function = input_function  # function to optimize
        self.initial_guess = initial_guess # initial guess position
        self.find_minimum = find_minimum # boolean indicating if we are seeking the minimum or maximum of the function
        self.max_iter = max_iter # maximum iterations that the algorithm will complete before stopping
        self.sol_threshold = sol_threshold # absolute threshold on the solution; after reaching this point, the algorithm will terminate
        self.store_results = store_results # boolean to store position and value data from all iterations for post-processing; this is off by default
        self.stored_positions = None # initialize stored results as None
        self.stored_values = None # initialize stored results as None
        self.best_position = None
        self.best_value = None
        self.completed_iter = 0

    @property
    def input_function(self):
        return self._input_function

    @input_function.setter
    def input_function(self, value):
        if type(value) is not types.FunctionType and type(value) is not types.BuiltinFunctionType:
            raise TypeError("User must input a <class 'function'> type.")
        else:
            self._input_function = value

    @property
    def initial_guess(self):
        return self._initial_guess

    @initial_guess.setter
    def initial_guess(self, value):
        if isinstance(value, np.ndarray):
            self._initial_guess = value
        elif type(value) is list:
            self._initial_guess = np.array(value, dtype=float)
        else:
            self._initial_guess = np.array([value], dtype=float)
        self.n_dimensions = self._initial_guess.size

    @property
    def find_minimum(self):
        return self._find_minimum

    @find_minimum.setter
    def find_minimum(self, value):
        if type(value) is not bool:
            raise TypeError("find_minimum must be of type boolean.")
        self._find_minimum = value

    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value):
        if type(value) is not int:
            raise TypeError("max_iter must be an integer.")
        self._max_iter = value

    @property
    def sol_threshold(self):
        return self._sol_threshold

    @sol_threshold.setter
    def sol_threshold(self, value):
        self._sol_threshold = value

    @property
    def store_results(self):
        return self._store_results

    @store_results.setter
    def store_results(self, value):
        self._store_results = value

    @property
    def stored_positions(self):
        return self._stored_positions

    @stored_positions.setter
    def stored_positions(self, value):
        self._stored_positions = value

    @property
    def stored_values(self):
        return self._stored_values

    @stored_values.setter
    def stored_values(self, value):
        self._stored_values = value

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

    @property
    def completed_iter(self):
        return self._completed_iter

    @completed_iter.setter
    def completed_iter(self, value):
        self._completed_iter = value

    @property
    def n_dimensions(self):
        return self._n_dimensions

    @n_dimensions.setter
    def n_dimensions(self, value):
        self._n_dimensions = value

    # method to compare two function values depending on whether it is a minimization or maximization problem
    def _first_is_better(self, first_value, second_value):
        output_bool = False
        if self.find_minimum:
            if first_value < second_value:
                output_bool = True
        else:
            if first_value > second_value:
                output_bool = True

        return output_bool

    # method to run at the beginning of the main solve() method in order to run some checks
    def _initialize_solve(self):
        # make sure the solution threshold is set correctly based on a minimizing or maximizing problem
        if self.sol_threshold is None:
            if self.find_minimum:
                self.sol_threshold = -np.Inf
            else:
                self.sol_threshold = np.Inf

        # setting a dummy initial best function value
        if self.find_minimum:
            self.best_value = np.Inf
        else:
            self.best_value = -np.Inf

    # method that checks stopping criteria and returns false if any are met
    def _check_stopping_criteria(self, iteration_count, best_value):
        keep_iterating = True

        if iteration_count >= self.max_iter:
            keep_iterating = False

        if self.sol_threshold is not None and ((self.find_minimum == True and best_value < self.sol_threshold) or (self.find_minimum == False and best_value > self.sol_threshold)):
            keep_iterating = False

        return keep_iterating

    def _initialize_stored_results(self, population_size):
        # initializing stored result arrays if the user chooses
        if self.store_results:
            # axis convention of arrays: [iterations, population members, dimensions]
            self.stored_positions = np.zeros(shape=(self.max_iter, population_size, self.n_dimensions))
            self.stored_values = np.zeros(shape=(self.max_iter, population_size, 1))


class cyclic_coordinate_descent(_direct_method):
    """
    A cyclic coordinate descent method.

    Attributes
    ----------
    best_position : ndarray
        Most optimal position found during the solution iterations.

    best_value : float
        Most optimal function value found during the solution iterations.

    completed_iter : int
        Number of iterations completed during the solution process.

    stored_positions : ndarray
        Positions for each particle for each iteration after the solver is finished. Set to None if user does not choose to store results.

    stored_values : ndarray
        Function values for each member of the population for each iteration. Set to None if user does not choose to store results.

    Methods
    -------
    solve()
        Executes the algorithm solution with the current parameters.
    """
    def __init__(self, input_function, initial_guess, find_minimum=True, max_iter=100, sol_threshold=None, store_results=False, max_step_size=1.0):
        """
        Constructs the necessary attributes for the algorithm.

        Parameters
        ----------
        input_function : function
            Function object for the algorithm to optimize.

        initial_guess : list of floats or ndarray
            An initial guess of the function minimum by the user. This has a great influence on the solution of the algorithm.

        find_minimum : bool, default = True
            Indicates whether the optimum of interest is a minimum or maximum. If false, looks for maximum.

        max_iter : int, default = 100
            Maximum number of iterations. If reached, the algorithm terminates.

        sol_threshold : float, default = None
            If a solution is found past this threshold, the iterations stop. None indicates that the algorithm will not consider this.

        store_results : bool, default = False
            Choose whether to save results or not. If true, results will be saved to the stored_positions and stored_values properties.

        max_step_size : float, default = 1.0
            Maximum step size that the algorithm can possibly take for each iteration in each direction.
        """
        super().__init__(input_function=input_function, initial_guess=initial_guess, find_minimum=find_minimum, max_iter=max_iter, sol_threshold=sol_threshold, store_results=store_results)
        self.max_step_size = max_step_size # maximum line search distance

    @property
    def max_step_size(self):
        return self._max_step_size

    @max_step_size.setter
    def max_step_size(self, value):
        if value <= 0:
            print("Warning: max_step_size must be > 0. max_step_size has been set to 1.")
            self.max_step_size = 1
        else:
            self._max_step_size = value

    # method for golden section search for internal use. finds an optimum along a single dimension while all others constant
    def _golden_section_search(self, input_function, current_position, search_index, x_low, x_high):
        # set a relative error threshold and maximum number of search iterations
        rel_error_threshold = 0.000001
        max_iter = 30
        phi = 1.618034

        # set the 4 test points according to the ratio phi:1:phi across the search space in the current dimension
        x = np.array([x_low, x_high - (x_high - x_low) / phi, x_low + (x_high - x_low) / phi, x_high])
        y = np.zeros(shape=(4,))

        # get the function values with the test points, keeping all other variables in the current position constant
        temp_input = current_position.copy()
        for i in range(4):
            temp_input[search_index] = x[i]
            y[i] = input_function(*temp_input)

        # initiate a relative error, previous value, and iteration counter; then, begin iterations
        rel_error = 1
        previous_value = 0
        golden_search_iter = 0
        while rel_error > rel_error_threshold and golden_search_iter < max_iter:
            # store a temporary check of the best value; this is important for checking if the extremes of the search space (x[0] or x[3]) hold the best value
            if self.find_minimum:
                best_value_temp = min(y)
            else:
                best_value_temp = max(y)

            # check whether the minimum is on the left or right sides of the current search space, then update the sample points
            if self._first_is_better(y[1], y[2]) or best_value_temp == y[0]:
                x[3] = x[2]
                y[3] = y[2]
                x[2] = x[1]
                y[2] = y[1]
                x[1] = x[3] - (x[3] - x[0]) / phi
                temp_input[search_index] = x[1]
                y[1] = input_function(*temp_input)
            else:
                x[0] = x[1]
                y[0] = y[1]
                x[1] = x[2]
                y[1] = y[2]
                x[2] = (x[3] - x[0]) / phi + x[0]
                temp_input[search_index] = x[2]
                y[2] = input_function(*temp_input)

            # find the best value out of the new sample points
            if self.find_minimum:
                current_value = min(y)
            else:
                current_value = max(y)

            # increment the iterations and calculate relative error
            golden_search_iter += 1
            if golden_search_iter > 1:
                if current_value != previous_value:
                    rel_error = abs((current_value - previous_value) / previous_value)
            previous_value = current_value

        # after stopping criteria is met, find the optimum value in the search space and return the full position with that new value
        optimum_x = x[np.where(y == current_value)][0]
        output_position = current_position.copy()
        output_position[search_index] = optimum_x

        return output_position

    def solve(self):
        # initializing solution process
        self._initialize_solve()
        self._initialize_stored_results(1)

        # initialize the current position as the initial guess
        current_position = self.initial_guess
        current_value = self.input_function(*current_position)

        # initialize iteration count and begin iterations
        iteration_count = 0
        while self._check_stopping_criteria(iteration_count, current_value):
            # step through the dimensions cyclically, one at a time
            current_dimension = np.remainder(iteration_count, self.n_dimensions)

            # calculate the lower and upper bounds of the current dimension using the specified maximum step size
            position_in_dimension = current_position[current_dimension]
            lower_bound = position_in_dimension - self.max_step_size
            upper_bound = position_in_dimension + self.max_step_size

            # execute a line search along the current dimension
            current_position = self._golden_section_search(self.input_function, current_position, current_dimension, lower_bound, upper_bound)
            current_value = self.input_function(*current_position)

            # store intermediate results for post-processing if specified
            if self.store_results:
                self.stored_positions[iteration_count, 0, :] = current_position.copy()
                self.stored_values[iteration_count, 0, :] = current_value

            # increment the iterations
            iteration_count += 1

        # storing final results
        self.best_position = current_position
        self.best_value = self.input_function(*self.best_position)
        self.completed_iter = iteration_count

        # truncate stored results
        if self.store_results:
            self.stored_positions = self.stored_positions[0: self.completed_iter, :, :]
            self.stored_values = self.stored_values[0: self.completed_iter, :, :]


class basic_pattern_search(_direct_method):
    """
    A basic Hooke-Jeeves pattern search algorithm.

    Attributes
    ----------
    best_position : ndarray
        Most optimal position found during the solution iterations.

    best_value : float
        Most optimal function value found during the solution iterations.

    completed_iter : int
        Number of iterations completed during the solution process.

    stored_positions : ndarray
        Positions for each particle for each iteration after the solver is finished. Set to None if user does not choose to store results.

    stored_values : ndarray
        Function values for each member of the population for each iteration. Set to None if user does not choose to store results.

    Methods
    -------
    solve()
        Executes the algorithm solution with the current parameters.
    """
    def __init__(self, input_function, initial_guess, find_minimum=True, max_iter=100, sol_threshold=None, store_results=False, max_step_size=1.0):
        """
        Constructs the necessary attributes for the algorithm.

        Parameters
        ----------
        input_function : function
            Function object for the algorithm to optimize.

        initial_guess : list of floats or ndarray
            An initial guess of the function minimum by the user. This has a great influence on the solution of the algorithm.

        find_minimum : bool, default = True
            Indicates whether the optimum of interest is a minimum or maximum. If false, looks for maximum.

        max_iter : int, default = 100
            Maximum number of iterations. If reached, the algorithm terminates.

        sol_threshold : float, default = None
            If a solution is found past this threshold, the iterations stop. None indicates that the algorithm will not consider this.

        store_results : bool, default = False
            Choose whether to save results or not. If true, results will be saved to the stored_positions and stored_values properties.

        max_step_size : float, default = 1.0
            Maximum step size that the algorithm can possibly take for each iteration in each direction.
        """
        super().__init__(input_function=input_function, initial_guess=initial_guess, find_minimum=find_minimum, max_iter=max_iter, sol_threshold=sol_threshold, store_results=store_results)
        self.max_step_size = max_step_size # maximum line search distance

    @property
    def max_step_size(self):
        return self._max_step_size

    @max_step_size.setter
    def max_step_size(self, value):
        if value <= 0:
            print("Warning: max_step_size must be > 0. max_step_size has been set to 1.")
            self.max_step_size = 1
        else:
            self._max_step_size = value

    def solve(self):
        # initializing solution process
        self._initialize_solve()
        self._initialize_stored_results(1)

        # initialize the current position as the initial guess, set up the current step size, and set up the sample position list
        current_position = self.initial_guess
        current_value = self.input_function(*current_position)
        current_step_size = self.max_step_size
        sample_positions = [None] * (self.n_dimensions * 2)
        sample_values = np.zeros(shape=(2 * self.n_dimensions,))

        # initialize iteration count and begin iterations
        iteration_count = 0
        while self._check_stopping_criteria(iteration_count, current_value):
            # get sample points by checking +/- the step size of each dimension relative to the current position
            for d in range(self.n_dimensions):
                position_variation = np.zeros(shape=(self.n_dimensions,), dtype=float)
                position_variation[d] = current_step_size
                sample_positions[2 * d] = current_position - position_variation
                sample_positions[2 * d + 1] = current_position + position_variation

            # calculate the function values of each of the sample points
            for i in range(2 * self.n_dimensions):
                sample_values[i] = self.input_function(*sample_positions[i])

            # find the best sample value and the index of the best of the sample points
            if self.find_minimum:
                best_sample_value = min(sample_values)
            else:
                best_sample_value = max(sample_values)
            index_of_best = np.where(sample_values == best_sample_value)[0][0]

            # if the best sample value is better than the current point, then move to the best sample value; otherwise, halve the current step size and iterate
            if self._first_is_better(best_sample_value, current_value):
                current_position = sample_positions[index_of_best].copy()
                current_value = best_sample_value
            else:
                current_step_size = 0.5 * current_step_size

            # store intermediate results for post-processing if specified
            if self.store_results:
                self.stored_positions[iteration_count, 0, :] = current_position.copy()
                self.stored_values[iteration_count, 0, :] = current_value

            # increment the iterations
            iteration_count += 1

        # storing final results
        self.best_position = current_position
        self.best_value = self.input_function(*self.best_position)
        self.completed_iter = iteration_count

        # truncate stored results
        if self.store_results:
            self.stored_positions = self.stored_positions[0: self.completed_iter, :, :]
            self.stored_values = self.stored_values[0: self.completed_iter, :, :]


class enhanced_pattern_search(_direct_method):
    """
    An enhanced Hooke-Jeeves pattern search aglorithm.

    Attributes
    ----------
    best_position : ndarray
        Most optimal position found during the solution iterations.

    best_value : float
        Most optimal function value found during the solution iterations.

    completed_iter : int
        Number of iterations completed during the solution process.

    stored_positions : ndarray
        Positions for each particle for each iteration after the solver is finished. Set to None if user does not choose to store results.

    stored_values : ndarray
        Function values for each member of the population for each iteration. Set to None if user does not choose to store results.

    Methods
    -------
    solve()
        Executes the algorithm solution with the current parameters.
    """
    def __init__(self, input_function, initial_guess, find_minimum=True, max_iter=100, sol_threshold=None, store_results=False, max_step_size=1.0):
        """
        Constructs the necessary attributes for the algorithm.

        Parameters
        ----------
        input_function : function
            Function object for the algorithm to optimize.

        initial_guess : list of floats or ndarray
            An initial guess of the function minimum by the user. This has a great influence on the solution of the algorithm.

        find_minimum : bool, default = True
            Indicates whether the optimum of interest is a minimum or maximum. If false, looks for maximum.

        max_iter : int, default = 100
            Maximum number of iterations. If reached, the algorithm terminates.

        sol_threshold : float, default = None
            If a solution is found past this threshold, the iterations stop. None indicates that the algorithm will not consider this.

        store_results : bool, default = False
            Choose whether to save results or not. If true, results will be saved to the stored_positions and stored_values properties.

        max_step_size : float, default = 1.0
            Maximum step size that the algorithm can possibly take for each iteration in each direction.
        """
        super().__init__(input_function=input_function, initial_guess=initial_guess, find_minimum=find_minimum, max_iter=max_iter, sol_threshold=sol_threshold, store_results=store_results)
        self.max_step_size = max_step_size # maximum line search distance

    @property
    def max_step_size(self):
        return self._max_step_size

    @max_step_size.setter
    def max_step_size(self, value):
        if value <= 0:
            print("Warning: max_step_size must be > 0. max_step_size has been set to 1.")
            self.max_step_size = 1
        else:
            self._max_step_size = value

    def solve(self):
        # initializing solution process
        self._initialize_solve()
        self._initialize_stored_results(1)

        # initialize the current position as the initial guess, set up the current step size, and set up the sample position list
        current_position = self.initial_guess
        current_value = self.input_function(*current_position)
        current_step_size = self.max_step_size
        sample_positions = [None] * (self.n_dimensions + 1)
        sample_values = np.zeros(shape=(self.n_dimensions + 1,))

        # initialize iteration count and begin iterations
        iteration_count = 0
        while self._check_stopping_criteria(iteration_count, current_value):
            # get sample points by adding the positive step size in each dimension to the current position
            for d in range(self.n_dimensions):
                position_variation = np.zeros(shape=(self.n_dimensions,), dtype=float)
                position_variation[d] = current_step_size
                sample_positions[d] = current_position + position_variation

            # get final sample point by taking the negative step size in all directions
            position_variation = np.ones(shape=(self.n_dimensions,), dtype=float) * -current_step_size
            sample_positions[self.n_dimensions] = current_position + position_variation

            # calculate the function values of each of the sample points
            for i in range(self.n_dimensions + 1):
                sample_values[i] = self.input_function(*sample_positions[i])

            # find the best sample value and the index of the best of the sample points
            if self.find_minimum:
                best_sample_value = min(sample_values)
            else:
                best_sample_value = max(sample_values)
            index_of_best = np.where(sample_values == best_sample_value)[0][0]

            # if the best sample value is better than the current point, then move to the best sample value; otherwise, halve the current step size and iterate
            if self._first_is_better(best_sample_value, current_value):
                current_position = sample_positions[index_of_best].copy()
                current_value = best_sample_value
            else:
                current_step_size = 0.5 * current_step_size

            # store intermediate results for post-processing if specified
            if self.store_results:
                self.stored_positions[iteration_count, 0, :] = current_position.copy()
                self.stored_values[iteration_count, 0, :] = current_value

            # increment the iterations
            iteration_count += 1

        # storing final results
        self.best_position = current_position
        self.best_value = self.input_function(*self.best_position)
        self.completed_iter = iteration_count

        # truncate stored results
        if self.store_results:
            self.stored_positions = self.stored_positions[0: self.completed_iter, :, :]
            self.stored_values = self.stored_values[0: self.completed_iter, :, :]












