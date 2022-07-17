import types
import numpy as np


class _individual:
    def __init__(self, n_dimensions=1):
        # initialize variables
        self._best_value = None
        self._function_value = None

        # set up class
        self.n_dimensions = n_dimensions

    @property
    def n_dimensions(self):
        return self._n_dimensions

    @n_dimensions.setter
    def n_dimensions(self, value):
        self._n_dimensions = value
        self.position = np.zeros(shape=(value,))
        self.velocity = np.zeros(shape=(value,))
        self.best_position = np.zeros(shape=(value,))

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def function_value(self):
        return self._function_value

    @function_value.setter
    def function_value(self, value):
        self._function_value = value

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        self._velocity = value

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

    def random_position(self, b_lower, b_upper):
        self.position = np.random.uniform(b_lower, b_upper, size=(b_lower.size,))
        self.best_position = self.position

class _metaheuristic:
    def __init__(self, input_function, b_lower=-10, b_upper=10, find_minimum=True, max_iter=100, sol_threshold=None, max_unchanged_iter=None, store_results=False):
        # initializing variables
        self.input_function = input_function  # function to optimize
        self.b_lower = b_lower # lower search bound values for each of the dimensions in an array
        self.b_upper = b_upper # upper search bound values for each of the dimensions in an array
        self.find_minimum = find_minimum # boolean indicating if we are seeking the minimum or maximum of the function
        self.max_iter = max_iter # maximum iterations that the algorithm will complete before stopping
        self.sol_threshold = sol_threshold # absolute threshold on the solution; after reaching this point, the algorithm will terminate
        self.max_unchanged_iter = max_unchanged_iter # number of iterations that best location does not change. if reached, algorithm terminates
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
    def b_lower(self):
        return self._b_lower

    @b_lower.setter
    def b_lower(self, value):
        if isinstance(value, np.ndarray):
            self._b_lower = value
        elif type(value) is list:
            self._b_lower = np.array(value, dtype=float)
        else:
            self._b_lower = np.array([value], dtype=float)
        self.n_dimensions = self._b_lower.size

    @property
    def b_upper(self):
        return self._b_upper

    @b_upper.setter
    def b_upper(self, value):
        if isinstance(value, np.ndarray):
            self._b_upper = value
        elif type(value) is list:
            self._b_upper = np.array(value, dtype=float)
        else:
            self._b_upper = np.array([value], dtype=float)
        self.n_dimensions = self._b_upper.size

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
    def max_unchanged_iter(self):
        return self._max_unchanged_iter

    @max_unchanged_iter.setter
    def max_unchanged_iter(self, value):
        self._max_unchanged_iter = value

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

        # checking to ensure that the variable lower and upper bound lists are equal in size, and setting dimension of function
        if len(self.b_lower) != len(self.b_upper):
            raise ValueError("Lower and upper bound lists are not the same length.")
        else:
            self.n_dimensions = len(self.b_lower)

        # if user doesn't want to use the unchanged iteration cap quitting criterion, then set to max iteration length
        if self.max_unchanged_iter is None:
            self.max_unchanged_iter = self.max_iter

        # getting bound widths of the dimensions
        self._bound_widths = self.b_upper - self.b_lower

        # setting a dummy initial best function value
        if self.find_minimum:
            self.best_value = np.Inf
        else:
            self.best_value = -np.Inf

    # method that checks stopping criteria and returns false if any are met
    def _check_stopping_criteria(self, iteration_count, unchanged_iterations, best_value):
        keep_iterating = True

        if iteration_count >= self.max_iter:
            keep_iterating = False

        if self.max_unchanged_iter is not None and unchanged_iterations >= self.max_unchanged_iter:
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


class _local_search(_metaheuristic):
    def __init__(self, input_function, b_lower=-10, b_upper=10, find_minimum=True, max_iter=100, sol_threshold=None, max_unchanged_iter=None, store_results=False, sigma_coeff=0.2, neighbor_dim_changes=1, initial_guess=None):
        super().__init__(input_function=input_function, b_lower=b_lower, b_upper=b_upper, find_minimum=find_minimum, max_iter=max_iter, sol_threshold=sol_threshold, max_unchanged_iter=max_unchanged_iter, store_results=store_results)
        self.sigma_coeff = sigma_coeff # ratio of bound width to be used for standard deviation in perturbation for that dimension for the candidate solution dimensions
        self.neighbor_dim_changes = neighbor_dim_changes # specifies how many dimensions to mutate for each neighbor candidate generated. if None, will mutatate all dimensions
        self.initial_guess = initial_guess # optional initial guess input from user. if blank, initial guess will be random

    @property
    def sigma_coeff(self):
        return self._sigma_coeff

    @sigma_coeff.setter
    def sigma_coeff(self, value):
        if value <= 0:
            print("Warning: sigma must be in (0, 0.5]. A value of 0.001 has been set.")
            self._sigma_coeff = 0.001
        elif value > 0.5:
            print("Warning: sigma must be in (0, 0.5]. A value of 0.5 has been set.")
            self._sigma_coeff = 0.5
        else:
            self._sigma_coeff = value

    @property
    def neighbor_dim_changes(self):
        return self._neighbor_dim_changes

    @neighbor_dim_changes.setter
    def neighbor_dim_changes(self, value):
        if type(value) is not int:
            raise ValueError("neighbor_dim_changes must be an integer.")
        elif value < 1:
            print("The neighbor_dim_changes parameter must be at least 1. A value of 1 has been set.")
            self._neighbor_dim_changes = 1
        elif value > self.n_dimensions:
            print("neighbor_dim_changes must be less than or equal to n_dimensions. It has been overwritten to n_dimensions.")
            self._neighbor_dim_changes = self.n_dimensions
        else:
            self._neighbor_dim_changes = value

    @property
    def initial_guess(self):
        return self._initial_guess

    @initial_guess.setter
    def initial_guess(self, value):
        if value is None or isinstance(value, np.ndarray):
            self._initial_guess = value
        else:
            self._initial_guess = np.array(value)

    # function to generate a new guess that neighbors the current guess using a sigma provided by the user
    def _generate_neighbor(self, current_guess):
        # if not specified, mutate all dimensions for new neighbor
        if self.neighbor_dim_changes is None:
            self.neighbor_dim_changes = self.n_dimensions

        # pick specified number of dimensions to mutate for the new neighbor position
        random_dimension_indices = np.random.choice(np.arange(0, self.n_dimensions), size=self.neighbor_dim_changes, replace=False)
        delta = np.zeros(shape=(self.n_dimensions,)) # initialize array of zeros for changes
        delta[random_dimension_indices] = 1 # change selected indices to 1
        delta = delta * np.random.normal(0, self.sigma_coeff * self._bound_widths) # multiply by a random variation; only selected indices will not be zero

        # formulating new neighbor
        new_guess = current_guess + delta

        return new_guess


class _particle(_individual):
    def __init__(self, n_dimensions):
        super().__init__(n_dimensions)

    # give the particle some initial velocity
    def random_velocity(self, b_lower, b_upper):
        self.velocity = np.random.uniform(-abs(b_upper - b_lower), abs(b_upper - b_lower))

class particle_swarm_optimizer(_metaheuristic):
    """
    A particle swarm optimization algorithm.

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
    def __init__(self, input_function, b_lower=-10, b_upper=10, find_minimum=True, max_iter=100, sol_threshold=None, max_unchanged_iter=None, store_results=False, n_particles=50, weight=0.25, phi_p=1, phi_g=2, zero_velocity=True):
        """
        Constructs the necessary attributes for the particle swarm optimizer.

        Parameters
        ----------
        input_function : function
            Function object for the algorithm to optimize.

        b_lower : float or list of floats or ndarray, default = -10
            List or array containing the lower bounds of each dimension of the search space.

        b_upper : float or list of floats or ndarray, default = 10
            List or array containing the upper bounds of each dimension of the search space.

        find_minimum : bool, default = True
            Indicates whether the optimum of interest is a minimum or maximum. If false, looks for maximum.

        max_iter : int, default = 100
            Maximum number of iterations. If reached, the algorithm terminates.

        sol_threshold : float, default = None
            If a solution is found past this threshold, the iterations stop. None indicates that the algorithm will not consider this.

        max_unchanged_iter : int, default = None
            If the solution does not improve after this many iterations, the iterations stop.

        store_results : bool, default = False
            Choose whether to save results or not. If true, results will be saved to the stored_positions and stored_values properties.

        n_particles : int, default = 50
            Number of particles to use in the particle swarm population.

        weight : float, default = 0.25
            Weight coefficient in [0, 1]. Lower weight gives the particles less momentum.

        phi_p : float, default = 1.0
            Cognitive coefficient in [0, 3]. Higher value indicates that the particles are drawn more towards their own best known position.

        phi_g : float, default = 2.0
            Social coefficient in [0, 3]. Higher value indicates that the particles are drawn more towards the swarm's collectively best known position.

        zero_velocity : bool, default = False
            Choose whether the particles start off with zero velocity or a random initial velocity. Initial velocities can sometimes be unstable.
        """
        super().__init__(input_function=input_function, b_lower=b_lower, b_upper=b_upper, find_minimum=find_minimum, max_iter=max_iter, sol_threshold=sol_threshold, max_unchanged_iter=max_unchanged_iter, store_results=store_results)
        # properties specific to PSO
        self.n_particles = n_particles # number of particles in the swarm
        self.weight = weight # "weight" of the particles, which acts as an inertia and resists change in velocity
        self.phi_p = phi_p # knowledge coefficient, which influences how much the particle's velocity is affected by it's own best known position
        self.phi_g = phi_g # social coefficient, which influences how much the particle's velocity is affected by swarms best known location
        self.zero_velocity = zero_velocity # boolean that indicates whether the particles should begin with zero velocity or not

    @property
    def n_particles(self):
        return self._n_particles

    @n_particles.setter
    def n_particles(self, value):
        if type(value) is not int:
            raise TypeError("n_particles must be an integer.")
        self._n_particles = value

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        if value < 0:
            print("Warning: weight must be between 0 and 1. A value of 0 has been set.")
            self._weight = 0
        elif value > 1:
            print("Warning: weight must be between 0 and 1. A value of 1 has been set.")
            self._weight = 1
        else:
            self._weight = value

    @property
    def phi_p(self):
        return self._phi_p

    @phi_p.setter
    def phi_p(self, value):
        if value < 0:
            print("Warning: phi_p must be between 0 and 3. A value of 0 has been set.")
            self._phi_p = 0
        elif value > 3:
            print("Warning: phi_p must be between 0 and 3. A value of 3 has been set.")
            self._phi_p = 3
        else:
            self._phi_p = value

    @property
    def phi_g(self):
        return self._phi_g

    @phi_g.setter
    def phi_g(self, value):
        if value < 0:
            print("Warning: phi_g must be between 0 and 3. A value of 0 has been set.")
            self._phi_g = 0
        elif value > 3:
            print("Warning: phi_g must be between 0 and 3. A value of 3 has been set.")
            self._phi_g = 3
        else:
            self._phi_g = value

    @property
    def zero_velocity(self):
        return self._zero_velocity

    @zero_velocity.setter
    def zero_velocity(self, value):
        if type(value) is not bool:
            raise TypeError("zero_velocity must be a boolean.")
        self._zero_velocity = value

    def solve(self):
        """
        Executes the solution iterations for the algorithm.

        Returns
        -------
        None
        """
        self._initialize_solve()
        self._initialize_stored_results(self.n_particles)

        # Create the swarm
        swarm = [_particle(n_dimensions=self.n_dimensions) for p in range(self.n_particles)]

        # give initial positions to each particle and store swarm's best known position and value
        for p in swarm:
            # Initialize a random position for each particle
            p.random_position(self.b_lower, self.b_upper)

            # Calculate the value of the function that is produced by the particle's position
            p.best_value = self.input_function(*p.position)

            # Update the swarm's best known position if necessary
            if self._first_is_better(p.best_value, self.best_value):
                self.best_position = p.position.copy()
                self.best_value = p.best_value

        # Initialize the random velocity of the particles if necessary
        if self.zero_velocity is False:
            for p in swarm:
                p.random_velocity(self.b_lower, self.b_upper)

        # Carry out the particle swarm optimization iterations
        iteration_count = 0
        unchanged_iterations = 0
        while self._check_stopping_criteria(iteration_count, unchanged_iterations, self.best_value):
            for p in swarm:
                # update velocity
                r_p = np.random.uniform(0, 1, size=(self.n_dimensions,))
                r_g = np.random.uniform(0, 1, size=(self.n_dimensions,))
                p.velocity = self.weight * p.velocity + self.phi_p * r_p * (p.best_position - p.position) + self.phi_g * r_g * (self.best_position - p.position)

                # update position
                p.position += p.velocity

                # calculate new function value
                p.function_value = self.input_function(*p.position)

                # update individually best known positions and swarm best known positions if necessary
                if self._first_is_better(p.function_value, p.best_value):
                    p.best_position = p.position.copy()
                    p.best_value = p.function_value
                    if self._first_is_better(p.best_value, self.best_value):
                        self.best_position = p.best_position.copy()
                        self.best_value = p.best_value
                        unchanged_iterations = 0

            # store intermediate results for post-processing if specified
            if self.store_results:
                for i in range(len(swarm)):
                    p = swarm[i]
                    self.stored_positions[iteration_count, i, :] = p.position.copy()
                    self.stored_values[iteration_count, i, :] = p.function_value

            # increment iteration counters
            iteration_count += 1
            unchanged_iterations += 1

        # store final iteration count
        self.completed_iter = iteration_count

        # truncate stored results
        if self.store_results:
            self.stored_positions = self.stored_positions[0: self.completed_iter, :, :]
            self.stored_values = self.stored_values[0: self.completed_iter, :, :]


class _firefly(_individual):
    def __init__(self, n_dimensions=1):
        super().__init__(n_dimensions)

class firefly_algorithm(_metaheuristic):
    """
    A Firefly Algorithm optimizer.

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
    def __init__(self, input_function, b_lower=-10, b_upper=10, find_minimum=True, max_iter=100, sol_threshold=None, max_unchanged_iter=None, store_results=False, n_fireflies=50, beta=1.0, alpha=0.01, gamma=1.0):
        """
        Constructs the necessary attributes for the algorithm.

        Parameters
        ----------
        input_function : function
            Function object for the algorithm to optimize.

        b_lower : float or list of floats or ndarray, default = -10
            List or array containing the lower bounds of each dimension of the search space.

        b_upper : float or list of floats or ndarray, default = 10
            List or array containing the upper bounds of each dimension of the search space.

        find_minimum : bool, default = True
            Indicates whether the optimum of interest is a minimum or maximum. If false, looks for maximum.

        max_iter : int, default = 100
            Maximum number of iterations. If reached, the algorithm terminates.

        sol_threshold : float, default = None
            If a solution is found past this threshold, the iterations stop. None indicates that the algorithm will not consider this.

        max_unchanged_iter : int, default = None
            If the solution does not improve after this many iterations, the iterations stop.

        store_results : bool, default = False
            Choose whether to save results or not. If true, results will be saved to the stored_positions and stored_values properties.

        n_fireflies : int, default = 50
            Number of fireflies to use in the population.

        beta : float, default = 1.0
            Beta coefficient in [0.1, 1.5]. Lower value indicates that the fireflies are less attracted to each other.

        alpha : float, default = 0.01
            Coefficient in [0, 0.1] that is a portion of each dimension's bound widths to use for the random walk.

        gamma : float, default = 1.0
            Social coefficient in [0.01, 1]. Higher value indicates that the fireflies are less attracted to each other.
        """
        super().__init__(input_function=input_function, b_lower=b_lower, b_upper=b_upper, find_minimum=find_minimum, max_iter=max_iter, sol_threshold=sol_threshold, max_unchanged_iter=max_unchanged_iter, store_results=store_results)
        self.n_fireflies = n_fireflies # number of fireflies in the population
        self.beta = beta # this is a coefficient on the visibility of the fireflies; linearly related to visibility
        self.alpha = alpha # coefficient for the random walk contribution
        self.gamma = gamma # coefficient for the dropoff distance representing light traveling through some medium; lower values mean more visibility

    @property
    def n_fireflies(self):
        return self._n_fireflies

    @n_fireflies.setter
    def n_fireflies(self, value):
        if type(value) is not int or value < 0:
            raise ValueError("n_fireflies must be a positive integer.")
        else:
            self._n_fireflies = value

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        if value < 0.1:
            print("Warning: beta must be in [0.1, 1.5]. beta set to 0.1.")
            self._beta = 0.1
        elif value > 1.5:
            print("Warning: beta must be in [0.1, 1.5]. beta set to 1.5.")
            self._beta = 1.5
        else:
            self._beta = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value < 0:
            print("Warning: alpha must be in [0, 0.1]. alpha set to 0")
            self._alpha = 0
        elif value > 0.1:
            print("Warning: alpha must be in [0, 0.1]. alpha set to 0.1.")
            self._alpha = 0.1
        else:
            self._alpha = value

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if value < 0.01:
            print("Warning: gamma must be in [0.01, 1]. gamma set to 0.01.")
            self._gamma = 0.01
        elif value > 1:
            print("Warning: gamma must be in [0.01, 1]. gamma set to 1.")
            self._gamma = 1
        else:
            self._gamma = value

    def solve(self):
        """
        Executes the solution iterations for the algorithm.

        Returns
        -------
        None
        """
        self._initialize_solve()
        self._initialize_stored_results(self.n_fireflies)

        # create the firefly swarm
        population = [_firefly(n_dimensions=self.n_dimensions) for f in range(self.n_fireflies)]

        # initialize firefly positions
        [f.random_position(self.b_lower, self.b_upper) for f in population]

        # begin iterations
        iteration_count = 0
        unchanged_iterations = 0
        while self._check_stopping_criteria(iteration_count, unchanged_iterations, self.best_value):
            # calculate function values of each firefly
            for f in population:
                f.function_value = self.input_function(*f.position)

                # update the best known function value and position if necessary
                if self._first_is_better(f.function_value, self.best_value):
                    self.best_value = f.function_value
                    self.best_position = f.position.copy()
                    unchanged_iterations = 0

            # Compare brightness and adjust positions when necessary
            for a in population:
                for b in population:
                    if self._first_is_better(b.function_value, a.function_value):
                        r = np.linalg.norm(b.position - a.position)
                        a.position += self.beta * np.exp(-self.gamma * r**2) * (b.position - a.position) + self.alpha * self._bound_widths * np.random.normal(0, 1, size=(self.n_dimensions,))

            # store intermediate results for post-processing of specified
            if self.store_results:
                for i in range(self.n_fireflies):
                    f = population[i]
                    self.stored_positions[iteration_count, i, :] = f.position.copy()
                    self.stored_values[iteration_count, i, :] = f.function_value

            # increment iteration counter
            iteration_count += 1
            unchanged_iterations += 1

        # store final iteration count
        self.completed_iter = iteration_count

        # truncate stored results
        if self.store_results:
            self.stored_positions = self.stored_positions[0: self.completed_iter, :, :]
            self.stored_values = self.stored_values[0: self.completed_iter, :, :]


class _agent(_individual):
    def __init__(self, n_dimensions=1):
        super().__init__(n_dimensions)

    @property
    def n_dimensions(self):
        return self._n_dimensions

    @n_dimensions.setter
    def n_dimensions(self, value):
        self._n_dimensions = value
        self.position = np.zeros(shape=(value,))
        self.velocity = np.zeros(shape=(value,))
        self.best_position = np.zeros(shape=(value,))
        self.potential_position = np.zeros(shape=(value,))
        self.potential_function_value = 0

    @property
    def potential_position(self):
        return self._potential_position

    @potential_position.setter
    def potential_position(self, value):
        self._potential_position = value

    @property
    def potential_function_value(self):
        return self._potential_function_value

    @potential_function_value.setter
    def potential_function_value(self, value):
        self._potential_function_value = value

class differential_evolution(_metaheuristic):
    """
    A Differential Evolution optimizer.

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
    def __init__(self, input_function, b_lower=-10, b_upper=10, find_minimum=True, max_iter=100, sol_threshold=None, max_unchanged_iter=None, store_results=False, n_agents=50, weight=0.2, p_crossover=0.5):
        """
        Constructs the necessary attributes for the algorithm.

        Parameters
        ----------
        input_function : function
            Function object for the algorithm to optimize.

        b_lower : float or list of floats or ndarray, default = -10
            List or array containing the lower bounds of each dimension of the search space.

        b_upper : float or list of floats or ndarray, default = 10
            List or array containing the upper bounds of each dimension of the search space.

        find_minimum : bool, default = True
            Indicates whether the optimum of interest is a minimum or maximum. If false, looks for maximum.

        max_iter : int, default = 100
            Maximum number of iterations. If reached, the algorithm terminates.

        sol_threshold : float, default = None
            If a solution is found past this threshold, the iterations stop. None indicates that the algorithm will not consider this.

        max_unchanged_iter : int, default = None
            If the solution does not improve after this many iterations, the iterations stop.

        store_results : bool, default = False
            Choose whether to save results or not. If true, results will be saved to the stored_positions and stored_values properties.

        n_agents : int, default = 50
            Number of fireflies to use in the population.

        weight : float, default = 0.2
            Differential weight coefficient in [0, 2].

        p_crossover : float, default = 0.5
            Probability in [0, 1] that a gene crossover will occur for each dimension.
        """
        super().__init__(input_function=input_function, b_lower=b_lower, b_upper=b_upper, find_minimum=find_minimum, max_iter=max_iter, sol_threshold=sol_threshold, max_unchanged_iter=max_unchanged_iter, store_results=store_results)
        self.n_agents = n_agents # number of agents in the algorithm; must be at least 4
        self.weight = weight # differential weight; must be in [0, 2]
        self.p_crossover = p_crossover # probability of crossover; must be between 0 and 1

    @property
    def n_agents(self):
        return self._n_agents

    @n_agents.setter
    def n_agents(self, value):
        if type(value) is not int:
            raise TypeError("n_agents must be an integer.")
        elif value < 4:
            print("Warning: n_agents must be at least 4. A value of 4 was used.")
            self._n_agents = 4
        else:
            self._n_agents = value

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        if value > 2:
            print("Warning: weight value must be in [0, 2]. weight has been set to 2.")
            self._weight = 2
        elif value < 0:
            print("Warning: weight value must be in [0, 2]. weight has been set to 0.")
            self._weight = 0
        else:
            self._weight = value

    @property
    def p_crossover(self):
        return self._p_crossover

    @p_crossover.setter
    def p_crossover(self, value):
        if value > 1:
            print("Warning: crossover probability must be in [0, 1]. p_crossover has been set to 1.")
            self._p_crossover = 1
        elif value < 0:
            print("Warning: crossover probability must be in [0, 1]. p_crossover has been set to 0.")
            self._p_crossover = 0
        else:
            self._p_crossover = value


    def solve(self):
        """
        Executes the solution iterations for the algorithm.

        Returns
        -------
        None
        """
        self._initialize_solve()
        self._initialize_stored_results(self.n_agents)

        # initialize the agents
        population = [_agent(n_dimensions=self.n_dimensions) for i in range(self.n_agents)]

        # give random positions to all agents in population
        [a.random_position(self.b_lower, self.b_upper) for a in population]

        # calculate initial function values of agents' positions and update best position and value if necessary
        for a in population:
            a.function_value = self.input_function(*a.position)
            if self._first_is_better(a.function_value, self.best_value):
                self.best_position = a.position.copy()
                self.best_value = a.function_value

        # begin optimization
        iteration_count = 0
        unchanged_iterations = 0
        while self._check_stopping_criteria(iteration_count, unchanged_iterations, self.best_value):
            for a in population:
                # pick 3 random agents from the population
                agent_1, agent_2, agent_3 = np.random.choice(population, size=3, replace=False)

                # pick a random dimension to mutate with absolute certainty
                random_dimension = np.random.randint(0, self.n_dimensions)

                # mutate the agent's potential position
                for i in range(self.n_dimensions):
                    r = np.random.uniform()
                    if r < self.p_crossover or i == random_dimension:
                        a.potential_position[i] = agent_1.position[i] + self.weight * (agent_2.position[i] - agent_3.position[i])
                    else:
                        a.potential_position[i] = a.position[i]

                # evaluate agent's potential position and replace current position if it is better
                a.potential_function_value = self.input_function(*a.potential_position)
                if self._first_is_better(a.potential_function_value, a.function_value):
                    a.position = a.potential_position.copy()
                    a.function_value = a.potential_function_value

                    # store this as best candidate solution if necessary, and reset unchanged iteration counter
                    if self._first_is_better(a.function_value, self.best_value):
                        self.best_position = a.position.copy()
                        self.best_value = a.function_value
                        unchanged_iterations = 0

            # store intermediate results for post-processing of specified
            if self.store_results:
                for i in range(self.n_agents):
                    a = population[i]
                    self.stored_positions[iteration_count, i, :] = a.position.copy()
                    self.stored_values[iteration_count, i, :] = a.function_value

            # increment iteration counter
            iteration_count += 1
            unchanged_iterations += 1

        # store final interation count
        self.completed_iter = iteration_count

        # truncate stored results
        if self.store_results:
            self.stored_positions = self.stored_positions[0: self.completed_iter, :, :]
            self.stored_values = self.stored_values[0: self.completed_iter, :, :]


class _male_mayfly(_individual):
    def __init__(self, n_dimensions=1):
        super().__init__(n_dimensions)

    def update_velocity(self, swarm_male_best_position, is_top_male, beta, alpha_cog, alpha_soc, nuptial_coeff, bound_widths, g):
        # calculating distances from personal and swarm best known optima
        r_p = np.linalg.norm(self.position - self.best_position)
        r_g = np.linalg.norm(self.position - swarm_male_best_position)

        # calculating the new velocity
        if is_top_male:
            self.velocity = g * self.velocity + nuptial_coeff * bound_widths * np.random.uniform(-1.0, 1.0, size=(self.n_dimensions,))
        else:
            self.velocity = g * self.velocity + alpha_cog * np.exp(-beta * r_p**2) * (self.best_position - self.position) - alpha_soc * np.exp(-beta * r_g**2) * (swarm_male_best_position - self.position)

class _female_mayfly(_individual):
    def __init__(self, n_dimensions=1):
        super().__init__(n_dimensions)

    def update_velocity(self, male_of_interest_position, is_attracted, alpha_attract, beta, fl, bound_widths, g):
        # calculating distance to the male of interest's position
        r_mf = np.linalg.norm(self.position - male_of_interest_position)

        # calculating new velocity
        if is_attracted:
            self.velocity = g * self.velocity + alpha_attract * np.exp(-beta * r_mf**2) * (male_of_interest_position - self.position)
        else:
            self.velocity = g * self.velocity + fl * bound_widths * np.random.uniform(-1.0, 1.0, size=(self.n_dimensions,))

class mayfly_algorithm(_metaheuristic):
    """
    A Mayfly Algorithm optimizer.

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
    def __init__(self, input_function, b_lower=-10, b_upper=10, find_minimum=True, max_iter=100, sol_threshold=None, max_unchanged_iter=None, store_results=False, n_mayflies=50, beta=0.7, gravity=0.6, alpha_cog=0.5, alpha_soc=1.5, alpha_attract=1.5, nuptial_coeff=0.05):
        """
        Constructs the necessary attributes for the algorithm.

        Parameters
        ----------
        input_function : function
            Function object for the algorithm to optimize.

        b_lower : float or list of floats or ndarray, default = -10
            List or array containing the lower bounds of each dimension of the search space.

        b_upper : float or list of floats or ndarray, default = 10
            List or array containing the upper bounds of each dimension of the search space.

        find_minimum : bool, default = True
            Indicates whether the optimum of interest is a minimum or maximum. If false, looks for maximum.

        max_iter : int, default = 100
            Maximum number of iterations. If reached, the algorithm terminates.

        sol_threshold : float, default = None
            If a solution is found past this threshold, the iterations stop. None indicates that the algorithm will not consider this.

        max_unchanged_iter : int, default = None
            If the solution does not improve after this many iterations, the iterations stop.

        store_results : bool, default = False
            Choose whether to save results or not. If true, results will be saved to the stored_positions and stored_values properties.

        n_mayflies : int, default = 50
            Number of mayflies to use in the population, split evenly between males and females.

        beta : float, default = 0.7
            Visibility coefficient in [0.1, 1]. Higher value means that mayflies are less drawn towards others.

        gravity : float, default = 0.6
            Gravity coefficient in [0.1, 1]. Lower value means that the mayflies have less momentum.

        alpha_cog : float, default = 0.5
            Cognitive coefficient in [0, 2]. Indicates how attracted the male mayflies are to their individually best known position.

        alpha_soc : float, default = 1.5
            Social coefficient in [0, 2]. Indicates how attracted the male mayflies are to the male swarm's best known position.

        alpha_attract : float, default = 1.5
            Attraction coefficient in [0, 2]. Indicates how attracted the females are to their matched male counterpart.

        nuptial_coeff : float, default = 0.05
            Nuptial coefficient in [0, 0.4], a multiplier on bound widths for each dimension used for the male and female random walks.
        """
        super().__init__(input_function=input_function, b_lower=b_lower, b_upper=b_upper, find_minimum=find_minimum, max_iter=max_iter, sol_threshold=sol_threshold, max_unchanged_iter=max_unchanged_iter, store_results=store_results)
        self.n_mayflies = n_mayflies # number of mayflies in the algorithm; must be at least 4, and it will be evenly split between males and females
        self.beta = beta # visibility coefficient in [0.1, 1]
        self.gravity = gravity # gravity coefficient that controls the intertia from previous velocities in [0, 1]
        self.alpha_cog = alpha_cog # cognitive coefficient used for male mayfly velocity updates in [0, 2]
        self.alpha_soc = alpha_soc # social coefficient used for male mayfly velocity updates in [0, 2]
        self.alpha_attract = alpha_attract # attraction coefficient for females to their matched male in [0, 2]
        self.nuptial_coeff = nuptial_coeff # nuptial dance coefficient for stochastic component to best male mayflies

    @property
    def n_mayflies(self):
        return self._n_mayflies

    @n_mayflies.setter
    def n_mayflies(self, value):
        # minimum of 4 mayflies (2 each sex) and must be an even number
        if type(value) is not int:
            raise ValueError("n_mayflies must be an integer.")
        if value < 4:
            print("Warning: n_mayflies must be at least 4. A value of 4 has been set.")
            value = 4
        if np.remainder(value, 2) != 0:
            value += 1
        self._n_mayflies = value

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        if value < 0.1:
            print("Warning: beta must be between 0.1 and 1.0. A value of 0.1 has been set.")
            self._beta = 0.1
        elif value > 1.0:
            print("Warning: beta must be between 0.1 and 1.0. A value of 1.0 has been set.")
            self._beta = 1.0
        else:
            self._beta = value

    @property
    def gravity(self):
        return self._gravity

    @gravity.setter
    def gravity(self, value):
        if value < 0.1:
            print("Warning: gravity must be between 0 and 1.0. A value of 0 has been set.")
            self._gravity = 0
        elif value > 1.0:
            print("Warning: gravity must be between 0 and 1.0. A value of 1.0 has been set.")
            self._gravity = 1.0
        else:
            self._gravity = value

    @property
    def alpha_cog(self):
        return self._alpha_cog

    @alpha_cog.setter
    def alpha_cog(self, value):
        if value < 0:
            print("Warning: alpha_cog must be between 0 and 2.0. A value of 0 has been set.")
            self._alpha_cog = 0
        elif value > 2:
            print("Warning: alpha_cog must be between 0 and 2.0. A value of 2.0 has been set.")
            self._alpha_cog = 2.0
        else:
            self._alpha_cog = value

    @property
    def alpha_soc(self):
        return self._alpha_soc

    @alpha_soc.setter
    def alpha_soc(self, value):
        if value < 0:
            print("Warning: alpha_soc must be between 0 and 2.0. A value of 0 has been set.")
            self._alpha_soc = 0
        elif value > 2:
            print("Warning: alpha_soc must be between 0 and 2.0. A value of 2.0 has been set.")
            self._alpha_soc = 2.0
        else:
            self._alpha_soc = value

    @property
    def alpha_attract(self):
        return self._alpha_attract

    @alpha_attract.setter
    def alpha_attract(self, value):
        if value < 0:
            print("Warning: alpha_attract must be between 0 and 2.0. A value of 0 has been set.")
            self._alpha_attract = 0
        elif value > 2:
            print("Warning: alpha_attract must be between 0 and 2.0. A value of 2.0 has been set.")
            self._alpha_attract = 2.0
        else:
            self._alpha_attract = value

    @property
    def nuptial_coeff(self):
        return self._nuptial_coeff

    @nuptial_coeff.setter
    def nuptial_coeff(self, value):
        if value < 0:
            print("Warning: nuptial_coeff must be between 0 and 0.4. A value of 0 has been set.")
            self._nuptial_coeff = 0
        elif value > 0.4:
            print("Warning: nuptial_coeff must be between 0 and 0.4. A value of 0.4 has been set.")
            self._nuptial_coeff = 0.4
        else:
            self._nuptial_coeff = value
        self.fl = self._nuptial_coeff # match fl coefficient to nuptial dance coefficient

    @property
    def fl(self):
        return self._fl

    @fl.setter
    def fl(self, value):
        self._fl = value

    def _rank_mayflies(self, unranked_population):
        # creating list of function values corresponding to each member of population
        function_values = [n.function_value for n in unranked_population]

        # creating a dictionary with the population and corresponding function values
        population_value_dict = dict(zip(unranked_population, function_values))

        # sorting population by key (function value), then extracting the ranked population list
        ranked_population = list(dict(sorted(population_value_dict.items(), key=lambda x: x[1], reverse=not self.find_minimum)).keys())

        return ranked_population

    def _mate_mayflies(self, male, female):
        # generate random crossover coefficient and initialize offspring
        L = np.random.uniform(0.3, 0.7, size=(self.n_dimensions,))
        male_offspring = _male_mayfly(self.n_dimensions)
        female_offspring = _female_mayfly(self.n_dimensions)

        # perform crossover to create new offspring positions
        male_offspring.position = L * male.position + (1 - L) * female.position
        female_offspring.position = L * female.position + (1 - L) * male.position

        return [male_offspring, female_offspring]

    def solve(self):
        """
        Executes the solution iterations for the algorithm.

        Returns
        -------
        None
        """
        self._initialize_solve()
        self._initialize_stored_results(self.n_mayflies)

        # initialize the mayflies and give them initial positions
        male_population = [_male_mayfly(n_dimensions=self.n_dimensions) for i in range(int(self.n_mayflies / 2))]
        female_population = [_female_mayfly(n_dimensions=self.n_dimensions) for i in range(int(self.n_mayflies / 2))]

        # generate initial positions for mayflies
        [m.random_position(self.b_lower, self.b_upper) for m in male_population]
        [f.random_position(self.b_lower, self.b_upper) for f in female_population]

        # calculate initial function values of mayflies' positions and store the best position and value. also store the best male position separately
        for m in male_population:
            m.function_value = self.input_function(*m.position)
            if self._first_is_better(m.function_value, self.best_value):
                self.best_position = m.position.copy()
                self.best_value = m.function_value
        for f in female_population:
            f.function_value = self.input_function(*f.position)
            if self._first_is_better(f.function_value, self.best_value):
                self.best_position = f.position.copy()
                self.best_value = f.function_value

        # rank the male and female mayflies by function values, and record top male position
        male_population = self._rank_mayflies(male_population)
        female_population = self._rank_mayflies(female_population)
        best_male_position = male_population[0].position.copy()

        # start solution iterations
        iteration_count = 0
        unchanged_iterations = 0
        while self._check_stopping_criteria(iteration_count, unchanged_iterations, self.best_value):
            # update velocities, positions, and function values of male mayflies
            current_male_rank = 0
            for m in male_population:
                # check if it is a "top male", hardcoded as the top 5% of male function values
                if current_male_rank <= np.ceil(len(male_population) * 0.05):
                    is_top_male = True
                else:
                    is_top_male = False

                # update the velocity of the current male
                m.update_velocity(best_male_position, is_top_male, self.beta, self.alpha_cog, self.alpha_soc, self.nuptial_coeff, self._bound_widths, self.gravity)

                # update the position and evaluate new function value
                m.position += m.velocity
                m.function_value = self.input_function(*m.position)
                current_male_rank += 1

            # update velocities, positions, and function values of female mayflies
            current_female_rank = 0
            for f in female_population:
                # get the male mayfly they are matched with
                male_of_interest = male_population[current_female_rank]

                # determine if they are attracted to the male by comparing function values
                if self._first_is_better(male_of_interest.function_value, f.function_value):
                    is_attracted = True
                else:
                    is_attracted = False

                # update the velocity
                f.update_velocity(male_of_interest.position, is_attracted, self.alpha_attract, self.beta, self.fl, self._bound_widths, self.gravity)

                # update the position and evaluate new function value
                f.position += f.velocity
                f.function_value = self.input_function(*f.position)
                current_female_rank += 1

            # rank the male and female mayflies by function values
            male_population = self._rank_mayflies(male_population)
            female_population = self._rank_mayflies(female_population)

            # mate the mayflies according to their matched ranks, then evaluate the function values of their offspring
            new_males = [None] * len(male_population)
            new_females = [None] * len(female_population)
            for i in range(len(male_population)):
                new_males[i], new_females[i] = self._mate_mayflies(male_population[i], female_population[i])
                new_males[i].function_value = self.input_function(*new_males[i].position)
                new_females[i].function_value = self.input_function(*new_females[i].position)

            # combine new offspring into existing populations, rank them, and kill off the worst ones to create the new standard populations
            total_male_population = self._rank_mayflies(male_population + new_males)
            total_female_population = self._rank_mayflies(female_population + new_females)
            male_population = total_male_population[0:int(self.n_mayflies / 2)]
            female_population = total_female_population[0:int(self.n_mayflies / 2)]

            # update best recorded solutions
            best_male_position = male_population[0].position.copy()
            top_male = male_population[0]
            top_female = female_population[0]
            if self._first_is_better(top_male.function_value, top_female.function_value):
                top_male_or_female = top_male
            else:
                top_male_or_female = top_female

            if self._first_is_better(top_male_or_female.function_value, self.best_value):
                self.best_position = top_male_or_female.position.copy()
                self.best_value = top_male_or_female.function_value
                unchanged_iterations = 0

            # store intermediate results for post-processing if specified
            if self.store_results:
                total_population = male_population + female_population
                for i in range(len(total_population)):
                    m = total_population[i]
                    self.stored_positions[iteration_count, i, :] = m.position.copy()
                    self.stored_values[iteration_count, i, :] = m.function_value

            # increment iterations
            iteration_count += 1
            unchanged_iterations += 1

        # save final iteration count
        self.completed_iter = iteration_count

        # truncate intermediate results
        if self.store_results:
            self.stored_positions = self.stored_positions[0: self.completed_iter, :, :]
            self.stored_values = self.stored_values[0: self.completed_iter, :, :]


class _flying_fox(_individual):
    def __init__(self, n_dimensions=1):
        super().__init__(n_dimensions)

    @property
    def previous_position(self):
        return self._previous_position

    @previous_position.setter
    def previous_position(self, value):
        self._previous_position = value

    @property
    def previous_function_value(self):
        return self._previous_function_value

    @previous_function_value.setter
    def previous_function_value(self, value):
        self._previous_function_value = value

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = value

    @property
    def pa(self):
        return self._pa

    @pa.setter
    def pa(self, value):
        self._pa = value

    def update_position(self, input_function, find_minimum, coolest_position, coolest_value, population, delta_1, delta_3):
        # set boolean to indicate that the fox is far from the coolest spot and needs to be replaced
        is_far = False

        # calculating new position based on proximity to coolest known spot
        if abs(coolest_value - self.function_value) > delta_1 / 2:
            # array of uniform random numbers in [0, 1)
            r = np.random.rand(self.n_dimensions, )

            # generate new position
            new_position = self.position + self.a * r * (coolest_position - self.position)
        else:
            # use an array of mutation probabilities to determine parameters to be mutated if the fox is close to suffocation
            mutation_probabilities = np.random.rand(self.n_dimensions,)
            mutation_bool = mutation_probabilities > self.pa

            # set a single dimension to mutate with 100% chance
            mutation_bool[np.random.randint(0, high=self.n_dimensions)] = True

            # two arrays of uniform random numbers in [0, 1)
            r1 = np.random.rand(self.n_dimensions,)
            r2 = np.random.rand(self.n_dimensions,)

            # positions of two randomly selected foxes from the population
            xR1 = population[np.random.randint(0, len(population))].position
            xR2 = population[np.random.randint(0, len(population))].position

            # generate new position, using mutation boolean to indicate dimensions to be changed
            new_position = self.position + mutation_bool * (r1 * (coolest_position - self.position) + r2 * (xR1 - xR2))

        # update position of the fox if the new_position is better
        new_function_value = input_function(*new_position)
        if (find_minimum == True and new_function_value < self.function_value) or (find_minimum == False and new_function_value > self.function_value):
            self.previous_function_value = self.function_value
            self.position = new_position.copy()
            self.function_value = new_function_value
        else:
            if abs(coolest_value - self.function_value) > delta_3:
                is_far = True

        return is_far

    def tune_parameters(self, find_minimum, delta_1, delta_2, delta_3, delta_max, coolest_value, a_crisps, pa_crisps):
        # crisps take the form [low, medium, high]

        # calulate delta and phi for parameter tuning
        delta = abs(coolest_value - self.function_value)
        if (find_minimum == True and self.function_value < self.previous_function_value) or (find_minimum == False and self.function_value > self.previous_function_value):
            phi_sign = 1
        else:
            phi_sign = -1
        phi = phi_sign * (self.function_value - self.previous_function_value) / delta_max

        # setting the Same, Near, and Far membership function values for delta
        if delta < delta_1:
            delta_same = 1
            delta_near = 0
            delta_far = 0
        elif delta_1 <= delta and delta < delta_2:
            delta_same = (delta_2 - delta) / (delta_2 - delta_1)
            delta_near = (delta - delta_1) / (delta_2 - delta_1)
            delta_far = 0
        elif delta_2 <= delta and delta < delta_3:
            delta_same = 0
            delta_near = (delta_3 - delta) / (delta_3 - delta_2)
            delta_far = (delta - delta_2) / (delta_3 - delta_2)
        else:
            delta_same = 0
            delta_near = 0
            delta_far = 1

        # setting the Better, Same, and Worse membership values for phi
        phi_same = min(max(1 - abs(phi), -1), 1)
        if phi < 0:
            phi_better = max(-phi, -1)
            phi_worse = 0
        else:
            phi_better = 0
            phi_worse = min(phi, 1)

        # tuning parameters with the fuzzy rule system
        membership_sum = delta_near + delta_same + delta_far + phi_worse + phi_same + phi_better
        self.a = (phi_better * a_crisps[0] + phi_same * a_crisps[1] + delta_same * a_crisps[1] + delta_near * a_crisps[1] + phi_worse * a_crisps[2] + delta_far * a_crisps[2]) / membership_sum
        self.pa = (phi_worse * pa_crisps[0] + delta_far * pa_crisps[0] + phi_same * pa_crisps[1] + delta_same * pa_crisps[1] + phi_better * pa_crisps[2] + delta_near * pa_crisps[2]) / membership_sum

class flying_fox_algorithm(_metaheuristic):
    """
    A Flying Fox Algorithm optimizer.

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
    def __init__(self, input_function, b_lower=-10, b_upper=10, find_minimum=True, max_iter=100, sol_threshold=None, max_unchanged_iter=None, store_results=False):
        """
        Constructs the necessary attributes for the algorithm.

        Parameters
        ----------
        input_function : function
            Function object for the algorithm to optimize.

        b_lower : float or list of floats or ndarray, default = -10
            List or array containing the lower bounds of each dimension of the search space.

        b_upper : float or list of floats or ndarray, default = 10
            List or array containing the upper bounds of each dimension of the search space.

        find_minimum : bool, default = True
            Indicates whether the optimum of interest is a minimum or maximum. If false, looks for maximum.

        max_iter : int, default = 100
            Maximum number of iterations. If reached, the algorithm terminates.

        sol_threshold : float, default = None
            If a solution is found past this threshold, the iterations stop. None indicates that the algorithm will not consider this.

        max_unchanged_iter : int, default = None
            If the solution does not improve after this many iterations, the iterations stop.

        store_results : bool, default = False
            Choose whether to save results or not. If true, results will be saved to the stored_positions and stored_values properties.
        """
        super().__init__(input_function=input_function, b_lower=b_lower, b_upper=b_upper, find_minimum=find_minimum, max_iter=max_iter, sol_threshold=sol_threshold, max_unchanged_iter=max_unchanged_iter, store_results=store_results)

    @property
    def n_foxes(self):
        return self._n_foxes

    @n_foxes.setter
    def n_foxes(self, value):
        self._n_foxes = value

    def _update_cool_hot_positions(self, population, coolest_position, coolest_value, hottest_position, hottest_value):
        for ff in population:
            if self.find_minimum:
                if ff.function_value < coolest_value:
                    coolest_value = ff.function_value
                    coolest_position = ff.position.copy()
                if ff.function_value > hottest_value:
                    hottest_value = ff.function_value
                    hottest_position = ff.position.copy()
            else:
                if ff.function_value > coolest_value:
                    coolest_value = ff.function_value
                    coolest_position = ff.position.copy()
                if ff.function_value < hottest_value:
                    hottest_value = ff.function_value
                    hottest_position = ff.position.copy()

        return [coolest_position, coolest_value, hottest_position, hottest_value]

    def _update_SL(self, population, NL):
        SL_dict = {}
        for ff in population:
            SL_dict[ff] = ff.function_value # adding to survival list, to be edited later

        # sort the function values and truncate to create the survival list
        SL = dict(sorted(SL_dict.items(), key=lambda x: x[1], reverse=not self.find_minimum)[0:NL]).keys()

        # create an array of positions on the survival list for future use
        SL_positions = np.zeros(shape=(NL, self.n_dimensions))
        i = 0
        for ff in SL:
            SL_positions[i, :] = ff.position
            i += 1

        return [SL, SL_positions]

    def _new_fox_from_SL(self, ff, NL, SL_positions):
        # generate random number in [2, NL]
        n_from_SL = np.random.randint(2, NL + 1)

        # calculate new position and corresponding function value
        ff.position = np.sum(SL_positions[0:n_from_SL, :], axis=0) / n_from_SL
        ff.previous_function_value = ff.function_value
        ff.function_value = self.input_function(*ff.position)

    def solve(self):
        """
        Executes the solution iterations for the algorithm.

        Returns
        -------
        None
        """
        # initialize solution process and calculate number of foxes
        self._initialize_solve()
        self.n_foxes = int(np.ceil(10 + 2 * np.sqrt(self.n_dimensions)))
        self._initialize_stored_results(self.n_foxes)

        # initialize the crisp rules for the internal parameters
        a_crisps = [0.1, 1.5, 1.9]
        pa_crisps = [0.5, 0.85, 0.99]

        # set crowding tolerance to be used as a measure of closeness when calculating pD
        crowding_tolerance = 0.03

        # initialize survival list size
        NL = int(np.ceil(self.n_foxes / 4))

        # setting m vector bounds
        m_max = np.array([0.2, 0.4, 0.6])
        m_min = np.array([0.02, 0.04, 0.06])

        # initialize the flying foxe population
        population = [_flying_fox(n_dimensions=self.n_dimensions) for i in range(int(self.n_foxes))]

        # generate initial positions for the flying foxes
        [ff.random_position(self.b_lower, self.b_upper) for ff in population]

        # calculate initial function values of flying foxes' positions
        for ff in population:
            ff.function_value = self.input_function(*ff.position)

            # intialize previous function value as same
            ff.previous_function_value = ff.function_value

        # create the survival list
        SL, SL_positions = self._update_SL(population, NL)

        # find the coolest and hottest positions and function values
        if self.find_minimum: # set initial values
            coolest_value = np.Inf
            hottest_value = -np.Inf
        else:
            coolest_value = -np.Inf
            hottest_value = np.Inf

        coolest_position = None
        hottest_position = None

        coolest_position, coolest_value, hottest_position, hottest_value = self._update_cool_hot_positions(population, coolest_position, coolest_value, hottest_position, hottest_value)

        # start solution iterations
        iteration_count = 0
        unchanged_iterations = 0
        while self._check_stopping_criteria(iteration_count, unchanged_iterations, self.best_value):
            # setting the m vector for the current iteration and calculating corresponding delta values
            m = m_max - (iteration_count / self.max_iter) * (m_max - m_min)
            delta_max = abs(coolest_value - hottest_value)
            delta_1 = m[0] * delta_max
            delta_2 = m[1] * delta_max
            delta_3 = m[2] * delta_max

            # tune parameters and update positions for all flying foxes
            for ff in population:
                ff.tune_parameters(self.find_minimum, delta_1, delta_2, delta_3, delta_max, coolest_value, a_crisps, pa_crisps)
                is_far = ff.update_position(self.input_function, self.find_minimum, coolest_position, coolest_value, population, delta_1, delta_3)

                # if fox is "too far", kill the fox and use survival list to replace
                if is_far:
                    self._new_fox_from_SL(ff, NL, SL_positions)

                # update the survival list and coolest and hottest positions
                SL, SL_positions = self._update_SL(population, NL)
                coolest_position, coolest_value, hottest_position, hottest_value = self._update_cool_hot_positions(population, coolest_position, coolest_value, hottest_position, hottest_value)

            # calculate pD and create a list of flying foxes in the "coolest spot"
            nc = 0
            foxes_in_coolest_spot = []
            for ff in population:
                # check to see if the position is close enough to be considered in the "same" position as coolest position
                if abs((ff.function_value - coolest_value) / coolest_value) < crowding_tolerance:
                    nc += 1
                    foxes_in_coolest_spot.append(ff)

            pD = (nc - 1) / self.n_foxes

            # if odd number of foxes in coolest spot (and more than 1), replace one with the SL
            if nc > 1 and np.remainder(len(foxes_in_coolest_spot), 2) != 0:
                ff = foxes_in_coolest_spot[0]
                self._new_fox_from_SL(ff, NL, SL_positions)
                del foxes_in_coolest_spot[0]

            # replace some foxes that die from "smothering" with either SL or reproduction
            while len(foxes_in_coolest_spot) >= 2:
                # determine whether they die or not
                p_replace = np.random.uniform(0, 1)
                if p_replace < pD:
                    ff_1 = foxes_in_coolest_spot[0]
                    ff_2 = foxes_in_coolest_spot[1]
                    # determine method of replacement with 50% probability
                    if np.random.uniform(0, 1) > 0.5:
                        self._new_fox_from_SL(ff_1, NL, SL_positions)
                        self._new_fox_from_SL(ff_2, NL, SL_positions)
                    else:
                        # pick 2 random flying foxes from the population
                        random_1, random_2 = np.random.choice(population, size=2, replace=False)
                        L = np.random.uniform(0, 1, size=(self.n_dimensions))
                        ff_1.position = L * random_1.position + (1 - L) * random_2.position
                        ff_2.position = L * random_2.position + (1 - L) * random_1.position
                        ff_1.function_value = self.input_function(*ff_1.position)
                        ff_2.function_value = self.input_function(*ff_2.position)
                        ff_1.previous_function_value = ff_1.function_value.copy()
                        ff_2.previous_function_value = ff_2.function_value.copy()

                # remove these flying foxes from the coolest spot list
                del foxes_in_coolest_spot[0]
                del foxes_in_coolest_spot[0]

            # update the coolest and hottest positions and survival list
            SL, SL_positions = self._update_SL(population, NL)
            coolest_position, coolest_value, hottest_position, hottest_value = self._update_cool_hot_positions(population, coolest_position, coolest_value, hottest_position, hottest_value)

            # update best recorded solutions
            if self._first_is_better(coolest_value, self.best_value):
                self.best_position = coolest_position.copy()
                self.best_value = coolest_value
                unchanged_iterations = 0

            # store intermediate results for post-processing of specified
            if self.store_results:
                for i in range(len(population)):
                    ff = population[i]
                    self.stored_positions[iteration_count, i, :] = ff.position.copy()
                    self.stored_values[iteration_count, i, :] = ff.function_value

            # increment iterations
            iteration_count += 1
            unchanged_iterations += 1

        # save final iteration count
        self.completed_iter = iteration_count

        # truncate intermediate results
        if self.store_results:
            self.stored_positions = self.stored_positions[0: self.completed_iter, :, :]
            self.stored_values = self.stored_values[0: self.completed_iter, :, :]


class simulated_annealing(_local_search):
    """
    A Simulated Annealing optimizer.

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
    def __init__(self, input_function, b_lower=-10, b_upper=10, find_minimum=True, max_iter=100, sol_threshold=None, max_unchanged_iter=None, sigma_coeff=0.2, neighbor_dim_changes=1, initial_guess=None, store_results=False, start_temperature=10, alpha=0.9):
        """
        Constructs the necessary attributes for the algorithm.

        Parameters
        ----------
        input_function : function
            Function object for the algorithm to optimize.

        b_lower : float or list of floats or ndarray, default = -10
            List or array containing the lower bounds of each dimension of the search space.

        b_upper : float or list of floats or ndarray, default = 10
            List or array containing the upper bounds of each dimension of the search space.

        find_minimum : bool, default = True
            Indicates whether the optimum of interest is a minimum or maximum. If false, looks for maximum.

        max_iter : int, default = 100
            Maximum number of iterations. If reached, the algorithm terminates.

        sol_threshold : float, default = None
            If a solution is found past this threshold, the iterations stop. None indicates that the algorithm will not consider this.

        max_unchanged_iter : int, default = None
            If the solution does not improve after this many iterations, the iterations stop.

        store_results : bool, default = False
            Choose whether to save results or not. If true, results will be saved to the stored_positions and stored_values properties.

        start_temperature : float, default = 10.0
            Initial temperature to start iterations with.

        alpha : float, default = 0.9
            Temperature decay coefficient in [0.6, 1). The current temperature is multiplied by this at the end of each iteration.

        sigma_coeff: float, default = 0.2
            Coefficient in (0, 0.5] to be multiplied by the bound widths for each dimension; the corresponding number is used for the standard deviation in the neighbor generation process.

        neighbor_dim_changes : int, default = 1
            Number of dimensions to mutate during the generation of a new neighbor position. Must be in [1, number of dimensions]

        initial_guess : list or ndarray, default = None
            Initial guess used in the solution process. Leave as None to start with a random initial guess.
        """
        super().__init__(input_function=input_function, b_lower=b_lower, b_upper=b_upper, find_minimum=find_minimum, max_iter=max_iter, sol_threshold=sol_threshold, max_unchanged_iter=max_unchanged_iter, store_results=store_results, sigma_coeff=sigma_coeff, neighbor_dim_changes=neighbor_dim_changes, initial_guess=initial_guess)
        self.start_temperature = start_temperature # temperature that the algorithm starts at
        self.alpha = alpha # coefficient for rate of decay of temperature

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value >= 1:
            print("Warning: alpha must be in [0.6, 1). alpha has been set to 0.99.")
            self._alpha = 0.99
        elif value < 0.6:
            print("Warning: alpha must be in [0.6, 1). alpha has been set to 0.6.")
            self._alpha = 0.60
        else:
            self._alpha = value

    @property
    def current_position(self):
        return self._current_position

    @current_position.setter
    def current_position(self, value):
        self._current_position = value

    def solve(self):
        """
        Executes the solution iterations for the algorithm.

        Returns
        -------
        None
        """
        self._initialize_solve()
        self._initialize_stored_results(1)

        # initialize the temperature
        T = self.start_temperature

        # either create random initial guess within bounds or start with user's initial guess, then get the corresponding function value
        if self.initial_guess is None:
            self.current_position = np.random.uniform(self.b_lower, self.b_upper)
        else:
            self.current_position = self.initial_guess

        self.current_value = self.input_function(*self.current_position)

        # initializing best solution
        self.best_position = self.current_position.copy()
        self.best_value = self.current_value

        # begin solution iteration
        iteration_count = 0
        unchanged_iterations = 0
        while self._check_stopping_criteria(iteration_count, unchanged_iterations, self.best_value):
            # generating a new potential position using the neighboring function
            potential_position = self._generate_neighbor(self.current_position)

            # find the difference in the solutions
            potential_value = self.input_function(*potential_position)
            delta_F = potential_value - self.current_value

            # accept the solution if it is better, or calculate the probability that we accept the solution otherwise
            if self._first_is_better(potential_value, self.current_value):
                self.current_position = potential_position.copy()
                self.current_value = potential_value
            else:
                p = np.exp(-abs(delta_F) / T)
                u = np.random.uniform(0, 1)
                if u <= p:
                    self.current_position = potential_position.copy()
                    self.current_value = potential_value

            # replace best position and value if necessary
            if self._first_is_better(self.current_value, self.best_value):
                self.best_position = self.current_position.copy()
                self.best_value = self.current_value
                unchanged_iterations = 0

            # reduce the temperature
            T = T * self.alpha

            # store intermediate results for post-processing of specified
            if self.store_results:
                self.stored_positions[iteration_count, 0, :] = self.current_position.copy()
                self.stored_values[iteration_count, 0, :] = self.current_value

            # increment the iteration counters
            iteration_count += 1
            unchanged_iterations += 1

        # store final iteration count
        self.completed_iter = iteration_count

        # truncate stored results
        if self.store_results:
            self.stored_positions = self.stored_positions[0: self.completed_iter, :, :]
            self.stored_values = self.stored_values[0: self.completed_iter, :, :]


class tabu_search(_local_search):
    """
    A Tabu Search optimizer.

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
    def __init__(self, input_function, b_lower=-10, b_upper=10, find_minimum=True, max_iter=100, sol_threshold=None, max_unchanged_iter=None, sigma_coeff=0.2, neighbor_dim_changes=1, initial_guess=None, store_results=False, tenure=5, n_candidates=5, neighbor_tolerance=0.02):
        """
        Constructs the necessary attributes for the algorithm.

        Parameters
        ----------
        input_function : function
            Function object for the algorithm to optimize.

        b_lower : float or list of floats or ndarray, default = -10
            List or array containing the lower bounds of each dimension of the search space.

        b_upper : float or list of floats or ndarray, default = 10
            List or array containing the upper bounds of each dimension of the search space.

        find_minimum : bool, default = True
            Indicates whether the optimum of interest is a minimum or maximum. If false, looks for maximum.

        max_iter : int, default = 100
            Maximum number of iterations. If reached, the algorithm terminates.

        sol_threshold : float, default = None
            If a solution is found past this threshold, the iterations stop. None indicates that the algorithm will not consider this.

        max_unchanged_iter : int, default = None
            If the solution does not improve after this many iterations, the iterations stop.

        store_results : bool, default = False
            Choose whether to save results or not. If true, results will be saved to the stored_positions and stored_values properties.

        tenure : int, default = 5
            Number of previous positions stored on the tabu list. These positions (within a specified tolerance) will be prohibited in following iterations.

        n_candidates : int, default = 5
            Number of new candidate solutions to guess at each iteration. The best solution that is not tabu is used.

        neighbor_tolerance : float, default = 0.02
            Portion of dimension width to use as a tolerance when determining whether a potential position is tabu.

        sigma_coeff: float, default = 0.2
            Coefficient in (0, 0.5] to be multiplied by the bound widths for each dimension; the corresponding number is used for the standard deviation in the neighbor generation process.

        neighbor_dim_changes : int, default = 1
            Number of dimensions to mutate during the generation of a new neighbor position. Must be in [1, number of dimensions]

        initial_guess : list or ndarray, default = None
            Initial guess used in the solution process. Leave as None to start with a random initial guess.
        """
        super().__init__(input_function=input_function, b_lower=b_lower, b_upper=b_upper, find_minimum=find_minimum, max_iter=max_iter, sol_threshold=sol_threshold, max_unchanged_iter=max_unchanged_iter, store_results=store_results, sigma_coeff=sigma_coeff, neighbor_dim_changes=neighbor_dim_changes, initial_guess=initial_guess)
        self.tenure = tenure # number many iterations a solution stays on the tabu list
        self.n_candidates = n_candidates # number of candidate solutions generated during each iteration
        self.neighbor_tolerance = neighbor_tolerance # ratio of bound width to be used for checking if guesses are too close to a guess on the tabu list

    @property
    def tenure(self):
        return self._tenure

    @tenure.setter
    def tenure(self, value):
        if type(value) is int:
            self._tenure = value
        else:
            print("Warning: tenure must be an integer. It has been rounded up from the input value.")
            self._tenure = np.ceil(value)

    @property
    def n_candidates(self):
        return self._n_candidates

    @n_candidates.setter
    def n_candidates(self, value):
        if type(value) is not int:
            raise TypeError("n_candidates must be an integer.")
        elif value < 1:
            print("Warning: n_candidates must be at least 1. A value of 1 has been set.")
            self._n_candidates = 1
        else:
            self._n_candidates = value

    @property
    def neighbor_tolerance(self):
        return self._neighbor_tolerance

    @neighbor_tolerance.setter
    def neighbor_tolerance(self, value):
        if value <= 0:
            print("Warning: neighbor_tolerance must be in (0, 1]. A value of 0.01 has been set.")
            self._neighbor_tolerance = 0.01
        elif value > 1:
            print("Warning: neighbor_tolerance must be in (0, 1]. A value of 1 has been set.")
            self._neighbor_tolerance = 1
        else:
            self._neighbor_tolerance = value

    # function to check whether or not the candidate is within the tolerance (specified by user) of any of the tabu solutions. returns true if too close
    def _check_candidate_proximity(self, candidate, tabu_list):
        too_close = False

        # find deltas from all tabu positions
        deltas_from_tabu = abs(tabu_list - candidate)

        # find where any delta is within the neighbor tolerance for that dimension
        where_delta_within_neighbor_tol = deltas_from_tabu < self.neighbor_tolerance

        # find matches where all position values for the candidate are close to those of a tabu position
        where_matches_tabu_sol = where_delta_within_neighbor_tol.sum(axis=1) == self.n_dimensions

        # if the candidate is too close to a single tabu position, return True
        if where_matches_tabu_sol.sum() > 0:
            too_close = True

        return too_close

    def solve(self):
        """
        Executes the solution iterations for the algorithm.

        Returns
        -------
        None
        """
        self._initialize_solve()
        self._initialize_stored_results(1)

        # either create random initial guess within bounds or start with user's initial guess, then get the corresponding function value
        if self.initial_guess is None:
            self.current_position = np.random.uniform(self.b_lower, self.b_upper)
        else:
            self.current_position = self.initial_guess

        self.current_value = self.input_function(*self.current_position)

        # initializing best solution
        self.best_position = self.current_position.copy()
        self.best_value = self.current_value

        # initializing tabu list and setting bound widths
        tabu_list = np.zeros(shape=(self.tenure, self.n_dimensions))
        tabu_list[0, :] = self.current_position.copy()

        # initializing candidate parameters
        best_candidate_position = self.current_position.copy()
        best_candidate_value = self.current_value

        # begin solution iterations
        current_tabu_index = 0 # keeps track of which tabu solution to be replaced. this is more efficient than deleting and appending tabu positions
        iteration_count = 0
        unchanged_iterations = 0
        while self._check_stopping_criteria(iteration_count, unchanged_iterations, self.best_value):
            # generate neighbors and evaluate their candidacy as potential solutions
            for c in range(self.n_candidates):
                current_candidate_position = self._generate_neighbor(self.current_position)
                current_candidate_value = self.input_function(*current_candidate_position)
                # check if candidate has close proximity to a position in the tabu list
                if self._check_candidate_proximity(current_candidate_position, tabu_list): # if in tabu list
                    # even if tabu, check to see if candidate solution is better than current overall best. can override tabu if so (aspiration criterion)
                    if self._first_is_better(current_candidate_value, self.best_value):
                        best_candidate_position = current_candidate_position.copy()
                        best_candidate_value = current_candidate_value
                else: # if not in tabu list
                    if self._first_is_better(current_candidate_value, best_candidate_value):
                        best_candidate_position = current_candidate_position.copy()
                        best_candidate_value = current_candidate_value

            # replace the best known value if the best candidate solution from this iteration is better
            if self._first_is_better(best_candidate_value, self.best_value):
                self.best_position = best_candidate_position.copy()
                self.best_value = best_candidate_value
                unchanged_iterations = 0

            # update the tabu list with the best candidate from this iteration
            current_tabu_index += 1 # increment that tabu index
            if current_tabu_index == self.tenure: # start back at beginning if index is larger than tabu memory
                current_tabu_index = 0

            tabu_list[current_tabu_index, :] = best_candidate_position.copy()

            # store intermediate results for post-processing of specified
            if self.store_results:
                self.stored_positions[iteration_count, 0, :] = best_candidate_position.copy()
                self.stored_values[iteration_count, 0, :] = best_candidate_value

            # update the iteration counter
            iteration_count += 1
            unchanged_iterations += 1

        # store final iteration count
        self.completed_iter = iteration_count

        # truncate stored results
        if self.store_results:
            self.stored_positions = self.stored_positions[0: self.completed_iter, :, :]
            self.stored_values = self.stored_values[0: self.completed_iter, :, :]










