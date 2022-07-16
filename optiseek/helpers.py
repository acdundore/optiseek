import itertools
import numpy as np

# TODO: this is in progress

class grid_search:
    def __init__(self, algorithm, input_function, param_options, show_progress=False):
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




























