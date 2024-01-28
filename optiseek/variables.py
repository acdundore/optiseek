import numpy as np


class _variable:
    def __init__(self, var_name, var_type, bounds, choices, internal_bounds):
        # initialize variables
        self.var_name = var_name
        self._var_type = var_type
        self.bounds = bounds
        self.choices = choices
        self._internal_bounds = np.array(internal_bounds) # convert to array

        # calculate parameters to transform from specified to internal bounds
        self._scale = (self.bounds[1] - self.bounds[0]) / (self._internal_bounds[1] - self._internal_bounds[0])
        self._shift = -self._scale * self._internal_bounds[0] + self.bounds[0]

    @property
    def var_name(self):
        return self._var_name

    @var_name.setter
    def var_name(self, value):
        self._var_name = value

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        if isinstance(value, np.ndarray):
            self._bounds = value
        elif type(value) is list:
            self._bounds = np.array(value, dtype=float)
        else:
            self._bounds = np.array([value], dtype=float)

    @property
    def choices(self):
        return self._choices

    @choices.setter
    def choices(self, value):
        if self._var_type not in ['categorical', 'bool']:
            self._choices = None
        else:
            self._choices = value

    # function to convert a value from internal position to specified position, including encodings for categorical, boolean, and integers
    def _internal_to_specified(self, internal_position):
        specified_position = self._scale * internal_position + self._shift
        if self._var_type in ['categorical', 'bool']:
            encoded_position = self.choices[int(min(max(self.bounds[0], np.floor(specified_position)), self.bounds[1] - 1))]
        elif self._var_type == 'int':
            encoded_position = int(min(max(self.bounds[0], np.round(specified_position)), self.bounds[1]))
        else:
            encoded_position = min(max(self.bounds[0], specified_position), self.bounds[1])

        return encoded_position

class var_float(_variable):
    def __init__(self, var_name, bounds):
        super().__init__(var_name, var_type='float', bounds=bounds, choices=None, internal_bounds=[-1, 1])

class var_int(_variable):
    def __init__(self, var_name, bounds):
        super().__init__(var_name, var_type='int', bounds=bounds, choices=None, internal_bounds=[-1, 1])

class var_categorical(_variable):
    def __init__(self, var_name, choices):
        if type(choices) is not list:
            raise TypeError('choices must be a list.')
        super().__init__(var_name, var_type='categorical', bounds=[0, len(choices)], choices=choices, internal_bounds=[-1, 1])

class var_bool(_variable):
    def __init__(self, var_name):
        super().__init__(var_name, var_type='bool', bounds=[0, 2], choices=[False, True], internal_bounds=[-1, 1])