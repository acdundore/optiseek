import numpy as np


class _variable:
    def __init__(self, var_name, var_type, bounds, choices, log_scale, internal_bounds):
        # initialize variables
        self.var_name = var_name
        self._var_type = var_type
        self.bounds = bounds
        self.choices = choices
        self.log_scale = log_scale
        self._internal_bounds = np.array(internal_bounds) # convert to array

        # calculate parameters to transform from specified to internal bounds
        if self.log_scale == False:
            self._scale = (self.bounds[1] - self.bounds[0]) / (self._internal_bounds[1] - self._internal_bounds[0])
            self._shift = -self._scale * self._internal_bounds[0] + self.bounds[0]
        else:
            self._scale = (np.log10(self.bounds[1]) - np.log10(self.bounds[0])) / (self._internal_bounds[1] - self._internal_bounds[0])
            self._shift = -self._scale * self._internal_bounds[0] + np.log10(self.bounds[0])

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

    @property
    def log_scale(self):
        return self._log_scale

    @log_scale.setter
    def log_scale(self, value):
        if value == True and np.any(self.bounds <= 0):
            raise ValueError('Bounds must both be positive to enable log scaling.')
        if type(value) is not bool:
            raise TypeError('log_scale must be either True or False.')
        else:
            self._log_scale = value

    # function to convert a value from internal position to specified position, including encodings for categorical, boolean, and integers
    def _internal_to_specified(self, internal_position):
        # convert from the internal coordinate system to the specified system
        if self.log_scale == False:
            specified_position = self._scale * internal_position + self._shift
        else:
            specified_position = 10 ** (self._scale * internal_position + self._shift)

        # encode the variable if necessary
        if self._var_type in ['categorical', 'bool']:
            encoded_position = self.choices[int(min(max(self.bounds[0], np.floor(specified_position)), self.bounds[1] - 1))]
        elif self._var_type == 'int':
            encoded_position = int(min(max(self.bounds[0], np.round(specified_position)), self.bounds[1]))
        else:
            encoded_position = min(max(self.bounds[0], specified_position), self.bounds[1])

        return encoded_position

class var_float(_variable):
    def __init__(self, var_name, bounds, log_scale=False):
        super().__init__(var_name, var_type='float', bounds=bounds, choices=None, log_scale=log_scale, internal_bounds=[-1, 1])

class var_int(_variable):
    def __init__(self, var_name, bounds, log_scale=False):
        super().__init__(var_name, var_type='int', bounds=bounds, choices=None, log_scale=log_scale, internal_bounds=[-1, 1])

class var_categorical(_variable):
    def __init__(self, var_name, choices):
        if type(choices) is not list:
            raise TypeError('choices must be a list.')
        super().__init__(var_name, var_type='categorical', bounds=[0, len(choices)], choices=choices, log_scale=False, internal_bounds=[-1, 1])

class var_bool(_variable):
    def __init__(self, var_name):
        super().__init__(var_name, var_type='bool', bounds=[0, 2], choices=[False, True], log_scale=False, internal_bounds=[-1, 1])