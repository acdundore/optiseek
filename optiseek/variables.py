import numpy as np


class _variable:
    def __init__(self, var_name, var_type, specified_bounds, internal_bounds, choices):
        # initialize variables
        self.var_name = var_name
        self.var_type = var_type
        self.specified_bounds = specified_bounds
        self.internal_bounds = internal_bounds
        self.choices = choices

        # calculate parameters to transform from specified to internal bounds
        self._scale = (self.specified_bounds[1] - self.specified_bounds[0]) / (self.internal_bounds[1] - self.internal_bounds[0])
        self._shift = -self._scale * self.internal_bounds[0] + self.specified_bounds[0]

    @property
    def var_name(self):
        return self._var_name

    @var_name.setter
    def var_name(self, value):
        self._var_name = value

    @property
    def var_type(self):
        return self._var_type

    @var_type.setter
    def var_type(self, value):
        if value not in ['float', 'int', 'categorical', 'bool']:
            raise TypeError('var_name must be one of the following: ''float'', ''int'', ''categorical'', ''bool''')
        self._var_type = value

    @property
    def specified_bounds(self):
        return self._specified_bounds

    @specified_bounds.setter
    def specified_bounds(self, value):
        if isinstance(value, np.ndarray):
            self._specified_bounds = value
        elif type(value) is list:
            self._specified_bounds = np.array(value, dtype=float)
        else:
            self._specified_bounds = np.array([value], dtype=float)

    @property
    def internal_bounds(self):
        return self._internal_bounds

    @internal_bounds.setter
    def internal_bounds(self, value):
        if isinstance(value, np.ndarray):
            self._internal_bounds = value
        elif type(value) is list:
            self._internal_bounds = np.array(value, dtype=float)
        else:
            self._internal_bounds = np.array([value], dtype=float)

    @property
    def choices(self):
        return self._choices

    @choices.setter
    def choices(self, value):
        if self.var_type not in ['categorical', 'bool']:
            self._choices = None
        else:
            self._choices = value

    # function to convert a value from internal position to specified position, including encodings for categorical, boolean, and integers
    def _internal_to_specified(self, internal_position):
        specified_position = self._scale * internal_position + self._shift
        if self.var_type in ['categorical', 'bool']:
            encoded_position = self.choices[int(min(max(self.specified_bounds[0], np.floor(specified_position)), self.specified_bounds[1] - 1))]
        elif self.var_type == 'int':
            encoded_position = int(min(max(self.specified_bounds[0], np.round(specified_position)), self.specified_bounds[1]))
        else:
            encoded_position = min(max(self.specified_bounds[0], specified_position), self.specified_bounds[1])

        return encoded_position

class var_float(_variable):
    def __init__(self, var_name, bounds):
        super().__init__(var_name, var_type='float', specified_bounds=bounds, internal_bounds=[-1, 1], choices=None)

class var_int(_variable):
    def __init__(self, var_name, bounds):
        super().__init__(var_name, var_type='int', specified_bounds=bounds, internal_bounds=[-1, 1], choices=None)

class var_categorical(_variable):
    def __init__(self, var_name, choices):
        if type(choices) is not list:
            raise TypeError('choices must be a list.')
        super().__init__(var_name, var_type='categorical', specified_bounds=[0, len(choices)], internal_bounds=[-1, 1], choices=choices)

class var_bool(_variable):
    def __init__(self, var_name):
        super().__init__(var_name, var_type='bool', specified_bounds=[0, 2], internal_bounds=[-1, 1], choices=[False, True])