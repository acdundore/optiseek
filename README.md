![optiseek logo](docs/images/optiseek_logo_small.png)

[![Downloads](https://static.pepy.tech/personalized-badge/optiseek?period=total&units=none&left_color=black&right_color=red&left_text=Downloads)](https://pepy.tech/project/optiseek)

An open source collection of single-objective optimization algorithms for multi-dimensional functions.

The purpose of this library is to give users access to a variety of versatile black-box optimization algorithms with extreme ease of use and interoperability.
The parameters of each of the algorithms can be tuned by the users and there is a high level of parameter uniformity between algorithms.

Benefits of using Optiseek include:

- support for float, integer, categorical, and boolean inputs for objective functions
- compatibility with black-box objective functions (requires no information on gradients or form of function)
- the algorithms are simple compared to alternatives (e.g. Bayesian optimization) with faster runtime on basic objective functions
- competitive convergence for computionally expensive objective functions in terms of number of function evaluations
- seamless integration into ML pipelines for hyper-parameter tuning
- access to a variety of stopping criteria, suitable for both computationally expensive and cheap objective functions
- carefully chosen default parameters for algorithms, with ability for user-defined fine tuning

## Installation

```bash
pip install optiseek
```

## Usage

### Basic Implementation

`optiseek` provides access to numerous optimization algorithms that require minimal effort from the user. An example using the well-known particle swarm optimization algorithm can be as simple as this:

```python
from optiseek.variables import var_float
from optiseek.metaheuristics import particle_swarm_optimizer
from optiseek.testfunctions import booth

# define a list of variables and their domains for the objective function
var_list = [
	var_float('x1', [-10, 10]),
	var_float('x2', [-10, 10])
]	

# create an instance of the algorithm to optimize the booth test function and set its parameters
alg = particle_swarm_optimizer(booth, var_list)

# define stopping criteria and optimize
alg.optimize(find_minimum=True, max_iter=10)

# show the results!
print(f'best_value = {alg.best_value:.5f}')
print(f'best_position = {alg.best_position}')
print(f'n_iter = {alg.completed_iter}')
```

```profile
best_value = 0.00217
best_position = {'x1': 0.9921537320723116, 'x2': 3.0265668104168326}
n_iter = 10
```

This is a fairly basic example implementation without much thought put into parameter selection. Of course, the user is free to tune the parameters of the algorithm any way they would like.

### Tuning ML Hyperparameters

The steps to tuning a custom ML algorithm with `optiseek` are straighforward:

1. Prepare the training data
- Create a function that performs cross-validation with the hyperparameters are arguments and error metric as output
- Define the search space
- Pass this to an algorithm in `optiseek` and optimize

```python
# imports
from optiseek.variables import *
from optiseek.metaheuristics import particle_swarm_optimizer

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_validate

from lightgbm import LGBMRegressor

# import and prepare the data
california_housing = fetch_california_housing(as_frame=True)
X = california_housing.data
y = california_housing.target


# set up cross-validation of the model as the objective function
def objective_function(learning_rate, num_leaves, max_depth, min_child_weight, min_child_samples, subsample, colsample_bytree, reg_alpha):
    # assign the parameters
    params = {
        'n_estimators': 300,
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'min_child_samples': min_child_samples,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': reg_alpha,
        'verbose': -1
    }

    # create the model
    model = LGBMRegressor(**params)

    # cross validate and return average validation MRSE
    cv_results = cross_validate(model, X, y, scoring='neg_root_mean_squared_error', cv=5)
    cv_score = -np.mean(cv_results['test_score'])

    return cv_score


# define the search space
var_list = [
    var_float('learning_rate', [0.001, 0.3]),
    var_int('num_leaves', [20, 3000]),
    var_int('max_depth', [3, 12]),
    var_float('min_child_weight', [0.0005, 0.1]),
    var_int('min_child_samples', [5, 50]),
    var_float('subsample', [0.5, 1]),
    var_float('colsample_bytree', [0.5, 1]),
    var_float('reg_alpha', [0.001, 0.1])
]

# instantiate an optimization algorithm with the function and search domain
alg = particle_swarm_optimizer(objective_function, var_list, results_filename='cv_results.csv')

# set stopping criteria and optimize
alg.optimize(find_minimum=True, max_function_evals=300)

# show the results!
print(f'best_value = {alg.best_value:.5f}')
print(f'best_position = {alg.best_position}')
```

```profile
best_value = 0.60881
best_position = {'learning_rate': 0.24843279626513076, 'num_leaves': 3000, 'max_depth': 3, 'min_child_weight': 0.06303795879741575, 'min_child_samples': 27, 'subsample': 0.5, 'colsample_bytree': 0.9620615099733333, 'reg_alpha': 0.022922559999999998}
```

## Documentation

For full documentation, visit the [github pages site](https://acdundore.github.io/optiseek/).

## License

`optiseek` was created by Alex Dundore. It is licensed under the terms of the MIT license.

## Credits and Dependencies

`optiseek` is powered by [`numpy`](https://numpy.org/).