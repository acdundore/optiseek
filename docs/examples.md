# Quick Start Guide

The purpose of this package is to give users access to a class of optimization algorithms called metaheuristics 
with an easy-to-use and versatile API. The word "metaheuristics" comes from "meta", meaning *beyond*, and "heuristic", meaning *to find*.
Most algorithms of this type are based on a naturally occurring process that has the emergent property of tending towards an optimum.
For example, there are metaheuristics influenced by natural selection of genes (evolutionary algorithms), 
swarm behavior of insects (particle swarm optimization), 
and annealing in metals at the atomic level (simulated annealing), to name a few.
Although they are not guaranteed to find the global optimum, these algorithms perform exceptionally well in both
exploration and exploitation of large, high-dimensional search spaces. They also have the benefit of not 
needing any knowledge of the form of the function (i.e. black box functions).

The algorithms in `optiseek` are applicable to a wide array of problems, including:

- high dimensionality black-box objective functions
- computationally expensive objective functions (i.e. the function evaluations are costly or slow), like hyperparameter tuning in Machine Learning
- non-expensive objective functions, as the optimizers have little computational overhead
- problems with a considerable amount of constraints on the search space
- objective functions with variables of mixed types (continuous, integer, categorical, boolean)

To demonstrate the versatility of the algorithms in this package, some examples are provided.

---

### Example: Continous Variables

First, let's optimize the 2-Dimensional Ackley function. This is a problem with 2 continuous variables as input,
and a minimum of zero at [0, 0]. The `ackley` function is included with `optiseek`.

```python
from optiseek.variables import var_float
from optiseek.metaheuristics import particle_swarm_optimizer
from optiseek.testfunctions import ackley

# define variable list and search domain
var_list = [
    var_float('x1', [-10, 10]),
    var_float('x2', [-10, 10])
]

# instantiate an optimization algorithm with the function and search domain
alg = particle_swarm_optimizer(ackley, var_list)

# set stopping criteria and optimize
alg.optimize(find_minimum=True, max_iter=100)

# show the results!
print(f'best_value = {alg.best_value:.5f}')
print(f'best_position = {alg.best_position}')
```

```profile
best_value = 0.00000
best_position = {'x1': 2.8710730581797484e-17, 'x2': -3.254202051602899e-16}
```

---

### Example: Mixed-Type Variables

Because `optiseek`'s algorithms also support mixed-type variables for the objective function, we will
create a custom version of the `ackley` for demonstration purposes. We will modify the standard function output
with an integer, categorical, and boolean variable. The way the modified function is written, it will maintain
a minimum of zero at [0, 0].

```python
from optiseek.variables import *
from optiseek.metaheuristics import particle_swarm_optimizer
from optiseek.testfunctions import ackley

# create modified version of the ackley function
def ackley_mixed(x1, x2, x_int, x_cat, x_bool):
    output = ackley(x1, x2)
    output += x_int * 2
    if x_cat == 'A':
        output = output
    elif x_cat == 'B':
        output = 3 * output
    else:
        output = 2 * output
    output += int(x_bool) * 2

    return output

# define variable list and search domain
var_list = [
    var_float('x1', [-10, 10]),
    var_float('x2', [-10, 10]),
    var_int('x_int', [0, 5]),
    var_categorical('x_cat', ['A', 'B', 'C']),
    var_bool('x_bool')
]

# instantiate an optimization algorithm with the function and search domain
alg = particle_swarm_optimizer(ackley_mixed, var_list)

# set stopping criteria and optimize
alg.optimize(find_minimum=True, max_iter=100)

# show the results!
print(f'best_value = {alg.best_value:.5f}')
print(f'best_position = {alg.best_position}')
```

```profile
best_value = 0.00000
best_position = {'x1': 2.9828990486468627e-16, 'x2': -1.8893781942141117e-16, 'x_int': 0, 'x_cat': 'A', 'x_bool': False}
```

---

### Example: Hyperparameter Tuning in ML Algorithms

For this example, we will use the `scikit-learn` and `lightgbm` packages for our ML tools,
the California Housing dataset, and MRSE as the error metric. Note that the example uses `max_function_evals` 
for the stopping criterion, because performing cross-validation makes for a greedy objective function.
In this case, we want to set a limit on the maximum number of allowed function evaluations.
Also, we pass a file name to the `results_filename` optimization argument to preserve our results in the case
that we need to terminate the algorithm prematurely.

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

---

##### Example: Constrained Objective Functions

Refer to the `penalty_constraints` page for multiple examples on applying constraints to objective functions.