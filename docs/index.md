![optiseek logo](images/optiseek_logo_small.png)

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

## License

`optiseek` was created by Alex Dundore. It is licensed under the terms of the MIT license.

## Credits and Dependencies

`optiseek` is powered by [`numpy`](https://numpy.org/).