# Firefly Algorithm

This class represents the firefly algorithm developed by Xin-She Yang. 

This algorithm is based on the flashing patterns and swarm behavior of fireflies. Fireflies are attracted to others based on their proximity in the search space and
the brightness (function values) of others. Their movements also have a stochastic component.

---

> *class* optiseek.metaheuristics.**firefly_algorithm**(*input_function, var_list, linspaced_initial_positions=True, results_filename=None, n_fireflies=None, beta=1.0, alpha=0.05, gamma=0.5*)

---

### Parameters

All parameters are also class attributes and may be modified after instantiation.

| Parameter | Description |
|---|---|
| input_function : *function* | Function that the algorithm will use to search for an optimum.<br/> \*args will be passed to the function within the solver. |
| var_list : *list of variables* | List of variables (see variable types) to define the search space.<br/> These correspond to the arguments of the objective function<br/> and must be in the exact same order. |
| linspaced_initial_positions : *bool* | If true, creates a linearly spaced set of points in each search<br/> dimension, and the initial positions of the population are set to<br/> mutually exclusive combinations of these points. This guarantees<br/> that there will be no empty spots in a single dimension. If false,<br/> random initial positions are chosen. |
| results_filename : *string* | If a file name is passed (ending in '.csv'), the results will be written<br/> to this file after each function evaluation. This can noticeably slow<br/> down solution iterations for quick objective functions. For greedy<br/> functions, it can be beneficial to do this in case the script is<br/> interrupted. |
| n_fireflies : *int* | Number of fireflies to use in the swarm population. If set to<br/> `None`, the population size will be based on the heuristic<br/> 10 + 2 \* sqrt(n_dims), where n_dims is the dimensionality of<br/> the search space. This is typically sufficient to explore the<br/> whole search space. |
| beta : *float* | Linear visibility coefficient in [0.1, 1.5]. Lower value indicates<br/> that the fireflies are less attracted to each other. |
| alpha : *float* | Coefficient in [0, 0.1] that is a portion of each dimension's<br/>  bound widths to use for the random walk. |
| gamma : *float* | Exponential visibility coefficient in [0.01, 1]. Higher value<br/> indicates that the fireflies are less attracted to each other. |

---

### Attributes

| Attribute | Description |
|---|---|
| best_position : *dict* | Dictionary containing the most optimal position found during the solution<br/> iterations, with variable names as keys and corresponding position values<br/> as values. |
| best_value : *float* | Most optimal function value found during the solution iterations. |
| completed_iter : *int* | Number of iterations completed during the solution process. |
| results : *pandas df* | DataFrame of results throughout the iterations. For each iteration, the<br/> function value and position for each member of the population are provided. |

---

### Methods

**.optimize**(*find_minimum, max_iter=None, max_function_evals=None, max_unchanged_iter=None, sol_threshold=None*)
	
> Executes the algorithm solution with the current parameters. 
Results will be stored to the class attributes. 
Either *max_iter* or *max_function_evals* must be specified in order to prevent an endless optimization loop.
If any of the criteria are met during optimization, the process is terminated.

> | Argument | Description |
|---|---|
| find_minimum : *bool* | Indicates whether the optimimum of interest is a minimum or<br/> maximum. If true, looks for minimum. If false, looks for maximum. |
| max_iter : *int* | Maximum number of iterations. The algorithm will terminate after<br/> completing this many iterations. `None` indicates that the algorithm<br/> will not consider this. |
| max_function_evals : *int* | Maximum number of function evaluations. Must be greater than the<br/> size of the population (i.e. complete at least one iteration). The<br/> algorithm will terminate after completing this many function<br/> evaluations. This is a preferable metric for greedy algorithms. <br/>`None` indicates that the algorithm will not consider this. |
| max_unchanged_iter : *int* | If the solution does not improve after this many iterations,<br/> the optimization terminates. `None` indicates that the algorithm<br/> will not consider this. |
| sol_threshold : *float* | If a solution is found better than this threshold, the iterations<br/> stop. `None` indicates that the algorithm will not consider this. |

---

### Example

```python
from optiseek.metaheuristics import firefly_algorithm
from optiseek.testfunctions import booth

# define a list of variables and their domains for the objective function
var_list = [
	var_float('x1', [-10, 10]),
	var_float('x2', [-10, 10])
]	

# create an instance of the algorithm to optimize the booth test function and set its parameters
alg = firefly_algorithm(booth, var_list)

# define stopping criteria and optimize
alg.optimize(find_minimum=True, max_iter=10, sol_threshold=0.05)

# show the results!
print(f'best_value = {alg.best_value:.5f}')
print(f'best_position = {alg.best_position}')
print(f'n_iter = {alg.completed_iter}')
```

```profile
best_value = 0.00290
best_position = {'x1': 1.0080388736395434, 'x2': 2.9699633384695314}
n_iter = 6
```

---

### References

[Firefly Algorithm on Wikipedia](https://en.wikipedia.org/wiki/Firefly_algorithm)