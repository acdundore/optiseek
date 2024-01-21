# Differential Evolution

This class represents the differential evolution algorithm developed by Storn and Price. 

This is an evolutionary algorithm that utilizes vector-based genetic crossovers. It contains the typical components of a genetic algorithm (mutation, crossover, & selection)
but has a special unique form of crossover that makes it widely applicable to a diverse set of problems. There are also very few parameters, making it simple to tune.

---

> *class* optiseek.metaheuristics.**differential_evolution**(*input_function, var_list, linspaced_initial_positions=True, results_filename=None, n_agents=None, weight=0.8, p_crossover=0.9*)

---

### Parameters

All parameters are also class attributes and may be modified after instantiation.

| Parameter | Description |
|---|---|
| input_function : *function* | Function that the algorithm will use to search for an optimum.<br/> \*args will be passed to the function within the solver. |
| var_list : *list of variables* | List of variables (see variable types) to define the search space.<br/> These correspond to the arguments of the objective function<br/> and must be in the exact same order. |
| linspaced_initial_positions : *bool* | If true, creates a linearly spaced set of points in each search<br/> dimension, and the initial positions of the population are set to<br/> mutually exclusive combinations of these points. This guarantees<br/> that there will be no empty spots in a single dimension. If false,<br/> random initial positions are chosen. |
| results_filename : *string* | If a file name is passed (ending in '.csv'), the results will be written<br/> to this file after each function evaluation. This can noticeably slow<br/> down solution iterations for quick objective functions. For greedy<br/> functions, it can be beneficial to do this in case the script is<br/> interrupted. |
| n_agents : *int* | Number of agents to use in the population. If<br/> set to `None`, the population size will be based on the heuristic<br/> 10 + 2 \* sqrt(n_dims), where n_dims is the dimensionality of<br/> the search space. This is typically sufficient to explore the<br/> whole search space. |
| weight : *float* | Differential weight coefficient in [0, 2]. Higher values increase<br/> the impact of genetic crossover. |
| p_crossover : *float* | Probability in [0, 1] that a gene crossover will occur for<br/> each dimension. |

---

### Attributes

| Attribute | Description |
|---|---|
| best_position : *dict* | Dictionary containing the most optimal position found during the solution<br/> iterations, with variable names as keys and corresponding position values<br/> as values. |
| best_value : *float* | Most optimal function value found during the solution iterations. |
| completed_iter : *int* | Number of iterations completed during the solution process. |
| results : *pd.DataFrame* | DataFrame of results throughout the iterations. For each iteration, the<br/> function value and position for each member of the population are provided. |

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
from optiseek.metaheuristics import differential_evolution
from optiseek.testfunctions import booth

# define a list of variables and their domains for the objective function
var_list = [
	var_float('x1', [-10, 10]),
	var_float('x2', [-10, 10])
]	

# create an instance of the algorithm to optimize the booth test function and set its parameters
alg = differential_evolution(booth, var_list)

# define stopping criteria and optimize
alg.optimize(find_minimum=True, max_iter=10, sol_threshold=0.05)

# show the results!
print(f'best_value = {alg.best_value:.5f}')
print(f'best_position = {alg.best_position}')
print(f'n_iter = {alg.completed_iter}')
```

```profile
best_value = 0.07129
best_position = {'x1': 0.9199999999999992, 'x2': 3.1733333333333325}
n_iter = 10
```

---

### References

[Differential Evolution on Wikipedia](https://en.wikipedia.org/wiki/Differential_evolution)