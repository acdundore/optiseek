# Simulated Annealing

This class represents the simulated annealing algorithm developed by Kirkpatrick et al. 

This is a local search method that takes inspiration from the annealing process in metals. 
Unlike deterministic gradient-based search methods, this algorithm has the ability to avoid being trapped in local optima. 
This is accomplished because there is a probability that a worse solution could be accepted during each iteration. 
As the iterations progress (i.e. temperature decreases), this probability diminishes and the algorithm is able to settle into what is hopefully a global optimum.

---

> *class* optiseek.metaheuristics.**simulated_annealing**(*input_function=None, var_list=None, results_filename=None, initial_guess=None, sigma_coeff=0.2, neighbor_dim_changes=-1, start_temperature=10, alpha=0.90*)

---

### Parameters

All parameters are also class attributes and may be modified after instantiation.

| Parameter | Description |
|---|---|
| input_function : *function* | Function that the algorithm will use to search for an optimum.<br/> \*args will be passed to the function within the solver. |
| var_list : *list of variables* | List of variables (see variable types) to define the search space.<br/> These correspond to the arguments of the objective function<br/> and must be in the exact same order. |
| results_filename : *string* | If a file name is passed (ending in '.csv'), the results will be<br/> written to this file after each function evaluation. This can<br/> noticeably slow down solution iterations for quick objective<br/> functions. For greedy functions, it can be beneficial to do this<br/> in case the script is interrupted. |
| initial_guess : *list of floats or ndarray* | Initial guess used in the solution process. Leave as `None` to<br/> start with a random initial guess. |
| sigma_coeff : *float* | Coefficient in (0, 0.5] to be multiplied by the bound widths<br/> for each dimension; the corresponding number is used for<br/> the standard deviation in the neighbor generation process. |
| neighbor_dim_changes : *int* | Number of dimensions to mutate during the generation of<br/> a new neighbor position. Must be in [1, number of dimensions].<br/> If set to -1, all dimensions will be mutated each iteration. |
| start_temperature : *float* | Initial temperature to start iterations with. |
| alpha : *float* | Temperature decay coefficient in \[0.6, 1). The current<br/> temperature is multiplied by this at the end of each iteration. |

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
In the case of local search algorithms such as this, *max_iter* or *max_function_evals* are handled the same way.
If any of the criteria are met during optimization, the process is terminated.

> | Argument | Description |
|---|---|
| find_minimum : *bool* | Indicates whether the optimimum of interest is a minimum or<br/> maximum. If true, looks for minimum. If false, looks for maximum. |
| max_iter : *int* | Maximum number of iterations. The algorithm will<br/> terminate after completing this many iterations. `None` indicates<br/> that the algorithm will not consider this. |
| max_function_evals : *int* | Maximum number of function evaluations. The algorithm will<br/> terminate after completing this many function evaluations. This<br/> is a preferable metric for greedy algorithms. `None` indicates that the algorithm will not consider this. |
| max_unchanged_iter : *int* | If the solution does not improve after this many iterations,<br/> the optimization terminates. `None` indicates that the algorithm<br/> will not consider this. |
| sol_threshold : *float* | If a solution is found better than this threshold, the iterations<br/> stop. `None` indicates that the algorithm will not consider this. |

---

### Example

```python
from optiseek.metaheuristics import simulated_annealing
from optiseek.testfunctions import booth

# define a list of variables and their domains for the objective function
var_list = [
	var_float('x1', [-10, 10]),
	var_float('x2', [-10, 10])
]	

# create an instance of the algorithm to optimize the booth test function and set its parameters
alg = simulated_annealing(booth, var_list)

# define stopping criteria and optimize
alg.optimize(find_minimum=True, max_iter=200, sol_threshold=0.05)

# show the results!
print(f'best_value = {alg.best_value:.5f}')
print(f'best_position = {alg.best_position}')
print(f'n_iter = {alg.completed_iter}')
```

```profile
best_value = 0.04947
best_position = {'x1': 0.9795801089700062, 'x2': 2.9176223437535667}
n_iter = 179
```

---

### References

[Simulated Annealing on Wikipedia](https://en.wikipedia.org/wiki/Simulated_annealing)