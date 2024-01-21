# Mayfly Algorithm

This class represents the mayfly algorithm developed by Zervoudakis and Tsafarakis. 

This algorithm takes components from swarm-based algorithms as well as genetic algorithms and combines them into a powerful hybrid algorithm based on the mating behavior of mayflies.
An initial population is split into males and females, each moving in different ways. The males exhibit swarm behavior to gather towards the best male (at the best function value),
similar to particle swarm optimization. The females are attracted to a matched male if the male has a better function value. In each iteration, there is a genetic crossover
between the males and females and selection of the best in the population takes place. Stochastic components are introduced into the movements to avoid local optima.

---

> *class* optiseek.metaheuristics.**mayfly_algorithm**(*input_function, var_list, linspaced_initial_positions=True, results_filename=None, n_mayflies=None, beta=0.7, gravity=0.6, alpha_cog=0.5, alpha_soc=1.5, alpha_attract=1.5, nuptial_coeff=0.05*)

---

### Parameters

All parameters are also class attributes and may be modified after instantiation.

| Parameter | Description |
|---|---|
| input_function : *function* | Function that the algorithm will use to search for an optimum.<br/> \*args will be passed to the function within the solver. |
| var_list : *list of variables* | List of variables (see variable types) to define the search space.<br/> These correspond to the arguments of the objective function<br/> and must be in the exact same order. |
| linspaced_initial_positions : *bool* | If true, creates a linearly spaced set of points in each search<br/> dimension, and the initial positions of the population are set to<br/> mutually exclusive combinations of these points. This guarantees<br/> that there will be no empty spots in a single dimension. If false,<br/> random initial positions are chosen. |
| results_filename : *string* | If a file name is passed (ending in '.csv'), the results will be written<br/> to this file after each function evaluation. This can noticeably slow<br/> down solution iterations for quick objective functions. For greedy<br/> functions, it can be beneficial to do this in case the script is<br/> interrupted. |
| n_mayflies : *int* | Number of mayflies to use in the population.  If<br/> set to `None`, the population size will be based on the heuristic<br/> 10 + 2 \* sqrt(n_dims), where n_dims is the dimensionality of<br/> the search space. This is typically sufficient to explore the<br/> whole search space. |
| beta : *float* | Exponential visibility coefficient in [0.1, 1]. Higher value<br/> means that mayflies are less drawn towards others. |
| gravity : *float* | Gravity coefficient in [0.1, 1]. Lower value means that the<br/> mayflies have less momentum. |
| alpha_cog : *float* | Cognitive coefficient in [0, 2]. Indicates how attracted the male<br/> mayflies are to their individually best known position. |
| alpha_soc : *float* | Social coefficient in [0, 2]. Indicates how attracted the male<br/> mayflies are to the male swarm's best known position. |
| alpha_attract : *float* | Attraction coefficient in [0, 2]. Indicates how attracted the<br/> females are to their matched male counterpart. |
| nuptial_coeff : *float* | Nuptial coefficient in [0, 0.4], a multiplier on bound widths<br/> for each dimension used for the male and female random<br/> walks. |


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
from optiseek.metaheuristics import mayfly_algorithm
from optiseek.testfunctions import booth

# define a list of variables and their domains for the objective function
var_list = [
	var_float('x1', [-10, 10]),
	var_float('x2', [-10, 10])
]	

# create an instance of the algorithm to optimize the booth test function and set its parameters
alg = mayfly_algorithm(booth, var_list)

# define stopping criteria and optimize
alg.optimize(find_minimum=True, max_iter=10, sol_threshold=0.05)

# show the results!
print(f'best_value = {alg.best_value:.5f}')
print(f'best_position = {alg.best_position}')
print(f'n_iter = {alg.completed_iter}')
```

```profile
best_value = 0.04024
best_position = {'x1': 1.000108782682335, 'x2': 3.0896223872777275}
n_iter = 9
```

---

### References

[A mayfly optimization algorithm, by Zervoudakis and Tsafarakis](https://www.sciencedirect.com/science/article/abs/pii/S036083522030293X)