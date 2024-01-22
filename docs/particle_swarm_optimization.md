# Particle Swarm Optimizer

This class represents a standard particle swarm optimization algorithm, originally developed by Kennedy and Eberhart. 

This algorithm is based on swarm behavior commonly observed in nature. 
A population of particles is introduced to traverse the search space. 
Their movement is influenced by their own previous positions, the best known position of the swarm, and some stochastic velocity.

---

> *class* optiseek.metaheuristics.**particle_swarm_optimizer**(*input_function=None, var_list=None, linspaced_initial_positions=True, results_filename=None, n_particles=None, weight=0.35, phi_p=1.5, phi_g=1.5, zero_velocity=False*)

---

### Parameters

All parameters are also class attributes and may be modified after instantiation.

| Parameter | Description |
|---|---|
| input_function : *function* | Function that the algorithm will use to search for an optimum.<br/> \*args will be passed to the function within the solver. |
| var_list : *list of variables* | List of variables (see variable types) to define the search space.<br/> These correspond to the arguments of the objective function<br/> and must be in the exact same order. |
| linspaced_initial_positions : *bool* | If true, creates a linearly spaced set of points in each search<br/> dimension, and the initial positions of the population are set to<br/> mutually exclusive combinations of these points. This guarantees<br/> that there will be no empty spots in a single dimension. If false,<br/> random initial positions are chosen. |
| results_filename : *string* | If a file name is passed (ending in '.csv'), the results will be written<br/> to this file after each function evaluation. This can noticeably slow<br/> down solution iterations for quick objective functions. For greedy<br/> functions, it can be beneficial to do this in case the script is<br/> interrupted. |
| n_particles : *int* | Number of particles to use in the particle swarm population. If<br/> set to `None`, the population size will be based on the heuristic<br/> 10 + 2 \* sqrt(n_dims), where n_dims is the dimensionality of<br/> the search space. This is typically sufficient to explore the<br/> whole search space. |
| weight : *float* | Weight coefficient in [0, 1]. Lower weight gives the particles<br/> less momentum. |
| phi_p : *float* | Cognitive coefficient in [0, 3]. Higher value indicates that the<br/> particles are drawn more towards their own best known<br/> position. |
| phi_g : *float* | Social coefficient in [0, 3]. Higher value indicates that the<br/> particles are drawn more towards the swarm's collectively best<br/> known position. |
| zero_velocity : *bool* | Choose whether the particles start off with zero velocity or<br/> a random initial velocity. |

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
alg.optimize(find_minimum=True, max_iter=10, sol_threshold=0.05)

# show the results!
print(f'best_value = {alg.best_value:.5f}')
print(f'best_position = {alg.best_position}')
print(f'n_iter = {alg.completed_iter}')
```

```profile
best_value = 0.04470
best_position = {'x1': 1.050827859446013, 'x2': 2.86984154026157}
n_iter = 8
```

---

### References

[Particle Swarm Optimization on Wikipedia](https://en.wikipedia.org/wiki/Particle_swarm_optimization)