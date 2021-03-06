# Particle Swarm Optimizer

This class represents a standard particle swarm optimization algorithm, originally developed by Kennedy and Eberhart. 

This algorithm is based on swarm behavior commonly observed in nature. A population of particles is introduced to traverse the search space. 
Their movement is influenced by their own previous positions, the best known position of the swarm, and some stochastic velocity.

---

> *class* optiseek.metaheuristics.**particle_swarm_optimizer**(*input_function, b_lower=-10, b_upper=10, find_minimum=True, max_iter=100, sol_threshold=None, max_unchanged_iter=None, store_results=False, n_particles=50, weight=0.25, phi_p=1, phi_g=2, zero_velocity=True*)

---

### Parameters

| Parameter | Description |
|---|---|
| input_function : *function* | Function that the algorithm will use to search for an optimum.<br/> \*args will be passed to the function within the solver. |
| b_lower : *float, list of floats, or ndarray* | Contains the lower bounds of each dimension in the search <br/>  space. Can be a float if the function is one-dimensional. |
| b_upper : *float, list of floats, or ndarray* | Contains the upper bounds of each dimension in the search <br/>  space. Can be a float if the function is one-dimensional. |
| find_minimum : *bool* | Indicates whether the optimimum of interest is a minimum<br/> or maximum. If true, looks for minimum. If false,<br/> looks for maximum. |
| max_iter : *int* | Maximum number of iterations. If reached, the algorithm<br/> terminates. |
| sol_threshold : *float* | If a solution is found better than this threshold, the iterations<br/> stop. `None` indicates that the algorithm will not consider this. |
| max_unchanged_iter : *int* | If the solution does not improve after this many iterations,<br/> the solver terminates. `None` indicates that the algorithm<br/> will not consider this. |
| store_results : *bool* | Choose whether to save intermediate iteration results for<br/> post-processing or not. If true, results will be saved. |
| n_particles : *int* | Number of particles to use in the particle swarm population. |
| weight : *float* | Weight coefficient in [0, 1]. Lower weight gives the particles<br/> less momentum. |
| phi_p : *float* | Cognitive coefficient in [0, 3]. Higher value indicates that the<br/> particles are drawn more towards their own best known<br/> position. |
| phi_g : *float* | Social coefficient in [0, 3]. Higher value indicates that the<br/> particles are drawn more towards the swarm's collectively best<br/> known position. |
| zero_velocity : *bool* | Choose whether the particles start off with zero velocity or<br/> a random initial velocity. Initial velocities can sometimes<br/> cause divergence. |

---

### Attributes

| Attribute | Description |
|---|---|
| best_position : *ndarray* | Most optimal position found during the solution iterations. |
| best_value : *float* | Most optimal function value found during the solution iterations. |
| completed_iter : *int* | Number of iterations completed during the solution process. |
| stored_positions : *ndarray* | Positions for each member of the population for each iteration after<br/> the solver is finished. Set to `None` if user does not choose to store results.<br/> The results are placed in an array in the following format:<br/> `[iteration, population member, position in each dimension]` |
| stored_values : *ndarray* | Function values for each member of the population for each iteration.<br/> Set to `None` if user does not choose to store results. The results are<br/> placed in an array in the following format:<br/> `[iteration, population member, function value]` |

---

### Methods

```python
.solve()
```
	
Executes the algorithm solution with the current parameters. Results will be stored to the class attributes. If the user opted to store intermediate results, these will also be stored.

- Parameters
	- None
- Returns
	- None

---

### Example

```python
from optiseek.metaheuristics import particle_swarm_optimizer
from optiseek.testfunctions import booth

# create an instance of the algorithm, set its parameters, and solve
alg = particle_swarm_optimizer(booth)  # create instance with booth test function
alg.b_lower = [-10, -10] # define lower bounds
alg.b_upper = [10, 10] # define upper bounds
alg.max_iter = 100 # set iteration limit
alg.sol_threshold = 0.001 # set a solution threshold
alg.n_particles = 20 # define population size
alg.weight = 0.3 # set weight
alg.phi_p = 0.5 # set cognitive coefficient
alg.phi_g = 1.5 # set social coefficient

# execute the algorithm
alg.solve()

# show the results!
print(alg.best_value)
print(alg.best_position)
print(alg.completed_iter)
```

---

### References

[Particle Swarm Optimization on Wikipedia](https://en.wikipedia.org/wiki/Particle_swarm_optimization)