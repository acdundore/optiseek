# Mayfly Algorithm

This class represents the mayfly algorithm developed by Zervoudakis and Tsafarakis. 

This algorithm takes components from swarm-based algorithms as well as genetic algorithms and combines them into a powerful hybrid algorithm based on the mating behavior of mayflies.
An initial population is split into males and females, each moving in different ways. The males exhibit swarm behavior to gather towards the best male (at the best function value),
similar to particle swarm optimization. The females are attracted to a matched male if the male has a better function value. In each iteration, there is a genetic crossover
between the males and females and selection of the best in the population takes place. Stochastic components are introduced into the movements to avoid local optima. This is a very
powerful algorithm, but requires many parameters.

---

> *class* optiseek.metaheuristics.**mayfly_algorithm**(*input_function, b_lower=-10, b_upper=10, find_minimum=True, max_iter=100, sol_threshold=None, max_unchanged_iter=None, store_results=False, n_mayflies=50, beta=0.7, gravity=0.6, alpha_cog=0.5, alpha_soc=1.5, alpha_attract=1.5, nuptial_coeff=0.05*)

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
| n_mayflies : *int* | Number of mayflies to use in the population. |
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
from optiseek.metaheuristics import mayfly_algorithm
from optiseek.testfunctions import booth

# create an instance of the algorithm, set its parameters, and solve
alg = mayfly_algorithm(booth)  # create instance with booth test function
alg.b_lower = [-10, -10] # define lower bounds
alg.b_upper = [10, 10] # define upper bounds
alg.max_iter = 100 # set iteration limit
alg.sol_threshold = 0.001 # set a solution threshold
alg.n_mayflies = 20 # set mayfly population
alg.beta = 0.5 # set visibility coefficient
alg.gravity = 0.4 # set gravity coefficient
alg.alpha_cog = 0.5 # set male cognitive coefficient
alg.alpha_soc = 1.5 # set male social coefficient
alg.alpha_attract = 1.0 # set female attraction coefficient
alg.nuptial_coeff = 0.02 # set random walk coefficient

# execute the algorithm
alg.solve()

# show the results!
print(alg.best_value)
print(alg.best_position)
print(alg.completed_iter)
```

---

### References

[A mayfly optimization algorithm, by Konstantinos and Tsafarakis](https://www.sciencedirect.com/science/article/abs/pii/S036083522030293X)