# Firefly Algorithm

This class represents the firefly algorithm developed by Xin-She Yang. 

This algorithm is based on the flashing patterns and swarm behavior of fireflies. Fireflies are attracted to others based on their proximity in the search space and
the brightness (function values) of others. Their movements also have a stochastic component.

---

> *class* optiseek.metaheuristics.**firefly_algorithm**(*input_function, b_lower=-10, b_upper=10, find_minimum=True, max_iter=100, sol_threshold=None, max_unchanged_iter=None, store_results=False, n_fireflies=50, beta=1.0, alpha=0.01, gamma=1.0*)

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
| n_fireflies : *int* | Number of fireflies to use in the swarm population. |
| beta : *float* | Linear visibility coefficient in [0.1, 1.5]. Lower value indicates<br/> that the fireflies are less attracted to each other. |
| alpha : *float* | Coefficient in [0, 0.1] that is a portion of each dimension's<br/>  bound widths to use for the random walk. |
| gamma : *float* | Exponential visibility coefficient in [0.01, 1]. Higher value<br/> indicates that the fireflies are less attracted to each other. |

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
from optiseek.metaheuristics import firefly_algorithm
from optiseek.testfunctions import booth

# create an instance of the algorithm, set its parameters, and solve
alg = firefly_algorithm(booth)  # create instance with booth test function
alg.b_lower = [-10, -10] # define lower bounds
alg.b_upper = [10, 10] # define upper bounds
alg.max_iter = 100 # set iteration limit
alg.sol_threshold = 0.001 # set a solution threshold
alg.n_fireflies = 20 # define population size
alg.beta = 0.3 # set linear visibility coefficient
alg.alpha = 0.05 # set random walk coefficient
alg.gamma = 1.5 # set exponential visibility coefficient

# execute the algorithm
alg.solve()

# show the results!
print(alg.best_value)
print(alg.best_position)
print(alg.completed_iter)
```

---

### References

[Firefly Algorithm on Wikipedia](https://en.wikipedia.org/wiki/Firefly_algorithm)