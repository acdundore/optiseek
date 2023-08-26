# Tabu Search

This class represents the tabu search algorithm developed by Glover.

This is a local search algorithm that utilizes memory of past positions in order to avoid getting stuck in local optima. The algorithm finds a new neighbor at each iteration;
if the neighbor matches a previous position on the "tabu list" (within a specified threshold), then that position is prohibited unless it is better than the currently best known position.
The tabu list consists of positions from a specified number of previous iterations. The length of the tabu list and tabu threshold can both be tuned.

---

> *class* optiseek.metaheuristics.**tabu_search**(*input_function, b_lower=-10, b_upper=10, find_minimum=True, max_iter=100, sol_threshold=None, max_unchanged_iter=None, sigma_coeff=0.2, neighbor_dim_changes=1, initial_guess=None, store_results=False, tenure=5, n_candidates=5, neighbor_tolerance=0.02*)

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
| sigma_coeff : *float* | Coefficient in (0, 0.5] to be multiplied by the bound widths<br/> for each dimension; the corresponding number is used for<br/> the standard deviation in the neighbor generation process. |
| neighbor_dim_changes : *int* | Number of dimensions to mutate during the generation of<br/> a new neighbor position. Must be in [1, number of dimensions] |
| initial_guess : *list of floats or ndarray* | Initial guess used in the solution process. Leave as `None` to<br/> start with a random initial guess. |
| store_results : *bool* | Choose whether to save intermediate iteration results for<br/> post-processing or not. If true, results will be saved. |
| tenure : *int* | Number of previous positions stored on the tabu list. These<br/> positions (within a specified tolerance) will be prohibited<br/> in following iterations. |
| n_candidates : *int* | Number of new candidate solutions to guess at each<br/> iteration. The best solution that is not tabu is used. |
| neighbor_tolerance : *float* | Portion of dimension width to use as a tolerance when<br/> determining whether a potential position is tabu. |

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
from optiseek.metaheuristics import tabu_search
from optiseek.testfunctions import booth

# create an instance of the algorithm, set its parameters, and solve
alg = tabu_search(booth)  # create instance with booth test function
alg.b_lower = [-10, -10] # define lower bounds
alg.b_upper = [10, 10] # define upper bounds
alg.max_iter = 100 # set iteration limit
alg.sol_threshold = 0.001 # set a solution threshold
alg.sigma_coeff = 0.02 # set a multiplier of bound widths for std. dev.
alg.neighbor_dim_changes = 1 # only mutate 1 dimension at a time for neighbors
alg.initial_guess = [2, 5] # set an initial guess of the optimum
alg.tenure = 7 # set the length of the tabu list to 7 previous positions
alg.n_candidates = 4 # generate four neighbors during each iteration and pick best
alg.neighbor_tolerance = 0.015 # fraction of bound widths to use for tolerance

# execute the algorithm
alg.solve()

# show the results!
print(alg.best_value)
print(alg.best_position)
print(alg.completed_iter)
```

---

### References

[Tabu Search on Wikipedia](https://en.wikipedia.org/wiki/Tabu_search)