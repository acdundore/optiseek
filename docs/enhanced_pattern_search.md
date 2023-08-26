# Enhanced Pattern Search

This class represents an enhanced Hooke-Jeeves pattern search algorithm. 

This is a basic black-box optimization function that requires no knowledge of the form of the function to be optimized. It is very similar to the standard Hooke-Jeeves Pattern Search algorithm 
with a slight variation. Rather than taking two sample points in each dimension for each iteration, a single sample point is taken for each dimension in the positive direction with the magnitude of the step size.
Then, a final sample point is taken in the negative direction of the step size in all dimensions at once. This results in a positive spanning set of search directions, with only n+1 sample points taken at
each iteration. This can be more efficient than the basic Hooke-Jeeves method.

---

> *class* optiseek.metaheuristics.**enhanced_pattern_search**(*input_function, initial_guess, find_minimum=True, max_iter=100, sol_threshold=None, store_results=False, max_step_size=1.0*)

---

### Parameters

| Parameter | Description |
|---|---|
| input_function : *function* | Function that the algorithm will use to search for an<br/> optimum. \*args will be passed to the function within<br/> the solver. |
| initial_guess : *float, list of floats, or ndarray* | The initial guess that the algorithm will start the<br/> search from. Can be a float if the function is<br/> one-dimensional. |
| find_minimum : *bool* | Indicates whether the optimimum of interest is a minimum<br/> or maximum. If true, looks for minimum. If false,<br/> looks for maximum. |
| max_iter : *int* | Maximum number of iterations. If reached, the algorithm<br/> terminates. |
| sol_threshold : *float* | If a solution is found better than this threshold, the<br/> iterations stop. `None` indicates that the algorithm will<br/> not consider this. |
| store_results : *bool* | Choose whether to save intermediate iteration results for<br/> post-processing or not. If true, results will be saved. |
| max_step_size : *float* | Maximum step size that the algorithm can possibly take<br/> for each iteration in each direction. |

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
from optiseek.direct import enhanced_pattern_search
from optiseek.testfunctions import booth

# create an instance of the algorithm, set its parameters, and solve
alg = enhanced_pattern_search(booth, [0, 0])  # create instance with booth test function and initial guess [0, 0]
alg.max_iter = 100 # set iteration limit
alg.sol_threshold = 0.001 # set a solution threshold
alg.max_step_size = 0.5 # define maximum step size

# execute the algorithm
alg.solve()

# show the results!
print(alg.best_value)
print(alg.best_position)
print(alg.completed_iter)
```

---

### References

[Hooke-Jeeves Pattern Search on Wikipedia](https://en.wikipedia.org/wiki/Pattern_search_(optimization))