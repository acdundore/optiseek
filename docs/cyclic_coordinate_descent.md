# Cyclic Coordinate Descent

This class represents a cyclic coordinate descent algorithm. 

This is a basic black-box optimization function that requires no knowledge of the form of the function to be optimized. The algorithm cycles through each of the dimensions in sequence
and does an individual line search (a golden section search) within the maximum step size specified by the user. While the line search is executed in a certain dimension, the position values
in all other dimensions are held constant. This is a deterministic method that is susceptible to getting stuck in local optima. In some cases, the algorithm gets stuck in a loop before it
even reaches a local optimum. In these cases, changing the initial guess can rectify the issue.

---

> *class* optiseek.metaheuristics.**cyclic_coordinate_descent**(*input_function, initial_guess, find_minimum=True, max_iter=100, sol_threshold=None, store_results=False, max_step_size=1.0*)

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
from optiseek.direct import cyclic_coordinate_descent
from optiseek.testfunctions import booth

# create an instance of the algorithm, set its parameters, and solve
alg = cyclic_coordinate_descent(booth, [0, 0])  # create instance with booth test function and initial guess [0, 0]
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

[Coordinate Descent on Wikipedia](https://en.wikipedia.org/wiki/Coordinate_descent)