# Parameter Grid Search

This class is a utility to help the user tune their optimization algorithms. This is accomplished by creating a grid of parameter permutations based on user input
and running a selected algorithm with those parameters to find the best set. The final results of every permutation of parameters are saved for post-processing.

---

> *class* optiseek.modelhelpers.**parameter_grid_search**(*algorithm, input_function, param_options, show_progress=False*)

---

### Parameters

| Parameter | Description |
|---|---|
| algorithm : *class* | Class of the algorithm that you would like to use.|
| input_function : *function* | Function object for the algorithm to optimize.|
| param_options : *dict* | Dictionary containing the grid of parameters to be explored<br/>with the parameter names (strings) as keys and a list of<br/>parameter values as values. All permutations of parameters<br/>in this dict will be tested. For inputs that would normally be<br/>in a list (like the search bounds on a 2+ dimensional function,<br/>for example), place that list inside another list.<br/>For any parameters not specified, the default will be used.<br/>See the example for more details.|
| show_progress : *bool* | Boolean to indicate whether the grid search will print progress<br/> to the console as the solve continues. The number of permutations<br/>increases exponentially with respect to parameter inputs, so for high<br/>numbers of parameter inputs, this can be useful to see how much<br/>longer the solver has left. |

---

### Attributes

| Attribute | Description |
|---|---|
| param_permutations : *list of dicts* | A list of dictionaries that represent the parameter<br/>permutations used for each iteration of the grid search.<br/>In the dicts, the parameter name as a string is the key<br/>and the parameter value is the value. |
| permutation_positions : *list of ndarrays* | A list containing the optimal positions found for each<br/>permutation of parameters. This corresponds with the<br/>permutations in the param_permutations attribute. |
| permutation_values : *list of floats* | A list containing the optimal values found for each<br/>permutation of parameters. This corresponds with the<br/>permutations in the param_permutations attribute. |
| best_parameters : *dict* | A dictionary containing the best performing set of<br/>parameters. The parameter names as strings are stored as<br/>keys and the corresponding values are stored as values. |
| best_position : *list or ndarray* | The most optimal position that was found using the<br/>best performing parameters. |
| best_value : *float* | The most optimal function value that was found using<br/>the best performing parameters. |

---

### Methods

```python
.solve()
```
	
Executes the parameter grid search process and stores the results in the class attributes.

- Parameters
	- None
- Returns
	- None

---

### Example

```python
from optiseek.modelhelpers import parameter_grid_search
from optiseek.metaheuristics import simulated_annealing
from optiseek.testfunctions import ackley2D

# set up the param_options dictionary
param_options = {
    "initial_guess": [[8, 4]],
    "b_lower": [[-10, -10]],
    "b_upper": [[10, 10]],
    "sigma_coeff": [0.05, 0.1, 0.2, 0.3],
    "max_iter": [50, 100, 500],
    "start_temperature": [20, 10, 5, 2],
    "alpha": [0.85, 0.93, 0.99],
    "neighbor_dim_changes": [1, 2]
}

# create the an instance of the grid search class and execute the solve method
pgs = parameter_grid_search(simulated_annealing, ackley2D, param_options, show_progress=False)
pgs.solve()

# show all permutations and their associated optimal positions and values found
for grid in range(len(pgs.param_permutations)):
    print(pgs.param_permutations[i])

print(pgs.permutation_positions)
print(pgs.permutation_values)

# print the best parameter permutation found and the associated optimal position and value
print(pgs.best_parameters)
print(pgs.best_position)
print(pgs.best_value)
```