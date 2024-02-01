# Parameter Grid Search
 
The purpose of the grid search is to provide an easy way to tune parameters for a given algorithm in a way that will work well on a certain class of problem.
This is accomplished by creating a grid of parameter permutations based on user input and running a selected algorithm with those parameters to find the best set. 
The best results of every permutation of parameters are saved for post-processing.

---

> *class* optiseek.modelhelpers.**parameter_grid_search**(*algorithm, input_function, var_list, param_grid, optimize_options, show_progress=False*)

---

### Parameters

| Parameter | Description |
|---|---|
| algorithm : *class* | Class of the algorithm that you would like to use.|
| input_function : *function* | Function object for the algorithm to optimize.|
| var_list : *list of variables* | List of all variable objects to define their names and domains in the<br/> search space. This is the same list you would pass to the optimization<br/> algorithm class.|
| param_grid : *dict* | Dictionary containing the grid of parameters to be explored<br/>with the parameter names (strings) as keys and a list of<br/>parameter values as values. All permutations of parameters<br/>in this dict will be tested. For any parameters not specified, the default<br/> will be used. See the example for more details.|
| optimize_options : *dict* | Dictionary containing the kwargs for the optimize() method of the<br/> algorithm.|
| show_progress : *bool* | Boolean to indicate whether the grid search will print progress<br/> to the console as the solve continues. The number of permutations<br/>increases exponentially with respect to parameter inputs, so for high<br/>numbers of parameter inputs, this can be useful to see how much<br/>longer the solver has left. |

---

### Attributes

| Attribute | Description |
|---|---|
| best_parameters : *dict* | A dictionary containing the best performing set of<br/>parameters. The parameter names as strings are stored as<br/>keys and the corresponding values are stored as values. |
| best_position : *list or ndarray* | The most optimal position that was found using the<br/>best performing parameters. |
| best_value : *float* | The most optimal function value that was found using<br/>the best performing parameters. |
| results : *pd.DataFrame* | A pandas DataFrame containing all results from the search. Columns<br/> represent the algorithm parameters, best position found, and best<br/> function value found with the respective parameters. |

---

### Methods

**.search**()
	
> Executes the parameter grid search process and stores the results in the class attributes.

---

### Example

```python
from optiseek.variables import var_float
from optiseek.modelhelpers import parameter_grid_search
from optiseek.metaheuristics import particle_swarm_optimizer
from optiseek.testfunctions import ackley

# define the variables for the Ackley2D function
var_list = [
    var_float('x', [-10, 10]),
    var_float('y', [-10, 10])
]

# set up the param_grid dictionary
param_grid = {
    'n_particles': [10],
    'weight': [0.20, 0.35, 0.50],
    'phi_p': [1.0, 1.5, 2.0],
    'phi_g': [1.0, 1.5, 2.0],
    'zero_velocity': [True, False]
}

# set up the optimize_options dictionary
optimize_options = {
    'find_minimum': True,
    'max_function_evals': 75
}

# create the an instance of the grid search class
pgs = parameter_grid_search(particle_swarm_optimizer, 
							ackley, 
							var_list, 
							param_grid, 
							optimize_options)

# start the search
pgs.search()

# show the optimal parameters, the best function value found, and a preview of all saved results
print(f'best parameters: {pgs.best_parameters}')
print(f'best value: {pgs.best_value}\n')
print(pgs.results.head())
```

```profile
best parameters: {'n_particles': 10, 'weight': 0.2, 'phi_p': 1.5, 'phi_g': 1.0, 'zero_velocity': False}
best value: 0.026961543641856878

   n_particles  weight  phi_p  ...         x         y  best_value
0           10    0.20    1.5  ...  0.007717  0.004236    0.026962
1           10    0.50    1.0  ...  0.021830 -0.003103    0.075259
2           10    0.20    2.0  ...  0.019337 -0.018130    0.093595
3           10    0.35    1.0  ... -0.028123  0.002833    0.101083
4           10    0.20    1.5  ...  0.001063  0.031908    0.117212

[5 rows x 8 columns]
```