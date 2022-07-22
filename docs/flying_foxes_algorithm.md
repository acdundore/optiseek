# Flying Foxes Algorithm

This class represents the flying foxes optimization algorithm developed by Zervoudakis and Tsafarakis. 

This algorithm is a powerful and efficient metaheuristic that takes inspiration from the group behavior of flying foxes during a heatwave. It also contains traits of other common
algorithms like genetic algorithms, which are utilized during the creation of new foxes. Foxes near the coolest spot are encouraged to explore nearby areas, preserving the breadth
of the search area. If the most optimal spot currently known gets too crowded, the foxes will die off and produce new ones; this is similar to the overheating that occurs in nature
when they crowd around cool areas during a heatwave. This algorithm is unique in the fact that it requires no user input for parameters; instead, a fuzzy self-tuning technique is
utilized to tune the algorithm parameters for each individual fox at the beginning of every iteration. This makes the algorithm simple to deploy even by inexperienced users.
It also outperforms most population-based metaheuristics in many engineering problems.

---

> *class* optiseek.metaheuristics.**flying_foxes_algorithm**(*input_function, b_lower=-10, b_upper=10, find_minimum=True, max_iter=100, sol_threshold=None, max_unchanged_iter=None, store_results=False*)

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
from optiseek.metaheuristics import flying_foxes_algorithm
from optiseek.testfunctions import booth

# create an instance of the algorithm, set its parameters, and solve
alg = flying_foxes_algorithm(booth)  # create instance with booth test function
alg.b_lower = [-10, -10] # define lower bounds
alg.b_upper = [10, 10] # define upper bounds
alg.max_iter = 100 # set iteration limit
alg.sol_threshold = 0.001 # set a solution threshold

# execute the algorithm
alg.solve()

# show the results!
print(alg.best_value)
print(alg.best_position)
print(alg.completed_iter)
```

---

### References

[A global optimizer inspired from the survival strategies of flying foxes, by Konstantinos and Tsafarakis](https://link.springer.com/article/10.1007/s00366-021-01554-w)