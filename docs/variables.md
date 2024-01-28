# Variable Types

Algorithms in `optiseek` can accept several variable types in addition to just continuous values.
Internally, the optimizers treat all variables as continuous numerical dimensions in the search space;
however, they are converted back to the appropriate user-specified format when passed to the objective function.
Currently, supported variable types are continuous (float), integer, categorical/ordinal, and boolean.

Each variable type has its own class. 
When defining variables and search space bounds for an algorithm, the user passes a list of variable classes into the `var_list` argument of the algorithm class.

---

> Floats/Continous  
> *class* optiseek.variables.**var_float**(*var_name, bounds*)

> Integers  
> *class* optiseek.variables.**var_int**(*var_name, bounds*)

> Categorical   
> *class* optiseek.variables.**var_categorical**(*var_name, choices*)

> Boolean (True/False)   
> *class* optiseek.variables.**var_bool**(*var_name*)

---

### Parameters

| Parameter | Description |
|---|---|
| var_name : *string* | Name of the variable. This will be used to track output and label the<br/> stored results if applicable. |
| bounds : *list of floats/int* | A list containing a lower and upper bound for the search space of the<br/> variable in the format *[lower, upper]*. The values can be integers/floats<br/> for `var_float` and must be integers for `var_int`. |
| choices : *list* | Contains a list of choices to be passed as an argument for the objective<br/> function. The list items may be any type that the objective function<br/> can accept. |

---

### Example

```python
# defining a variable list to be passed to an algorithm
var_list = [
    var_float('x_float', [-10.5, 10.5]),
    var_int('x_int', [-2, 5]),
    var_categorical('x_cat', ['small', 'medium', 'large']),
    var_bool('x_bool')
]
```