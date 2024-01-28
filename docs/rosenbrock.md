# Rosenbrock Function

This is a 2D function with a global minimum of zero at [1, 1].

![Rosenbrock Function Plot](images/plot_rosenbrock.png)

Form of the function is as follows: 

![Rosenbrock Equation](images/equation_rosenbrock.png)

---

> *function* optiseek.testfunctions.**rosenbrock**(*x1, x2*)

---

### Parameters

| Parameter | Description |
|---|---|
| x1 : *float* | Input value for the first dimension. |
| x2 : *float* | Input value for the second dimension. |

---

### Example

```python
from optiseek.testfunctions import rosenbrock

y = rosenbrock(1, 1)
```

```compile
0
```

---

### References

[List of Test Functions on Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization)