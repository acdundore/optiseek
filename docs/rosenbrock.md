# Rosenbrock's Function

This is a 2D function with a global minimum of zero at [1, 1].

Form of the function is as follows: 

*f(x, y) = (1 - x1)^2 + 5(x2 - x1^2)^2*

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

---

### References

[List of Test Functions on Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization)