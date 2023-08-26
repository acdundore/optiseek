# Wheeler's  Ridge

This is a 2D function with a global minimum in a deep valley. It is mostly smooth other than two ridges along each of the principal axes. These cause some algorithms to converge into local minima or diverge.
In this form, the minimum is at [1, 1.5] with a value of -1.

Form of the function is as follows: 

*f(x, y) = -exp(-(x1\*x2 - 1.5)^2 - (x2 - 1.5)^2)*

---

> *function* optiseek.testfunctions.**wheelers_ridge**(*x1, x2*)

---

### Parameters

| Parameter | Description |
|---|---|
| x1 : *float* | Input value for the first dimension. |
| x2 : *float* | Input value for the second dimension. |

---

### Example

```python
from optiseek.testfunctions import wheelers_ridge

y = wheelers_ridge(1, 1.5)
```

---

### References

[List of Test Functions on Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization)