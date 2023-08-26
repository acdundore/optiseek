# Ackley's Function (2-Dimensional)

This is a non-convex function with many local optima around a single global minimum of zero<br/> at [0, 0].

Form of the function is as follows: 

*f(x, y) = -20exp(-0.2sqrt(0.5(x1^2+x2^2))) - exp(0.5(cos(2πx1) + cos(2πx2))) + exp(1) + 20*

---

> *function* optiseek.testfunctions.**ackley2D**(*x1, x2*)

---

### Parameters

| Parameter | Description |
|---|---|
| x1 : *float* | Input value for the first dimension. |
| x2 : *float* | Input value for the second dimension. |

---

### Example

```python
from optiseek.testfunctions import ackley2D

y = ackley2D(0, 0)
```

---

### References

[List of Test Functions on Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization)