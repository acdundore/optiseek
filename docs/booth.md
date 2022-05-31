# Booth's Function

This is a simple 2D quadratic function with a minimum of zero at [1, 3].

Form of the function is as follows: 

*f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2*

---

> *function* optiseek.testfunctions.**booth**(*x1, x2*)

---

### Parameters

| Parameter | Description |
|---|---|
| x1 : *float* | Input value for the first dimension. |
| x2 : *float* | Input value for the second dimension. |

---

### Example

```python
from optiseek.testfunctions import booth

y = booth(1, 3)
```

---

### References

[List of Test Functions on Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization)