# Release Notes

## v1.0.0 (2024.01.31)

### Features

- inputs for objective functions now support float, integer, categorical, and boolean variables
- the algorithms now scale search domain dimensions behind the scenes to ensure they are all the same (no discrepancies between dimension bound sizes)
- results are now stored in a Pandas DataFrame
- hard search bounds have been implemented; the search can no longer go out of bounds, and algorithms with velocity will "bounce" off of the boundaries
- linearly spaced initial position sampling has been introduced for population-based algorithms; this guarantees a wider search spread for the initial iteration
- an option to save results after each function evaluation was added; this is particularly helpful for expensive objective functions, in case an exception occurs during optimization
- rules of thumb have been implemented for selection of default population size for population-based algorithms
- maximum function evaluations has been added as a stopping criterion, which is useful for computationally expensive objective functions
- the .solve() method has been changed to .optimize()
- former attributes of the objective functions have been modified to become arguments of the .optimize() method
- log sampling/scaling is now supported for float and integer variable types
- general documentation updates, including additional examples (constrained optimization, tuning ML hyperparameters, etc.)

## v0.2.0 (2022.07.23)

### Features

- New `modelhelpers` module, containing parameter grid search class and penalty constraints function
- Flying Foxes Optimization Algorithm added

## v0.1.0 (2022.05.30)

- Original Release