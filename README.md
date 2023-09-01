![optiseek logo](docs/images/optiseek_logo_small.png)

[![Downloads](https://static.pepy.tech/personalized-badge/optiseek?period=total&units=none&left_color=black&right_color=red&left_text=Downloads)](https://pepy.tech/project/optiseek)

An open source collection of single-objective optimization algorithms for multi-dimensional functions.

The purpose of this library is to give users access to a variety of optimization algorithms with extreme ease of use and interoperability.
The parameters of each of the algorithms can be tuned by the users and there is a high level of input uniformity between algorithms of similar type.

## Installation

```bash
pip install optiseek
```

## Usage

`optiseek` provides access to numerous optimization algorithms that require minimal effort from the user. An example using the well-known particle swarm optimization algorithm can be as simple as this:

```python
from optiseek.metaheuristics import particle_swarm_optimizer
from optiseek.testfunctions import booth

# create an instance of the algorithm, set its parameters, and solve
my_algorithm = particle_swarm_optimizer(booth)  # create instance to optimize the booth function
my_algorithm.b_lower = [-10, -10] # define lower bounds
my_algorithm.b_upper = [10, 10] # define upper bounds
my_algorithm.n_particles = 14 # define number of particles
my_algorithm.max_iter = 20 # define iteration limit


# execute the algorithm
my_algorithm.solve()

# show the results!
print(my_algorithm.best_value)
print(my_algorithm.best_position)
print(my_algorithm.completed_iter)
```

This is a fairly basic example implementation without much thought put into parameter selection. Of course, the user is free to tune the parameters of the algorithm any way they would like.

## Documentation

For full documentation, visit the [github pages site](https://acdundore.github.io/optiseek/).

## License

`optiseek` was created by Alex Dundore. It is licensed under the terms of the MIT license.

## Credits and Dependencies

`optiseek` is powered by [`numpy`](https://numpy.org/).