# Numerical schmemes testing playground.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.constants as cst


def test_function(x):
    return x**2


def test_numerical_scheme(integrand, low=0, high=1, step=10):
    steps = np.linspace(low, high, num=step)
    result = 0.0
    result_array = []
    j = 0

    # Scheme follows defintion in Scientific Computing textbook by Heath.
    for i in steps[1:]:
        a = steps[j]
        result += ((i - a) / 6) * (integrand(a) + 4 *
                                   integrand((a + i) / 2) + integrand(i))
        result_array.append(result)

        j += 1

    return result, step, result_array


def plotting_scheme(x_array, y_array):
    fig, ax = plt.subplots()
    ax.scatter(x_array, y_array, color='green')

    fig.show()


if __name__ == '__main__':
    result, step, num_array = test_numerical_scheme(test_function)
