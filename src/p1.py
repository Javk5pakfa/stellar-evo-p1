from matplotlib.cbook import simple_linear_interpolation
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as pc
import scipy as sp


def planck_function(nu, T):
    '''
    Planck function definition.
    nu: Filter wavelength.
    T:  temperature.
    '''

    h = pc.Planck
    k = pc.Boltzmann
    c = pc.speed_of_light

    exponent = (h * c) / (nu * k * T)

    # B
    return (2.0 * h * c**2) / nu**5 * 1.0 / (np.e**exponent - 1.0)


def luminosity_mass_function(mass):
    '''
    Luminosity mass function. Tout (96) eqn (1).

    mass: Stellar mass in solar masses.
    '''

    a = [0.39704170,
         8.52762600,
         0.00025546,
         5.43288900,
         5.56357900,
         0.78866060,
         0.00586685]

    return (a[0]*mass**5.5 + a[1]*mass**11) / (a[2] + mass**3 + a[3]*mass**5 + a[4]*mass**7 + a[5]*mass**8 + a[6]*mass**9.5)


def radius_mass_function(mass):
    '''
    Radius mass function. Tout (96) eqn (2).

    mass: Stellar mass in solar masses.
    '''

    a = [1.71535900,
         6.59778800,
         10.08855000,
         1.01249500,
         0.07490166,
         0.01077422,
         3.08223400,
         17.84778000,
         0.00022582]

    return (a[0]*mass**2.5 + a[1]*mass**6.5 + a[2]*mass**11 + a[3]*mass**19 + a[4]*mass**19.5) / (a[5] + a[6]*mass**2 + a[7]*mass**8.5 + a[8]*mass**18.5 + a[9]*mass**19.5)


# The following functions come from the Inversion_Sampling.pdf document,
# provided by Ben Amend.


def salpeter_initial_mass_function(mass, alp=2.35):
    """
    Salpeter initial stellar mass function.

    max_mass: stellar mass upper bound.
    min_mass: stellar mass lower bound.
    alp:      weighting parameter.
    """

    return mass**(-alp)


def probability_density_function(mass, imf, max_mass, min_mass=0):
    """
    Normalized probability density function. Normalized to 1.

    imf:      Initial mass function to be normalized.
    max_mass: stellar mass upper bound.
    min_mass: stellar mass lower bound.
    """

    A = 1 / (simpsons_rule_integrate(imf, min_mass, max_mass, 100))

    return A * imf(mass)


def simpsons_rule_integrate(integrand, low=0, high=1, step=10):
    """
    Simpson's rule 1-D numerical integration method. It takes the target integrand, 
    which must be an integratable function defined elsewhere that returns a 
    numerical value, with integration bounds defined by low and high, and
    it integrates n = step - 1 times and produces an approximate result.

    integrand: function definition that represents an integratable,
               1 dimensional function.
    low:       lower bound of integration. Default is 0.
    high:      upper bound of integration. Default is 1.
    step:      number of steps for integration to take place. Default is 10 steps.
    TODO:      Add time-keeping feature to this method.
    """

    steps = np.linspace(low, high, num=step)
    result = 0.0
    j = 0

    # Scheme follows defintion in Scientific Computing textbook by Heath.
    for i in steps[1:]:
        a = steps[j]
        result += ((i - a) / 6) * (integrand(a) + 4 *
                                   integrand((a + i) / 2) + integrand(i))
        j += 1

    return result


def test_function(x, alp=1):
    return alp*x**2


if __name__ == '__main__':

    # Define ranges for plotting
    x_range = np.linspace(0.1, 100, 100)
    y_range = probability_density_function(
        x_range, salpeter_initial_mass_function, 100, 0.1)

    print(type(y_range))

    # Plotting schematics.
    fig, ax = plt.subplots()
    ax.scatter(x_range, y_range, color='green')
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.show()
