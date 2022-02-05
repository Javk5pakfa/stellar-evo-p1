import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import random

alp = 2.35
solar_lum = 3.83e26  # Watts
solar_rad = 6.957e8  # m
zero_lum = 3.018e28  # Watts


def test_function(x, alp=1):
    return alp * x ** 2


def plotting_scheme_loglog(x_range,
                           y_range,
                           options=None,
                           x_axis='log x',
                           y_axis='log y',
                           title='Title',
                           invert_x=False,
                           invert_y=False):
    """
    Takes x_range and y_range and plot Y-range to X_range.

    options: a dictionary of settings.
    """

    # Plotting schematics.
    if options is None:
        options = {}
    fig, ax = plt.subplots()
    ax.scatter(x_range, y_range, **options)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    if invert_x is True:
        ax.invert_xaxis()
    if invert_y is True:
        ax.invert_yaxis()
    plt.title(title)
    plt.show()


def plotting_scheme(x_range,
                    y_range,
                    options=None,
                    x_axis='x',
                    y_axis='y',
                    title='Title',
                    invert_x=False,
                    invert_y=False):
    """
    Takes x_range and y_range and plot Y-range to X_range.

    options: a dictionary of settings.
    """

    # Plotting schematics.
    if options is None:
        options = {}
    fig, ax = plt.subplots()
    ax.scatter(x_range, y_range, **options)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    if invert_x is True:
        ax.invert_xaxis()
    if invert_y is True:
        ax.invert_yaxis()
    plt.title(title)
    plt.show()


def luminosity_mass_function(mass):
    """
    Luminosity mass function. Tout (96) eqn (1).

    mass: Stellar mass in solar masses.
    returns: luminosity in solar masses.
    """

    a = [0.39704170,
         8.52762600,
         0.00025546,
         5.43288900,
         5.56357900,
         0.78866060,
         0.00586685]

    return (a[0] * mass ** 5.5 + a[1] * mass ** 11) / (
            a[2] + mass ** 3 + a[3] * mass ** 5 + a[4] * mass ** 7 +
            a[5] * mass ** 8 + a[6] * mass ** 9.5)


def radius_mass_function(mass):
    """
    Radius mass function. Tout (96) eqn (2).

    mass: Stellar mass in solar masses.
    """

    a = [1.71535900,
         6.59778800,
         10.08855000,
         1.01249500,
         0.07490166,
         0.01077422,
         3.08223400,
         17.84778000,
         0.00022582]

    return (a[0] * mass ** 2.5 + a[1] * mass ** 6.5 + a[2] * mass ** 11 +
            a[3] * mass ** 19 + a[4] * mass ** 19.5) / (
                   a[5] + a[6] * mass ** 2 + a[7] * mass ** 8.5 + mass ** 18.5 +
                   a[8] * mass ** 19.5)


def simpsons_rule_integrate(integrand, low=0, high=1, step=10):
    """
    Simpson's rule 1-D numerical integration method. It takes the target
    integrand, which must be an integrable function defined elsewhere that
    returns a numerical value, with integration bounds defined by low and high,
    and it integrates n = step - 1 times and produces an approximate result.

    integrand: function definition that represents an integrable,
               1 dimensional function.
    low:       lower bound of integration. Default is 0.
    high:      upper bound of integration. Default is 1.
    step:      number of steps for integration to take place. Default is
               10 steps.
    TODO:      Add time-keeping feature to this method.
    """

    steps = np.linspace(low, high, num=step)
    result = 0.0
    j = 0

    # Scheme follows definition in Scientific Computing textbook by Heath.
    for i in steps[1:]:
        a = steps[j]
        result += ((i - a) / 6) * (integrand(a) + 4 *
                                   integrand((a + i) / 2) + integrand(i))
        j += 1

    return result


# ------------------------------------------------------------------------------
# The following functions come from the Inversion_Sampling.pdf document,
# provided by Ben Amend.
# ------------------------------------------------------------------------------

def salpeter_initial_mass_function(mass, alpha=alp):
    """
    Salpeter initial stellar mass function.

    max_mass: stellar mass upper bound.
    min_mass: stellar mass lower bound.
    alp:      weighting parameter.
    """

    return mass ** (-alpha)


def probability_density_function(mass, imf, max_mass, min_mass=0):
    """
    Normalized probability density function. Normalized to 1.

    imf:      Initial mass function to be normalized.
    max_mass: stellar mass upper bound.
    min_mass: stellar mass lower bound.
    """

    a = 1 / (simpsons_rule_integrate(imf, min_mass, max_mass, 100))

    return a * imf(mass)


# def cumulative_distribution_function(pdf, min_mass, max_mass):
#     """TODO"""

#     return simpsons_rule_integrate(pdf, min_mass, max_mass, 100)


def cumulative_distribution_function(u, mmin, mmax, alpha=alp):
    """
    This version is obtained through analytical means. 
    Credit to Ben Amend.

    """

    fac1 = np.power(mmax, 1.0 - alpha) - np.power(mmin, 1.0 - alpha)
    fac2 = u * fac1 + np.power(mmin, 1.0 - alpha)
    return np.power(fac2, 1.0 / (1.0 - alpha))


def mass_sampling_function(n, mmin, mmax):
    """TODO"""

    masses = []

    for i in range(n):
        masses.append(cumulative_distribution_function(
            random.uniform(0, 1), mmin, mmax)
        )

    return masses


# ------------------------------------------------------------------------------
# HRD of stars generated from IMF. Log L vs. Log T.
# ------------------------------------------------------------------------------

def hrd_generated_stars(n_mass, min_mass, max_mass):
    # Define ranges for plotting
    mass_list = mass_sampling_function(n_mass, min_mass, max_mass)
    effective_temp_list = []
    luminosity_list = []
    for mass in mass_list:
        effective_temp_list.append(effective_temperature(mass=mass))
        luminosity_list.append(luminosity_mass_function(mass))

    plotting_scheme_loglog(effective_temp_list, luminosity_list,
                           x_axis='Log T (K)', y_axis='Log L',
                           title='N=1000 Log L v. Log T', invert_x=True)


def effective_temperature(mass):
    """
    From L = 4*pi*radius^2*sigma*T^4.

    mass: in solar masses.
    """

    return (luminosity_mass_function(mass) * solar_lum /
            (4 * np.pi * (radius_mass_function(mass) * solar_rad) ** 2 *
             const.Stefan_Boltzmann)) ** (1 / 4)


# ------------------------------------------------------------------------------
# Color Indices Portion.
# ------------------------------------------------------------------------------

def planck_function(nu, t):
    """
    Planck function definition.
    nu: Filter wavelength.
    T:  temperature.
    """

    h = const.Planck
    k = const.Boltzmann
    c = const.speed_of_light

    exponent = (h * c) / (nu * k * t)

    # B
    return (2.0 * h * c ** 2) / nu ** 5 * 1.0 / (np.e ** exponent - 1.0)


def color_index_scheme(masses):
    temp_range = []
    abs_mags = []
    for mass in masses:
        temp_range.append(effective_temperature(mass))
        abs_mags.append(abs_mag(mass))

    bv = []
    ub = []
    uv = []

    for T in temp_range:
        u_temp = planck_function(3.65e-7, T)
        b_temp = planck_function(4.45e-7, T)
        v_temp = planck_function(5.51e-7, T)

        bv.append(-2.5 * np.log(b_temp / v_temp))
        ub.append(-2.5 * np.log(u_temp / b_temp))
        uv.append(-2.5 * np.log(u_temp / v_temp))

    plotting_scheme(uv,
                    abs_mags,
                    invert_y=True,
                    title='Magnitude vs. CI',
                    x_axis='CI',
                    y_axis='Absolute Magnitude')


def abs_mag(mass):
    """TODO"""

    return -2.5 * np.log((luminosity_mass_function(mass) *
                          solar_lum) / zero_lum)


if __name__ == '__main__':
    min_mass = 0.1
    max_mass = 100
    n_mass = 1000

    sampled_masses = mass_sampling_function(n_mass, min_mass, max_mass)
    color_index_scheme(sampled_masses)
