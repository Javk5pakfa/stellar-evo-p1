import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as pc
import scipy as sp


def planck_function(nu, T):
    '''
    Planck function definition.
    nu: Filter wavelength.
    T: temperature.
    '''

    h = pc.Planck
    k = pc.Boltzmann
    c = pc.speed_of_light

    exponent = (h * c) / (nu * k * T)
    B = (2.0 * h * c**2) / nu**5 * 1.0 / (np.e**exponent - 1.0)

    return B


if __name__ == '__main__':

    temp_range = np.linspace(3000, 30000)

    U = planck_function(3.65e-7, temp_range)
    B = planck_function(4.45e-7, temp_range)
    V = planck_function(5.51e-7, temp_range)

    BV = -2.5 * np.log(B/V)
    UB = -2.5 * np.log(U/B)
    UV = -2.5 * np.log(U/V)

    # Add Solar CI info.
    B_Sun = planck_function(4.45e-7, 5772)
    V_Sun = planck_function(5.51e-7, 5772)
    BV_Sun = -2.5 * np.log(B_Sun/V_Sun)

    sun_pt = (5772, BV_Sun)

    # Plotting.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Color Index vs. Temperature')

    ax.plot(temp_range, BV, label='B-V')
    ax.plot(temp_range, UB, 'r', label='U-B')
    ax.plot(temp_range, UV, label='U-V')
    ax.plot(sun_pt[0], sun_pt[1], '*', label='Sun (0.65)')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('CI').set_rotation(0)
    ax.legend(loc='upper right')

    plt.xscale('log')
    plt.show()
