import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as pc
import scipy as sp


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

    return (a[0]*mass**2.5 + a[1]*mass**6.5 + a[2]*mass**11 + a[3]*mass**19 +a[4]*mass**19.5) / (a[5] + a[6]*mass**2 + a[7]*mass**8.5 + a[8]*mass**18.5 + a[9]*mass**19.5)


if __name__ == '__main__':
    print('yo mama')

    print(luminosity_mass_function(1))
