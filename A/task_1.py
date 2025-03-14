#!/usr/bin/env python

from numpy import exp, isclose, inf
from scipy import constants
from scipy.integrate import fixed_quad, quad

# AB: Compute Stefan-Boltzmann Constant
integration_result = fixed_quad(lambda z: z**3 / (1 - z)**5 / (exp(z / (1 - z)) - 1), 0, 1, n=40)[0]
stefan_boltzmann_constant = (2 * constants.pi * constants.k**4) / (constants.c**2 * constants.h**3) * integration_result

print(f'Stefan-Boltzmann constant = {stefan_boltzmann_constant:.6e} W/m^2/K^4')
if isclose(stefan_boltzmann_constant, constants.Stefan_Boltzmann):
    print('The value is consistent with the scipy.constants value.')

# C: Verify Consistency with Alternative Integral
alternative_integral_result = quad(lambda x: x**3 / (exp(x) - 1), 0, inf)[0]

if isclose(integration_result, alternative_integral_result):
    print('The two integrals are consistent.')
