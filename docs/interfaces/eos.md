# Equation of State Module
This module is responsible for evaluating solutions to an equation of state.
There are a few ways to go about this, you might assume a classical ideal gas
equation of state (an okay approximation for stars between 0.7 and 5 solar
masses). You migth provide multiple regimes and then have seperate equations of
state for each regime (i.e. a low mass regime, a ultra-relativistic regime, and
a classical regime); finally, you migth choose to use tabulated equations of state
to achive the most physical accuracy at the cost of a good but more software 
complexity.

Regardless of what you choose your equation of state you must be able to meet the
following minimum interface

## Inputs
- Pressure
- Temperature
- Composition

## Outputs
- Density
- Specific Internal Energy
- Specific Entropy
- Derivatives
    - $\chi_{\rho} = \left(\frac{\partial \ln P}{\partial \ln \rho}\right)_{T}$
    - $\chi_{T} = \left(\frac{\partial \ln P}{\partial \ln T}\right)_{\rho}$
    - $c_{v} = \left(\frac{\partial u}{\partial T}\right)_{\rho}$
    - $\nabla_{ad} = \left(\frac{\partial \ln T}{\partial \ln P}\right)_{s}$


## Implementation
There are different choices which can be made here; I would push you to start 
with a classical ideal gas + radiation pressure. The way to think about what is 
happening is that the solver will guess some pressure and temperature and 
then ask you for the thermodynamic state based on that temperature
and pressure. You report to the solver all the various outputs and then it may 
iterate its solution, update it perhapse, and re query you.

Note that one challenge (which you could choose to either address or simpley document)
is that we have choosen to parameterize by pressure and temperature and solve for density. 
In reality it is often easier to solve parameterize by density and temperature and
solver for pressure. The scheme we have used here will present challenges down the line
related to inversion. You may choose to work in a density and temperature 
scheme from the outset if you like, its really not much worse; thought it does
require careful coordination with the solver module to enure that it is guessing
density and not pressure and is instead taking pressure from the EOS.


## Some Equations of State

Classical Ideal Gas

$$
P_{gas} = \frac{k_{B}\rho T}{\mu m_{H}}
$$

Radiation Pressure

$$
P_{rad} = \frac{1}{3}aT^{4}
$$

Degenerate Electron Gas ($\mu_{e}\equiv \frac{2}{1+X}$)

$$
P_{e} = 1.0036\times 10^{13}\left(\frac{\rho}{\mu_{e}}\right)^{5/3}
$$

Ultra-relativistic electon gas

$$
P_{e} = 1.24^{15}\left(\frac{\rho}{\mu_{e}}\right)^{4/3}
$$

Note that pressure is linear when considering distinc species so the total
pressure is the sum of all the individual terms here so long as there 
is only one term of gas, radiation, and electron pressure. You can approximate 
the electron pressure as the quadrature sum of the two electron 
pressure terms.
