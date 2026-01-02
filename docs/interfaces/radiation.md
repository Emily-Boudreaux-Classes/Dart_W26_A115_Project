# Radiation Module
Perhaps this module should be called boundary conditions or opacity, or
perhaps it should be two smaller modules. That is a choice you are free to
make

There are two key components of this module

1. Implement some prescription for radiative opacity so that the solver may access it
2. Implement some atmospheric boundary condition prescription

The structure of a course has forced these two separate but related modules together so that
I can assign various students to various components. That is why I say it might make
sense to split these into two separate, each smaller, modules.

## Radiative Opacity
The structure equations need a way to probe the opacity of a material. This is
non trivial but may be approximated with Kramer's opacity law. Kramer's opacity
law is given in separate versions for bound-free, free-free, and Thomson
scattering opacities. Note that it is also grey, that is to say it is
wavelength independent.

Bound-free

$$
\bar{\kappa}_{bf} = 4.34\times 10^{25}\frac{g_{bf}}{t}Z(1+X)\rho T^{-7/2}
$$

Free-free

$$
\bar{\kappa}_{ff} = 3.68\times 10^{22}g_{ff}(1-Z)(1+X)\rho T^{-7/2}
$$

Thomson Scattering

$$
\bar{\kappa}_{es} = 0.2(1+X)
$$

The total opacity is the sum of all individual opacity sources.

Aside from using Kramer's opacity law you may also choose to use opacity
tables. Opacity tables are generally how SSE codes handle opacity since they
are much more accurate, originating from detailed plasma physics simulations. 
You can find these tables online or talk to me to get them, some tables to look for 
are 

- OP 
- OPAL
- OPLIB
- Aesopus

These cover different parts of the parameter space (primarily separated by high
or low temperatures) and can be retrieved from webforms normally given some
input composition. The most common approach stellar structure codes will take
is to generate a grid of tables for some characteristic set of compositions
rescaled along [Fe/H] and [$\alpha$/Fe]. At every time step the current composition 
will be queried and the tables closest to the composition will be interpolated together
to get an approximation of the actual opacity table there. 

If you go the table route it should be noted there will be a lot of interpolation to do,
both between compositions but also between temperature and density (which these tables
are generally parameterized by). Do not feel the need to write your own interpolator,
there are loads of tools which do this well. Make use of them.


## Gray Model Atmosphere
The other component to this module (or other module to work on) is the
atmospheric boundary module. I would like you to implement a gray model
atmosphere. We discuss this in class in the notes labeled "Boundary Conditions
of Stellar Models". 

In brief what you will need to do is implement a prescription to find the 
so-called "fitting" temperature and pressure for the structure module to target.

### Pressure
There is a derivation of this in the notes. The pressure at the surface of a star, where
the surface is defined by some optical depth, when making a gray atmosphere approximation
is then the pressure is 

$$
P_{R} = \frac{g_{0}\tau_{s}}{\bar{\kappa}}
$$

and the temperature is 

$$
T(\tau) = \left[\frac{3}{4}\frac{L}{4\pi R^{2}\sigma_{SB}}\left(\tau + \frac{2}{3}\right)\right]^{1/4}
$$

Note that this is equivalent to 

$$
T(\tau) = \left[\frac{3}{4}T_{eff}^{4}\left(\tau + \frac{2}{3}\right)\right]^{1/4}
$$

There are a few ways which you can use this, namley

1. Use the results as a fitting point
2. Incorporate these as additional ``closure'' equations into your system

I'm not sure which will be more straight forward, I suspect the former. 

In the fitting approach you will normally actually let the atmosphere extend to
a much deeper depth (say $\tau = 100$) and then compute the pressure and
temperature using the model atmosphere at this point, we will call this out
fitting point.

First lets regognize that we have the results from some previous converged model at
time $t^{(i-1)}$ (if this is the first time step we must somehow manufacture this) 

$$
 [r^{(i-1)}(m), P^{(i-1)}(m), T^{(i-1)}(m), L^{(i-1)}(m), X^{(i-1)}(m)]
$$

Now we need to guess a few things for the current time step, $i$
we will say that the initial guess stellar radius at time step $i$ is
$R^{(i,0)}= r^{(i-1)}(M)$ and that the initial guess for luminosity is
$L^{(i,0)} = L^{(i-1)}(M)$. Note that by using $M$ we are really saying we 
want the outer most gridpoint.

We can then compute

$$
g^{(i, 0)} = \frac{GM}{\left(R^{(i,0)}\right)}^{2}
$$

and

$$
T_{eff}^{(i,0)} = \left(\frac{L^{(i,0)}}{4\pi\sigma_{SB}\left(R^{(i,0)}\right)^{2}}\right)^{1/4}
$$

Now we uses these guesses to evaluate the model atmosphere If you are using a
value of $\tau$ much greater than $2/3$, which is useful for the fitting
approach then you do need to integrate (or you can assume that opacity is
constant through to the fitting optical depth, but that is not a good
approximation)

$$
P_{fit}^{(i,0)} \approx \int_{0}^{\tau_{fit}}\frac{g^{(i,0)}}{\kappa(P,T)}d\tau
$$

and

$$
T_{fit}^{(i,0)} = T_{eff}^{(i,0)}\left(\frac{3}{4}\tau_{fit}+\frac{2}{3}\right)^{1/4}
$$

You now have two concreate numeric values. You can report these to the structure solver. 
It is then the structure solvers job to solve the equations with the boundary conditions

$$
r(0) = 0
$$

$$
L(0) = 0
$$

$$
P^{(i)}(M) = P_{fit}^{(i,0)}
$$

$$
T^{(i)}(M) = T_{fit}^{(i,0)}
$$

Again these are saying that the temperature and pressure at the outter grid
point must match whatever the fitting temperature and pressure are. From these
the solver module / structure module needs to produce

$$
[ r^{i}(m), P^{(i)}(m), T^{(i)}(m), L^{(i)}(m)]
$$

Note that the values predicted by the solver module at this stage will generally not
be the exact same as the guessed values from earlier.

As you can see this is very strongly coupled to the structure module so whomever is working 
on these two modules will need to make sure they are on the same page about things
