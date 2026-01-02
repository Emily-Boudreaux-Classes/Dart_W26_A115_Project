# Transport module
Material moves through a star as it evolves, this can be due to both
convection, diffusion, and advection. This module is responsible 
for handling the composition of the star at some time step given 
the previous composition and time step size

A few considerations. 

1. Convective zones mix so efficiently that most code assumes that all convective regions
have a homogeneous composition. This means that this module should either be 
able to evaluate the stability criterion or ask the structure module for the 
stability of various shells. All contiguous convective shells may be homogenized. This is
generally valid during the main sequence.
2. A more detailed handling would be to calculate the diffusion coefficient ($D$) for every shell
the structure module would then need to also calculate the following equation to move material between shells 
each time step

$$
\frac{\partial X_{i}}{\partial t} = \frac{\partial}{\partial m}\left((4\pi r^{2}\rho)^{2}D\frac{\partial X_{i}}{\partial m}\right) + R_{nuc}
$$

## The Diffusion Coefficient

The diffusion coefficient can be build from a few term 

### Convective Mixing
When in a convective region diffusion is dominated by the term

$$
D_{conv} \approx \frac{1}{3}v_{mlt}\ell_{mlt}
$$

Where $v_{mlt}$ is the convective mixing velocity from mixing length theory and $\ell_{mlt}$ is the
mixing length

### Convective Overshoot
Generally we do not want to just stop convection at a boundary; rather, there is some exponential falloff in mixing 
through the boundary. This has a significant impact on mixing fresh fuel into radiative cores for example

$$
D_{over}(z) = D_{conv,0}\exp(-\frac{2z}{f H_{p}})
$$


where f is the convective overshoot parameter (often of order 0.01), $H_{p}$ is the pressure scale height

### Semiconvection
There are some regions of stars stable to the Ledoux criterion but unstable to the Schwarzchild criterion. We say that
these regions are undergoing semiconvection

$$
D_{semi}= \alpha_{sc}\frac{K}{C_{P}\rho}\frac{\nabla_{rad}-\nabla_{ad}}{\nabla_{\mu}}
$$

Where $\alpha_{sc}$ is the semiconvective efficiency parameter (often roughly $\sim$0.01) and $K$ is the 
radiative conductivity

$$
K\equiv \frac{4acT^{3}}{3\kappa\rho}
$$

### Thermohaline Mixing
In regions which are stable to convection (i.e convection is not happening in a significant amount) there is still
some mixing due to the difference in weights of different species. The criterion for if a region is unstable to 
thermohaline mixing is

$$
\frac{\phi}{\delta}\nabla_{\mu} \leq \nabla_{rad} - \nabla_{ad} \leq 0
$$

$$
D_{thermo} = -\alpha_{th}\frac{3K}{2\rho c_{P}}\frac{\frac{\phi}{\delta}\nabla_{\mu}}{\nabla_{ad} - \nabla_{rad}}
$$

### Gravitational Settling
A careful accounting would also include the effects of gravitational settling, which is not true mixing. 
This takes the form of an advert in term in the diffusion equation, where for some advective velocity $v_{i}$

$$
\frac{\partial X_{i}}{\partial t} = \frac{\partial}{\partial m}\left((4\pi r^{2}\rho)^{2}D\frac{\partial X_{i}}{\partial m}\right) + R_{nuc} - \frac{\partial}{\partial m}(X_{i}v_{i})
$$

Note that gravitational settling is quite complex to get right and while it is
relevant for some domains of stellar modeling (white dwarf models, seismology),
I recommend that you do not include this.

### Putting it all together
The total diffusion coefficient for each shell is the sum of all individual components.


## Interface
So we need to think about what the inputs and outputs should be. This module
should be able to take all the inputs needed to compute the diffusion
coefficients (if you decide to adopt a diffusion based model, which I would
strongly push for) and return the net diffusion coefficient for each shell.
This means you will likely be returning some array that is parallel to the mass
coordinate array where each entry is the diffusion coefficient for that mass
coordinate.
