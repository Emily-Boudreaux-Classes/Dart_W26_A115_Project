# Nuclear Module
This module is responsible for orchestrating the nuclear network

Stars stay in quasi-equilibrium due to the energy release by fusion in their cores. Solving for these is complex, 
it requires solving 

$$
\frac{dY_{i}}{dt} = \sum_{j}^{r}c_{i,j}\mathcal{R}_{j}(\mathbf{Y}, \rho, T) \ \ \forall i=1..N
$$

and

$$
\frac{d\epsilon_{nuc}}{dt} = -N_{A}c^{2}\sum_{i}^{N}\frac{dY_{i}}{dt}m_{i}
$$

Where $\mathcal{R}_{j}$ is the molar reaction flow for some reaction $j$ and
$\mathbf{Y}$ is the vector of molar abunacnes for all species in the network.
Effectivley there is a seperate equation for every possible nuclear reaction
pathway and they are all coupled to each other. These systems can easily
balloon to thousands, tens of thousands, or even hundreds of thousands of
coupled differential equations which must be solved simultaniously. Further complicating 
matter is the extreme stiffness of this system of equations (consider that solar
deuterium evolves on a timescale of roughly $10^{-18}$s while solar hydrogen evolves on
a timescale of roughly $10^{18}$s, a 36 order of magnitude difference in scale).

There are loads of way to simplify this; however, frankly I do not belive they 
are of signifigant pedagogical value for this class. Therefore this module is
quite simple physically. You will interface with my gridfire package which preforms
these calculations.

The goal of this module is then to collect the required input from the rest of
this SSE program, ask gridfire for what the updated composition, specific energy
generation, and specific energy lost to neutrinos is and then report these values
back to the structure module.

There are certainly still a few considerations

1. Nuclear networks generally operate on molar abundances not mass fractions. If the rest of the code uses mass fractions these will need to be carefully handled
2. The returned energy generation is in two parts, generation and  neutrino loss, the neutrino loss must be removed from the total energy generation
3. The returned valus is given as specific energy generation meaning the total energy is the product of that with mass. 
