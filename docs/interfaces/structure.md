# Structure Module
This module is responsible for collecting input from other modules to solve the
structure equations

This module is the closest thing thing which this project has to a main
component. One might think of it as the organizer of all the other modules
(side-note: I hate that when I try to use a metaphor now I need to worry that
it will sound like AI).

## Structure
The other key part of this module is the structure equations. These are the
four (five) equations of stellar structure we disscussed in class, they must
be encoded and passed to some sort of solver. I do not expect you to implement 
your own custom solver; rather, please use a solver that already exists.

There are few key parts of this

### Collecting Microphysics
The structure equations depend on loads of microphysics, some part of this module will need
to be responsible for calling the microphysics modules to get values such at the 
opacity (perhapse as a function), the specific nuclear energy generation,
and the density.

### Discritization
Numeric solutions must be solved in some discrete space, this module will need
to decide where to place the shells that compose our model. I reccomend you
work in a lagarngian scheme where you use enclosed mass as your indpendent,
radiaus-like, variable. You will need to consider how to space those shells, 
are there regions where shells can be more sparsly placed and are there
regions where they must be more densly spaced (hint: the answer is yes to 
both of those questions).

### Solution
Actual SSE codes uses something called a Henyey method to solve the structure
equations; for this course I reccomend you look into something a bit simpler
like a shooting-method. There are loads of tools which exist to solve coupled
systems of equations with a shooting method. They will not be as stable as a
Henyey method but they will be easier to implement.

### Regime
Based on either the Schwartzchild or Ledoux criterion this module will need to decide
if a particular region is stable to convection. This will determine what equation of 
energy transport is used. Solving this will require careful integration with the 
eos module.

The Schwartzchild criterion is 

$$
\nabla_{rad} < \nabla_{ad}
$$

while the Ledoux criterion is 

$$
\nabla_{rad} < \nabla_{ad} + \frac{\phi}{\delta}\nabla\mu
$$

### Reporting
Finally, this module must be able to report the state in some way back to the time module 
and eventually to some user so that the results can be either stepped through time
or saved and analyzed.
