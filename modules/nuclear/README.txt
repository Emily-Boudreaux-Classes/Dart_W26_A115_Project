-----------------------------------------------------------------------------------------------
nuc_burn.py is the main script with only functions defined. the burn() function is the only function you need from here. 
you should import nuc_burn at the top of your script.
- nuc_burn.burn() takes inputs of temp: float, rho: float, time: float, comp=None
'temp', 'rho', and 'comp' should be arrays. 'time' is your time step in seconds. 
burn() outputs a list of NetOut structs that contain the results for each shell evolved over the time step.
This includes delta energy, mean molecular mass, mass fraction, etc.
* for the very first time step, leave 'comp' as none (it calls a function with an initial composition)
* for the following time steps, update 'comp' with the previous mass fraction burn() output (see how to access below)

To run burn() within a time stepping function without a bunch of stuff outputting to your terminal,
use burn() within a script and call the script with:
python myscript.py > stdout 2>&1 
- you can then type 'cat stdout' to see the logging of your code if you want to check for errors or anything. 

Example usage: 
- For initial run
temps = np.linspace(1.5e7, 2e7, 100)
rhos = np.linspace(1.5e2, 1.5e2, 100)
results = nuc_burn.burn(temps, rhos, 1000)

# To see the functions you can access within results:
print(results[0].__dir__())
# To access mass fraction from Composition object:
print(results[0].composition.getMassFraction())

- For following run:
results2 = nuc_burn.burn(newtemp, newrho, 1000, newcomp)
- and so on...
-----------------------------------------------------------------------------------------------
scratch.py has the same code as nuc_burn.py, but is where I played with stuff and left messy comments for myself. 
this code can be run + altered if needed for trouble-shooting. it has arrays with dummy values defined for temp and rho at the end of the script, and a print statement.
-----------------------------------------------------------------------------------------------
test.py is simply testing the importing of nuc_burn and the functionality of burn() with dummy values for temp and rho.