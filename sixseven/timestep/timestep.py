import numpy as np
from scipy.integrate import odeint

def dyn_timestep(X_i, dX_i, step, hfactor, min_step):
    """
    dynamic timestep function, decides how long to step based on paramater change of previous step
    Decreases/increases time step if previous step is more than twice / less than half of the estimate

    X_i (arr): N-dim array containing paramaters of system that are changing over time
    dX_i: (arr): array of shape (X_i.shape) that contains the change in X_i over the previous time step
    step (float): time step of previous iteration
    hfactor (float): paramater multiplied by dp/p to estimate next step
    min_step (float): minimum time step allowed during simulation
    """
    # finds mass element and property that change the most over a given step
    ### impliemnt row,col when searching through multiple mass elements
    idx = np.argmax(dX_i)
    # row, col = np.unravel_index(idx, dX_i.shape)
    dp = dX_i[idx]#[row, col]
    p = X_i[idx]#[row, col]
    
    # estimated time step
    step_est = hfactor*np.sqrt(np.abs(p/dp))
    
    # decrease time step if it is more than twice estimated
    if step_est < step/2 and step > min_step:
        step /= 2
    # increase time step if it is less than half estimated
    if step_est > 2*step: 
        step *= 2
    # do not allow time step to drop below this minimum
    if step < min_step:    
        step = min_step
    
    return step, p, dp


### ----- TESTING FUNCTIONS ----- ###
# predator - prey coupled odes
def dX_idt(t, X_i):
    """
    Takes in simulation/iteration time (t) and a vector (X_i), returns dX_idt over time interval t

    t (float): time of simulation / iteration
    X_i ( (2, ) array): values to evolve using the predator - prey coupled odes

    returns new values, change in values over time, new step size, value most changed, and largest change
    """
    x,y = X_i
    alpha=beta=gamma=delta = 1
    dxdt = (alpha*x - beta*x*y)
    dydt = (-gamma*y + delta*x*y)
    return np.array([dxdt,dydt])

def rungekutta(f, t, X_i, step):
    """
    rk4 implimentation to numerically solve odes, vectorized
    
    :param f: ode(s) to solve 
    :param t: time 
    :param X_i: vector of values to evolve in ode 
    :param step: time step
    """
    k1 = f(t, X_i)
    k2 = f(t+0.5*step, X_i+0.5*k1)*step
    k3 = f(t+0.5*step, X_i+0.5*k2)*step
    k4 = f(t+step, X_i+k3)*step
    new_X_i = X_i + (k1+2*k2+2*k3+k4)/6
    dX_i = k4/step
    new_step, p, dp = dummy_timestep(new_X_i, dX_i, step)
    return new_X_i, dX_i, new_step, p, dp

def dummy_evol(t, X_i, step):
    new_X_i, dX_i, step, p, dp = rungekutta(dX_idt, t, X_i, step)
    return new_X_i, dX_i, step, p, dp 