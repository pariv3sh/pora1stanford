import math
import typing as T
import sys

import numpy as np
from numpy import linalg
# from scipy.integrate import cumtrapz  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from utils import save_dict, maybe_makedirs

class State:
    def __init__(self, x: float, y: float, V: float, th: float) -> None:
        self.x = x
        self.y = y
        self.V = V
        self.th = th

    @property
    def xd(self) -> float:
        return self.V*np.cos(self.th)

    @property
    def yd(self) -> float:
        return self.V*np.sin(self.th)


phi = [lambda t: 1, lambda t: t, lambda t: t ** 2, lambda t: t ** 3]
dphi = [lambda t: 0, lambda t: 1, lambda t: 2 * t, lambda t: 3 * (t ** 2)]
ddphi = [lambda t: 0, lambda t: 0, lambda t: 2, lambda t: 6 * t]
    

def compute_traj_coeffs(initial_state: State, final_state: State, tf: float) -> np.ndarray:
    """
    Inputs:
        initial_state (State)
        final_state (State)
        tf (float) final time
    Output:
        coeffs (np.array shape [8]), coefficients on the basis functions

    basis functions fi1(t)=1; fi2(t)=t; fi3(t)=t^2; fi4(t)=t^3
    Hint: Use the np.linalg.solve function.
    """
    ########## Code starts here ##########

    istate, fstate = initial_state, final_state
    A = np.array([ 
            [phi[i](0) for i in range(len(phi))], 
            [phi[i](tf) for i in range(len(phi))],
            [dphi[i](0) for i in range(len(dphi))],
            [dphi[i](tf) for i in range(len(dphi))]
         ])

    Bx = np.array([ istate.x, 
           fstate.x, 
           istate.xd, 
           fstate.xd 
         ] )

    By = np.array([ istate.y,
           fstate.y,
           istate.yd, 
           fstate.yd])
          
    xcoeffs = np.linalg.solve(A, Bx)
    ycoeffs = np.linalg.solve(A, By)
    
    #print(f'{xcoeffs=} {xcoeffs.shape=}')
    #print(f'{ycoeffs=} {ycoeffs.shape=}')
    
    coeffs = np.append(xcoeffs, ycoeffs)
    
    ########## Code ends here ##########
    return coeffs

def compute_traj(coeffs: np.ndarray, tf: float, N: int) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Inputs:
        coeffs (np.array shape [8]), coefficients on the basis functions
        tf (float) final_time
        N (int) number of points
    Output:
        t (np.array shape [N]) evenly spaced time points from 0 to tf
        traj (np.array shape [N,7]), N points along the trajectory, from t=0
            to t=tf, evenly spaced in time
    """
    t = np.linspace(0, tf, N) # generate evenly spaced points from 0 to tf
    traj = np.zeros((N, 7)) # x, y, theta, dx, dy, ddx ddy
    xcoeffs = coeffs[0:4]
    ycoeffs = coeffs[4:]

    phi_s = [phi[i](t) for i in range(len(phi))]
    dphi_s = [dphi[i](t) for i in range(len(dphi))]
    ddphi_s = [ddphi[i](t) for i in range(len(ddphi))]
    
    # ########## Code starts here ##########
    xt = sum([xcoeffs[i] * phi_s[i] for i in range(len(xcoeffs))])
    yt = sum([ycoeffs[i] * phi_s[i] for i in range(len(ycoeffs))])
    xdott = sum([xcoeffs[i] * dphi_s[i] for i in range(len(xcoeffs))])
    ydott = sum([ycoeffs[i] * dphi_s[i] for i in range(len(ycoeffs))])
    dxdott = sum([xcoeffs[i] * ddphi_s[i] for i in range(len(xcoeffs))])
    dydott = sum([ycoeffs[i] * ddphi_s[i] for i in range(len(ycoeffs))])
    theta_t = np.arctan2(ydott, xdott)
    
    traj[:, 0] = xt
    traj[:, 1] = yt
    traj[:, 2] = theta_t
    traj[:, 3] = xdott
    traj[:, 4] = ydott
    traj[:, 5] = dxdott
    traj[:, 6] = dydott

    ########## Code ends here ##########
    
    print(traj.shape)
    return t, traj

def compute_controls(traj: np.ndarray) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Input:
        traj (np.array shape [N,7])
    Outputs:
        V (np.array shape [N]) V at each point of traj
        om (np.array shape [N]) om at each point of traj
    """
    ########## Code starts here ##########
    V = np.sqrt(traj[:, 3] ** 2  + traj[:, 4] ** 2)
    xdot, ydot, xdotdot, ydotdot = traj[:, 3], traj[:, 4], traj[:, 5], traj[:, 6]
    om = ( xdot * ydotdot - ydot * xdotdot ) / ( xdot ** 2 + ydot ** 2)
    om[0] = 0 # first value is nan when x, y = 0, 0
    #print(om)
    ########## Code ends here ##########

    return V, om

def interpolate_traj(
    traj: np.ndarray,
    tau: np.ndarray,
    V_tilde: np.ndarray,
    om_tilde: np.ndarray,
    dt: float,
    s_f: State
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inputs:
        traj (np.array [N,7]) original unscaled trajectory
        tau (np.array [N]) rescaled time at orignal traj points
        V_tilde (np.array [N]) new velocities to use
        om_tilde (np.array [N]) new rotational velocities to use
        dt (float) timestep for interpolation
        s_f (State) final state

    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    """
    # Get new final time
    tf_new = tau[-1]

    # Generate new uniform time grid
    N_new = int(tf_new/dt)
    t_new = dt*np.array(range(N_new+1))

    # Interpolate for state trajectory
    traj_scaled = np.zeros((N_new+1,7))
    traj_scaled[:,0] = np.interp(t_new,tau,traj[:,0])   # x
    traj_scaled[:,1] = np.interp(t_new,tau,traj[:,1])   # y
    traj_scaled[:,2] = np.interp(t_new,tau,traj[:,2])   # th
    # Interpolate for scaled velocities
    V_scaled = np.interp(t_new, tau, V_tilde)           # V
    om_scaled = np.interp(t_new, tau, om_tilde)         # om
    # Compute xy velocities
    traj_scaled[:,3] = V_scaled*np.cos(traj_scaled[:,2])    # xd
    traj_scaled[:,4] = V_scaled*np.sin(traj_scaled[:,2])    # yd
    # Compute xy acclerations
    traj_scaled[:,5] = np.append(np.diff(traj_scaled[:,3])/dt,-s_f.V*om_scaled[-1]*np.sin(s_f.th)) # xdd
    traj_scaled[:,6] = np.append(np.diff(traj_scaled[:,4])/dt, s_f.V*om_scaled[-1]*np.cos(s_f.th)) # ydd

    return t_new, V_scaled, om_scaled, traj_scaled

if __name__ == "__main__":
    # Constants
    tf = 25.

    # time
    dt = 0.005
    N = int(tf/dt)+1
    t = dt*np.array(range(N))

    # Initial conditions
    s_0 = State(x=0, y=0, V=0.5, th=-np.pi/2)

    # Final conditions
    s_f = State(x=5, y=5, V=0.5, th=-np.pi/2)

    coeffs = compute_traj_coeffs(initial_state=s_0, final_state=s_f, tf=tf)
    t, traj = compute_traj(coeffs=coeffs, tf=tf, N=N)    
    V,om = compute_controls(traj=traj)

    maybe_makedirs('plots')

    # Plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(traj[:,0], traj[:,1], 'k-',linewidth=2)
    plt.grid(True)
    plt.plot(s_0.x, s_0.y, 'go', markerfacecolor='green', markersize=15)
    plt.plot(s_f.x, s_f.y, 'ro', markerfacecolor='red', markersize=15)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title("Path (position)")
    plt.axis([-1, 6, -1, 6])

    theta_t = traj[:, 2]
    ax = plt.subplot(1, 2, 2)
    plt.plot(t, V, linewidth=2)
    plt.plot(t, om, linewidth=2)
    plt.plot(t, theta_t, linewidth=2)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc="best")
    plt.title('Original Control Input')
    plt.tight_layout()

    plt.savefig("plots/differential_flatness.png")
    plt.show()
