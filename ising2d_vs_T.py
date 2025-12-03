
import numpy as np
import random
import math
import time
import os
import matplotlib.pyplot as plt
from numba import jit

# Parameters
NX = 64
NY = 64
ntherm = 1000
VisualDisplay = 0  # Set to 1 if you want lattice display
SleepTime = 300000  # in microseconds

@jit
def update_spin(nx, ny, env, spin):
    """Do a Metropolis update on a spin at position (nx, ny) whose environment is env"""
    current_spin = spin[nx, ny]
    newspin = 1 if np.random.random() < 0.5 else -1
    DeltaBetaE = -(newspin - current_spin) * env
    if DeltaBetaE <= 0 or np.random.random() < math.exp(-DeltaBetaE):
        spin[nx, ny] = newspin

@jit
def sweep(beta, h, spin):
    """Sweep through all lattice sites"""
    for nx in range(1, NX + 1):
        for ny in range(1, NY + 1):
            environment = (beta * (spin[nx, ny-1] + spin[nx, ny+1] +
                                   spin[nx-1, ny] + spin[nx+1, ny]) + h)
            update_spin(nx, ny, environment, spin)

@jit
def initialize_hot(spin):
    """Initialize lattice with random spins"""
    spin[1:-1, 1:-1] = np.where(np.random.random((NX, NY)) < 0.5, 1, -1)

@jit
def magnetization(spin):
    """Calculate average magnetization"""
    return np.mean(spin[1:-1, 1:-1])

@jit
def energy(spin, h):
    """Calculate energy of the configuration"""
    E = 0
    for nx in range(1, NX + 1):
        for ny in range(1, NY + 1):
            s = spin[nx, ny]
            E -= s * (spin[nx+1, ny] + spin[nx-1, ny] +
                      spin[nx, ny+1] + spin[nx, ny-1])
            E -= h * s
    return E / 2.0  # divide by 2 because each pair counted twice

def display_lattice(T, spin):
    """Display the lattice configuration"""
    if SleepTime > 0:
        os.system('cls' if os.name == 'nt' else 'clear')
    
    chars = np.where(spin[1:-1, 1:-1] == 1, 'X', '-')
    for row in chars:
        print(''.join(row))
    
    print(f"T = {T:.6f}:   magnetization <sigma> = {magnetization(spin):.6f}")
    
    if SleepTime > 0:
        time.sleep(SleepTime / 1_000_000)
    else:
        print()

def main():
    spin = np.zeros((NX + 2, NY + 2), dtype=np.int8)
    
    output_filename = "ising2d_vs_T.dat"
    
    print(f"Program calculates <sigma>, <E>, Cv vs. T for a 2D Ising model of "
          f"{NX}x{NY} spins with free boundary conditions.\n")
    
    np.random.seed(int(time.time()))
    
    nsweep = int(input("Enter # sweeps per temperature sample:\n"))
    h = float(input("Enter value of magnetic field parameter h:\n"))
    Tmax = float(input("Enter starting value (maximum) of temperature T (=1/beta):\n"))
    ntemp = int(input("Enter # temperatures to simulate:\n"))
    
    initialize_hot(spin)
    
    T_list, E_list, Cv_list, M_list = [], [], [], []
    
    with open(output_filename, 'w') as output:
        for itemp in range(ntemp, 0, -1):
            T = (Tmax * itemp) / ntemp
            beta = 1 / T
            
            # Thermalization
            for _ in range(ntherm):
                sweep(beta, h, spin)
            
            # Measurement
            total_E = 0
            total_E2 = 0
            total_M = 0
            total_M2 = 0
            
            for _ in range(nsweep):
                sweep(beta, h, spin)
                E = energy(spin, h)
                M = np.sum(spin[1:-1, 1:-1])
                
                total_E += E
                total_E2 += E**2
                total_M += M
                total_M2 += M**2
            
            avg_E = total_E / nsweep
            avg_E2 = total_E2 / nsweep
            avg_M = total_M / (nsweep * NX * NY)
            
            Cv = beta**2 * (avg_E2 - avg_E**2) / (NX * NY)
            
            T_list.append(T)
            E_list.append(avg_E / (NX * NY))  # energy per spin
            Cv_list.append(Cv)
            M_list.append(avg_M)
            
            output.write(f"{T:.6f} {avg_E/(NX*NY):.6f} {Cv:.6f} {avg_M:.6f}\n")
            
            if VisualDisplay:
                display_lattice(T, spin)
    
    print(f"Output file is {output_filename}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(T_list, E_list, 'r-', label='Energy per spin')
    plt.plot(T_list, Cv_list, 'b:', label='Specific Heat')
    plt.plot(T_list, M_list, 'g--', label='Magnetization')
    plt.xlabel('Temperature T')
    plt.ylabel('Observable')
    plt.title('2D Ising Model Observables vs Temperature')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('ising.pdf')
    plt.show()
if __name__ == "__main__":
    main()
