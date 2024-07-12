import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from simulation_module import simulation
import time
import scipy
import matplotlib.pyplot as plt
import jaxopt
import optax
from scipy.signal import find_peaks
import pandas as pd
import os

#Creating box and grid
box_size_x = 1e-2
box_size_y = 1e-2
box_size_z = 1e-2
box_size = (box_size_x,box_size_y,box_size_z)

dx=3e-4
grid = jnp.arange(-box_size_x/2+dx/2,box_size_x/2+dx/2,dx)
staggered_grid = grid + dx/2

#Creating particle ICs
no_pseudoelectrons = 3000
L= box_size_x
xs = jnp.array([jnp.linspace(-L/2,L/2,no_pseudoelectrons)])
seed = 1701
key = jax.random.PRNGKey(seed)
electron_ys = jax.random.uniform(key,shape=(1,no_pseudoelectrons),minval=-box_size_y/2,maxval=box_size_y/2)
electron_zs = jax.random.uniform(key,shape=(1,no_pseudoelectrons),minval=-box_size_z/2,maxval=box_size_z/2)
electron_xs_array = jnp.transpose(jnp.concatenate((xs,electron_ys,electron_zs)))
#%%
'Electron-electron stream with stationary ions'
ion_ys = jax.random.uniform(key,shape=(1,no_pseudoelectrons),minval=-box_size_y/2,maxval=box_size_y/2)
ion_zs = jax.random.uniform(key,shape=(1,no_pseudoelectrons),minval=-box_size_z/2,maxval=box_size_z/2)
ion_xs_array = jnp.transpose(jnp.concatenate((xs,ion_ys,ion_zs)))

particle_xs_array = jnp.concatenate((electron_xs_array,ion_xs_array))
no_pseudoparticles = len(particle_xs_array)

alternating_ones = (-1)**jnp.array(range(0,no_pseudoelectrons))
v0=0.5e8
electron_vxs = v0*alternating_ones
ion_vxs = jnp.zeros(no_pseudoelectrons)
vxs = jnp.concatenate((electron_vxs,ion_vxs))
vys = jnp.zeros(no_pseudoparticles)
vzs = jnp.zeros(no_pseudoparticles)
particle_vs_array = jnp.transpose(jnp.concatenate((jnp.array([vxs]),jnp.array([vys]),jnp.array([vzs]))))

weight = 1e13
q_es = -1.6e-19*weight*jnp.ones(shape=(no_pseudoelectrons,1))
q_ps = 1.6e-19*weight*jnp.ones(shape=(no_pseudoelectrons,1))
qs = jnp.concatenate((q_es,q_ps))
m_es = 9.1e-31*weight*jnp.ones(shape=(no_pseudoelectrons,1))
m_ps = 1.67e-27*weight*jnp.ones(shape=(no_pseudoelectrons,1))
ms = jnp.concatenate((m_es,m_ps))
q_mes = -1.76e11*jnp.ones(shape=(no_pseudoelectrons,1))
q_mps = 9.56e7*jnp.ones(shape=(no_pseudoelectrons,1))
q_ms = jnp.concatenate((q_mes,q_mps))

#%%
particles = (particle_xs_array,particle_vs_array,qs,ms,q_ms,
             (no_pseudoelectrons,no_pseudoparticles-no_pseudoelectrons),
             weight)


#Creating initial fields

E_fields = jnp.zeros(shape=(len(grid),3))
B_fields = jnp.zeros(shape=(len(grid),3))

fields = (E_fields,B_fields)

ICs = (box_size,particles,fields)


def func(x):
    A=x
    k=2042.0353
    ext_E = jnp.zeros(shape=(len(grid),3))
    for i in range(len(grid)):
        ext_E = ext_E.at[i].set(jnp.array(
            [-weight*1.6e-19*no_pseudoelectrons*A*jnp.sin(k*(grid[i]+dx/2))/(k*L*8.85e-12),0,0])
            )
        
    ext_B = jnp.zeros(shape=(len(grid),3))
    ext_fields = (ext_E,ext_B)

    dt = dx/(2*3e8)
    steps_per_snapshots=1
    total_steps=350

    start = time.perf_counter()
    Data = simulation(steps_per_snapshots,total_steps,ICs,ext_fields,dx,dt,(0,0,0,0))
    end = time.perf_counter()
    t = jnp.array(Data['Time'])
    E_field_energy_density = jnp.array(Data['E-field Energy'])
    E_field_energy = jnp.sum(E_field_energy_density,axis=1)

    # Find the maximum E-field energy and its corresponding time
    max_E_field_energy = jnp.max(E_field_energy)
    max_E_field_energy_index = jnp.argmax(E_field_energy)
    corresponding_time = t[max_E_field_energy_index]

    # Calculate the average of the E-field energy after the corresponding time
    E_field_energy_after_max = E_field_energy[max_E_field_energy_index+1:]

    return jnp.mean(E_field_energy_after_max)

# Define the range of values for A and k
A_values = jnp.linspace(0.001, 0.5, 450)

for i in range(len(A_values)):

        # Append the data to the list
        data = {
            'A': A_values[i],
            'grad': func(A_values[i])
        }

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame([data])

        # File path
        file_path = 'D:\\Desktop\\code\\PiC-Code-Jax-main\\new\\300_steps_new.csv'

        # Check if file exists
        if os.path.exists(file_path):
            # If file exists, append data without writing the header
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            # If file does not exist, create it and write the data with the header
            df.to_csv(file_path, mode='w', header=True, index=False)

        print(f"Data saved to {file_path}")
