import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from simulation_module import simulation
import time
import scipy
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import bayex
import jaxopt
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
    steps_per_snapshots=20
    total_steps=1000

    start = time.perf_counter()
    Data = simulation(steps_per_snapshots,total_steps,ICs,ext_fields,dx,dt,(0,0,0,0))
    end = time.perf_counter()
    t = jnp.array(Data['Time'])

    ke_over_time = jnp.array(Data['Kinetic Energy'])
    E_field_energy_density = jnp.array(Data['E-field Energy'])
    E_field_energy = jnp.sum(E_field_energy_density,axis=1)

    return jnp.mean(E_field_energy)


domain = {'x': bayex.domain.Real(0.0, 0.5)}
optimizer = bayex.Optimizer(domain=domain, maximize=True, acq='PI')

# Define some prior evaluations to initialise the GP.
params = {'x': [0.09599999]}
ys = [func(x) for x in params['x']]
opt_state = optimizer.init(ys, params)

# Sample new points using Jax PRNG approach.
ori_key = jax.random.key(42)
for step in range(100):
    key = jax.random.fold_in(ori_key, step)
    new_params = optimizer.sample(key, opt_state)
    y_new = func(**new_params)
    opt_state = optimizer.fit(opt_state, y_new, new_params)
    print(y_new,new_params)

# from scipy.optimize import least_squares
# x0=jnp.array([0.09599999])
# #x0=jnp.array([0.09599999,2042.0353])
# res_1 = least_squares(func, x0)
# print(res_1.x)
# print(res_1.cost)
# print(res_1.optimality)

# x0=jnp.array([0.09599999])
# solver = jaxopt.LBFGS(fun=func, maxiter=1)
# res = solver.run(x0)
# params, state = res
# print(params, state)
