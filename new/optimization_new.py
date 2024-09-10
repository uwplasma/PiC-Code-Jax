import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from simulation_module import simulation
import time
import scipy
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import bayex
from scipy.optimize import least_squares, minimize
from tqdm import tqdm
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

# @jax.jit
def func(x):
    A=x[0]
    k=0.3

    ext_E = jnp.zeros(shape=(len(grid),3))
    for i in range(len(grid)):
        ext_E = ext_E.at[i].set(jnp.array(
            [-weight*1.6e-19*no_pseudoelectrons*A*jnp.sin(k*20*jnp.pi*(grid[i]+dx/2)/L)/(k*20*jnp.pi*8.85e-12),0,0])
            )
        
    ext_B = jnp.zeros(shape=(len(grid),3))
    ext_fields = (ext_E,ext_B)

    dt = dx/(2*3e8)
    steps_per_snapshots=20
    total_steps=300

    start = time.perf_counter()
    Data = simulation(steps_per_snapshots,total_steps,ICs,ext_fields,dx,dt,(0,0,0,0))
    end = time.perf_counter()
    t = jnp.array(Data['Time'])

    ke_over_time = jnp.array(Data['Kinetic Energy'])
    E_field_energy_density = jnp.array(Data['E-field Energy'])
    E_field_energy = jnp.sum(E_field_energy_density,axis=1)

    return jnp.mean(E_field_energy)

start_time = time.time();sol = func([1]);print(f'For solution {sol} time taken is {time.time()-start_time:.2f}s')

# domain = {'x': bayex.domain.Real(0.0, 0.5)}
# optimizer = bayex.Optimizer(domain=domain, maximize=True, acq='PI')

# # Define some prior evaluations to initialise the GP.
# params = {'x': [0.09599999]}
# ys = [func(x) for x in params['x']]
# opt_state = optimizer.init(ys, params)

# # Sample new points using Jax PRNG approach.
# ori_key = jax.random.key(42)
# for step in range(100):
#     key = jax.random.fold_in(ori_key, step)
#     new_params = optimizer.sample(key, opt_state)
#     y_new = func(**new_params)
#     opt_state = optimizer.fit(opt_state, y_new, new_params)
#     print(y_new,new_params)

x0=jnp.array([0.15])
max_number_function_evaluations = 15
max_number_iterations = 10
tolerance_to_stop_optimization = 1e-5

## Using least squares optimization
start_time = time.time()
res_ls = least_squares(func, x0, verbose=2, ftol=tolerance_to_stop_optimization, 
                       max_nfev=max_number_function_evaluations, bounds=(0.07, jnp.inf))
sol_ls = res_ls.x[0]
print(f'For solution with Least Squares x={sol_ls} time taken is {time.time()-start_time}')


## Using BFGS optimization
from scipy.optimize import Bounds
bounds = Bounds(0.07, jnp.inf)  # Lower bound 0.1 and no upper limit

start_time = time.time()
res_bfgs = minimize(func, x0, method='L-BFGS-B', options={'disp': True,
                               'maxiter': max_number_iterations, 'maxfun': max_number_function_evaluations,
                               'gtol': tolerance_to_stop_optimization}, bounds=bounds)
sol_bfgs = res_bfgs.x[0]
print(f'For solution with L-BFGS-B x={sol_bfgs} time taken is {time.time()-start_time}')


x0_array = jnp.linspace(min(min(sol_bfgs*1.1,sol_ls*1.1),0.01),max(max(sol_bfgs*1.1,sol_ls*1.1),0.6),30)
sol_array = [func([x]) for x in tqdm(x0_array)]

plt.figure()
plt.axvline(x=x0[0], linestyle='--', color='k', linewidth=2, label='Initial Guess')
plt.axvline(x=sol_ls, linestyle='--', color='r', linewidth=2, label='Least Squares Minimum')
plt.axvline(x=sol_bfgs, linestyle='--', color='b', linewidth=2, label='L-BFGS-B Minimum')
plt.plot(x0_array,sol_array)
plt.xlabel("A")
plt.ylabel("mean E-field Energy")
plt.legend()
plt.show()