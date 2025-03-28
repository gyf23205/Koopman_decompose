import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

def generate_dataset(dynamics_model, iteration_len, training_number, testing_number, simulation_time, bias = False, Generate_Koopman_data = False):
    if dynamics_model == 'vdp':
        state_dimension = 2
        input_dimension = 0
        mu = 2          # Van der Pol parameter
        state_bounds = { # Make sure this order == state array order
        'position_x': [-3,3],
        'position_y': [-3,3],
        } 
    else:
        print('TBD')

    Koopman_training_trajectory = np.zeros([state_dimension + input_dimension, iteration_len, training_number]) # [state, input]
    Koopman_testing_trajectory = np.zeros([state_dimension + input_dimension, iteration_len, testing_number]) # [state, input]
    key_list = list(state_bounds.keys())

    # Generate training data
    for i in range(training_number):
        for j in range(0,len(key_list)):
            if bias:
                if j == 0:
                    Koopman_training_trajectory[j,0,i] = np.random.uniform(state_bounds[key_list[j]][0],state_bounds[key_list[j]][1]*0.5)
                else:
                    Koopman_training_trajectory[j,0,i] = np.random.uniform(state_bounds[key_list[j]][0],state_bounds[key_list[j]][1])
            else:
                Koopman_training_trajectory[j,0,i] = np.random.uniform(state_bounds[key_list[j]][0],state_bounds[key_list[j]][1])

    # Generate testing data
    for i in range(testing_number):
        for j in range(0,len(key_list)):
            if bias:
                if j == 0:
                    Koopman_testing_trajectory[j,0,i] = np.random.uniform(state_bounds[key_list[j]][1]*0.5,state_bounds[key_list[j]][1])
                else: 
                    Koopman_testing_trajectory[j,0,i] = np.random.uniform(state_bounds[key_list[j]][0],state_bounds[key_list[j]][1])
            else:
                Koopman_testing_trajectory[j,0,i] = np.random.uniform(state_bounds[key_list[j]][0],state_bounds[key_list[j]][1])

    if Generate_Koopman_data:
        if dynamics_model == 'vdp':
            for i in range(0,training_number):
                _, Koopman_training_trajectory[0:state_dimension,:,i] = simulate_vdp(mu, simulation_time, iteration_len=iteration_len, init_state = Koopman_training_trajectory[0:state_dimension,0,i])
            for i in range(0,testing_number):
                _, Koopman_testing_trajectory[0:state_dimension,:,i] = simulate_vdp(mu, simulation_time, iteration_len=iteration_len, init_state = Koopman_testing_trajectory[0:state_dimension,0,i])
            with open("datas/Koopman_data_vdp"+str(training_number)+".pkl", "wb") as f:
                pickle.dump([Koopman_training_trajectory, Koopman_testing_trajectory], f)
        else:
            print("Under construction")

    else:
        if dynamics_model == 'vdp':
            with open("datas/Koopman_data_vdp"+str(training_number)+".pkl", 'rb') as f:
                Koopman_training_trajectory, Koopman_testing_trajectory = pickle.load(f)
        else:
            print("Under construction")

    return Koopman_training_trajectory, Koopman_testing_trajectory, state_dimension, input_dimension, state_bounds

# Van der pol simulation ----------------------------------
def van_der_pol(t, state, mu):
    x, y = state
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]

def simulate_vdp(mu, T, iteration_len=100, init_state=None):
    t_span = (0, T)
    t_eval = np.linspace(0, T, iteration_len)
    sol = solve_ivp(van_der_pol, t_span, init_state, args=(mu,), t_eval=t_eval)
    return sol.t, sol.y