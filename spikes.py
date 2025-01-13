import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import json




with open('../mackey_glass_data.json', 'r') as jsonfile:
    data = json.load(jsonfile)
    t = np.array(data['t'])
    x = np.array(data['y'])

t_end_mg = 2000
x = x[:t_end_mg]
t = t[:t_end_mg]


print(len(t))

class DifferentialEquationSolver:
    def __init__(self, params):
        self.a = params['a']
        self.q_e = params['q_e']
        self.k_B = params['k_B']
        self.T = params['T']
        self.b = params['b']
        self.c = params['c']
        self.d = params['d']
        self.n1 = params['n1']
        self.n2 = params['n2']
        self.h = params['h']
        self.R = params['R']
        self.mu = params['mu']
        self.kappa = params['kappa']
        self.gamma_m = params['gamma_m']
        self.gamma_l = params['gamma_l']
        self.gamma_nr = params['gamma_nr']
        self.eta = params['eta']
        self.q_e = params['q_e']
        self.J = params['J']
        self.N0 = params['N0']
        self.tau_p = params['tau_p']
        self.V0 = params['V0']
        self.len_V0 = params['len_V0']
    # def Vm(self,V0, t):
    #     if 0.05e-9 <= t <= 0.1e-9:  # replace t_start and t_end with your interval
    #         return V0 # constant amplitude pulse
    #     else:
    def Vm(self, V0, t):
        dt = time_step
        index = int(t / dt)
        if index < 0:
            index = 0
        elif index >= len(V0):
            index = len(V0) - 1
        return V0[index]
        

    def f(self, V):
        term1 = (1 + np.exp(self.q_e * (self.b - self.c + self.n1 * V) / (self.k_B * self.T))) / \
                (1 + np.exp(self.q_e * (self.b - self.c - self.n1 * V) / (self.k_B * self.T)))
        term2 = np.pi / 2 + np.arctan((self.c - self.n1 * V) / self.d)
        term3 = self.h * (np.exp(self.q_e * self.n2 * V / (self.k_B * self.T)) - 1)
        return self.a * np.log(term1) * term2 + term3

    def system_derivs(self, y, t):
        V, I = y
        term1 = (1 + np.exp(self.q_e * (self.b - self.c + self.n1 * V) / (self.k_B * self.T))) / \
                (1 + np.exp(self.q_e * (self.b - self.c - self.n1 * V) / (self.k_B * self.T)))
        term2 = np.pi / 2 + np.arctan((self.c - self.n1 * V) / self.d)
        term3 = self.h * (np.exp(self.q_e * self.n2 * V / (self.k_B * self.T)) - 1)
        fV = self.a * np.log(term1) * term2 + term3
        V, I = y
        dVdt = (I+self.Vm(self.V0, t) - fV )/(self.mu)# Simplified m(t)
        dIdt = self.mu *(2.26 - V - self.R * (I)) # Simplified Vm(t)
        #dSdt = (self.gamma_m * (N - self.N0) - 1 / self.tau_p) * S + self.gamma_m * N +np.sqrt(self.gamma_m*N*S) * np.random.normal(0, 1)
        #dNdt = (self.J + self.eta * I) / self.q_e - (self.gamma_l + self.gamma_m + self.gamma_nr) * N - self.gamma_m * (N - self.N0) * S
        return np.array([dVdt, dIdt])

    def rk4_step(self, y, t, dt):
        k1 = dt * self.system_derivs(y, t)
        k2 = dt * self.system_derivs(y + 0.5 * k1, t + 0.5 * dt)
        k3 = dt * self.system_derivs(y + 0.5 * k2, t + 0.5 * dt)
        k4 = dt * self.system_derivs(y + k3, t + dt)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def solve(self, y0, t0, t_final, dt):
        num_steps = int((t_final - t0) / dt)
        
        t = t0
        y = y0
        results = []
        I_values = []
        V_values = []
        for _ in range(num_steps):
            results.append(y)
            I_values.append(y[1]) 
            V_values.append(y[0])  
            y = self.rk4_step(y, t, dt)
            t += dt
        return np.array(results), np.array(I_values), np.array(V_values)

    def plot_fV(self, V_min, V_max):
        V_values = np.linspace(V_min, V_max, 400)
        fV_values = [self.f(V) for V in V_values]
        plt.figure(figsize=(10, 5))
        plt.plot(V_values, fV_values, label='f(V)')
        plt.xlabel('Voltage V')
        plt.ylabel('f(V)')
        plt.title('Plot of f(V)')
        plt.legend()
        plt.grid(True)
        plt.show()
    def count_unique_maxima(self, data, tolerance=1e-7):
        unique_maxima = np.array([])
        if np.all(np.isclose(data, data[0], atol=tolerance)):
            num_maxima = 0
        else:
            max_indices = np.where((np.roll(data, 1) < data) & (np.roll(data, -1) < data))[0]
            
            max_indices = max_indices[(max_indices != 0) & (max_indices != len(data) - 1)]
            unique_maxima = np.unique(np.round(data[max_indices], int(np.ceil(-np.log10(tolerance)))))
        num_maxima = len(unique_maxima)
        return num_maxima, unique_maxima
    def count_unique_minima(self, data, tolerance=1e-7):
        unique_minima = np.array([])
        if np.all(np.isclose(data, data[0], atol=tolerance)):
            num_minima = 0
        else:
            min_indices = np.where((np.roll(data, 1) > data) & (np.roll(data, -1) > data))[0]
           
            min_indices = min_indices[(min_indices != 0) & (min_indices != len(data) - 1)]
            unique_minima = np.unique(np.round(data[min_indices], int(np.ceil(-np.log10(tolerance)))))
        num_minima = len(unique_minima)
        return num_minima, unique_minima


with open('../mackey_glass_data.json', 'r') as jsonfile:
    data = json.load(jsonfile)
    t = np.array(data['t'])
    x = np.array(data['y'])


x = x[:t_end_mg]
t = t[:t_end_mg]

#x = (x - np.min(x)) / (np.max(x) - np.min(x))
buffer_data_1_t = t[:round(0.05*len(t))]
buffer_data_1_y = x[:round(0.05*len(t))]
train_data_t = t[round(0.05*len(t)):round(0.55*(len(t)))]
train_data_y = x[round(0.05*len(t)):round(0.55*(len(t)))]
buffer_data_2_t = t[round(0.55*(len(t))):round(0.7*(len(t)))]
buffer_data_2_y = x[round(0.55*(len(t))):round(0.7*(len(t)))]
test_data_t = t[round(0.7*(len(t))):round(0.95*(len(t)))]
test_data_y = x[round(0.7*(len(t))):round(0.95*(len(t)))]
buffer_data_3_t = t[round(0.95*(len(t))):]
buffer_data_3_y = x[round(0.95*(len(t))):]
print(f"total divisions: ",len(buffer_data_1_t)+len(train_data_t)+len(buffer_data_2_t)+len(test_data_t))
x_min = np.min(x)-0.1
x_max = np.max(x)+0.1
no_virtual_nodes = 100
random_heights = np.random.choice([0, 1], no_virtual_nodes)  # Randomly choose between 0 and 1
int_len = t[1]-t[0]
# mask = np.linspace(0, 1, int((t_end//time_step/len(x))+1)  )
mask = np.linspace(0, 1,  no_virtual_nodes* 100 )


print(f"number of Lorenz time steps: ",len(t))


h = np.zeros(len(mask)) 
line_width = int(len(mask)/no_virtual_nodes)
print(f"len(mask):",len(mask))
print(f"line width:",line_width)

for j, height in enumerate(random_heights):
    x_start = int((j) * line_width)
    x_end = int((j + 1) * line_width)
    if x_end > len(h):
        x_end = len(h)
    
    h[x_start:x_end] = height

num_raw_data = len(x)
h_rep = np.tile(h, num_raw_data)
h_repeated = np.repeat(h[np.newaxis, :], num_raw_data, axis=0)

h_input = h_repeated

for i in range(num_raw_data):
    h_input[i] = h_repeated[i] * x[i] 
print(h_input.shape)

h_input_flattened = h_input.flatten() 
print(f"length of h_input_flattened: ",len(h_input_flattened))
# plt.plot(h_input_flattened)
# plt.show()


print(len(h_input_flattened))
step_indices = [i * line_width for i in range(0, num_raw_data * no_virtual_nodes)]



print(f"number step indices: ",len(step_indices))

#normalize h_input_flattened
h_input_flattened = (h_input_flattened - np.min(h_input_flattened)) / (np.max(h_input_flattened) - np.min(h_input_flattened))






print(len(h_input_flattened))

#normalize h_input_flattened
#h_input_flattened = 2*(h_input_flattened - np.min(h_input_flattened)) / (np.max(h_input_flattened) - np.min(h_input_flattened)) -1
h_input_flattened = h_input_flattened

print(f"length of inputs flattened: ",len(h_input_flattened))
# normalize inputs between 10e-6 and 30e-6

#normalized_inputs = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs)) 
with open('../mackey_glass_data.json', 'r') as jsonfile:
    data = json.load(jsonfile)
    t_delayed = np.array(data['t'])
    x_delayed = np.array(data['y'])    
x_dealyed = x_delayed[10:t_end_mg+10]
t_delayed = t_delayed[10:t_end_mg+10]
# normalize x_delayed
x_dealyed = (x_dealyed - np.min(x_dealyed)) / (np.max(x_dealyed) - np.min(x_dealyed))
random_heights_d = np.random.uniform(0, 1, no_virtual_nodes)
int_len = t_delayed[1]-t_delayed[0]
#mask = np.linspace(0, 1, int((t_end//time_step/len(x))+1)  )
mask = np.linspace(0, 1,  5000 )




h_d = np.zeros(len(mask)) 
line_width = len(mask)/no_virtual_nodes


for j, height in enumerate(random_heights_d):
    x_start = int(j * line_width)
    x_end = int(x_start + line_width)
    for i in range(x_start, x_end):
        h_d[i] = height

h_rep_d = np.tile(h_d, num_raw_data)
h_repeated_d = np.repeat(h_d[np.newaxis, :], num_raw_data, axis=0)
h_input_d = h_repeated_d
for i in range(num_raw_data):
    h_input_d[i] = h_repeated_d[i] * x_dealyed[i] 

h_input_flattened_d = h_input_d.flatten() 
  
x_repeated_d = np.linspace(t_delayed[0] - int_len, t_delayed[0] + num_raw_data * int_len, len(h_input_flattened_d))
print(len(h_input_flattened))
inputs =  h_input_flattened #+ h_input_flattened_d 
inputs =  (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs)) * (56e-6)
V0 = inputs

print(f"max inputs: ",max(inputs))
print(f"min inputs: ",min(inputs))

len_V0 = len(V0)
time_step = 0.01
t_end = len_V0 * time_step
print(f"number of V0 time steps: ",len(h_input_flattened))
params = {
    'a': 0.0039, 'q_e': 1.6e-19, 'k_B': 1.38e-23, 'T': 300,
    'b': 0.05, 'c': 0.0874, 'd': 0.0073, 'n1': 0.0352, 'n2': 0.0031, 'h': 0.0367,
    'R': 0.1, 'kappa': 1.1e-7, 'gamma_m': 1e7, 'gamma_l': 1e9, 'gamma_nr': 2e9,
    'eta': 1, 'J': 210e-6, 'N0': 5e5, 'tau_p': 5e-13, 'mu': 0.001, 'V0': V0, 'len_V0': len_V0
}



initial_conditions = [2.26, 0.028]

solver = DifferentialEquationSolver(params)
results, I_values, V_values = solver.solve(initial_conditions, 0, t_end, time_step)

print("time total: ", t_end)    
plt.figure(figsize=(10, 5))
plt.subplot2grid((2, 1), (0, 0), rowspan=1)
plt.plot(V0)
# plot horizontal line at 2.8e-6
plt.axhline(y=3.4e-5, color='r', linestyle='--')
plt.xlabel(r't', fontsize = 17)
plt.ylabel(r'$I[A]$', fontsize = 17)
#plt.xticks([0, 400/time_step, 800/time_step, 1200/time_step, 1600/time_step, 2000/time_step], ['0', '400' ,'800', '1200', '1600', '2000'])
#plt.yticks([0,10e-6,0.75], [r'0', r'700', r'750'])
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)  # Set y-axis tick label size
plt.title('V')
plt.grid(True)
plt.subplot2grid((2, 1), (1, 0), rowspan=1)
plt.plot(V_values)
plt.xlabel(r't', fontsize = 17)
plt.ylabel(r'V[V]', fontsize = 17)
#plt.xticks([0, 400/time_step, 800/time_step, 1200/time_step, 1600/time_step, 2000/time_step], ['0', '400' ,'800', '1200', '1600', '2000'])
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)  # Set y-axis tick label size
plt.title('I')
plt.grid(True)
plt.tight_layout()
plt.savefig('many_nodes_binary_longer.pdf')
plt.show()


I_values = V_values
def count_maxima_larger_than_threshold(I_values, step_indices, line_width, threshold=2.6):
    maxima_counts = []
    for i in step_indices:
        start_index = max(0, i - line_width)
        end_index = i
        interval = I_values[start_index:end_index]
        maxima_indices = np.where((np.roll(interval, 1) < interval) & (np.roll(interval, -1) < interval))[0]
        maxima_values = interval[maxima_indices]
        count = np.sum(maxima_values > threshold)
        maxima_counts.append(count)
    return maxima_counts
print(f"number of step indices: ", len(step_indices))
step_indices_training = step_indices[round(0.05*len(step_indices)):round(0.55*len(step_indices))]

I_values_training_read = count_maxima_larger_than_threshold(I_values, step_indices_training, int(line_width))

#print("I_values_training_read:", I_values_training_read)




step_indices_testing = step_indices[round(0.7*len(step_indices)):round(0.95*len(step_indices))]
print(f"length of I_train reads", len(I_values_training_read))
print(f"length of train data",len(train_data_t))



S_matrix_training = np.array(I_values_training_read).reshape(len(train_data_t), no_virtual_nodes)
ones_column_training = np.ones((S_matrix_training.shape[0], 1))
S_matrix_training = np.hstack((S_matrix_training, ones_column_training))
S_training_1d = S_matrix_training.ravel()

print(f"length of training steps: ", len(step_indices_training))
print(f"length of testing steps: ", len(step_indices_testing))
I_values_testing_read = count_maxima_larger_than_threshold(I_values, step_indices_testing, int(line_width))
print(f"length of I_test reads", len(I_values_testing_read))
S_matrix_testing = np.array(I_values_testing_read).reshape(len(test_data_t), no_virtual_nodes)
ones_column_testing = np.ones((S_matrix_testing.shape[0], 1))

S_matrix_testing = np.hstack((S_matrix_testing, ones_column_testing))
S_testing_1d = S_matrix_testing.ravel()


# no_unique_maxima, unique_maxima = solver.count_unique_maxima(I_values)
# print(f'Number of unique maxima: {no_unique_maxima}')
# print(f'Unique maxima: {unique_maxima}')

alpha_values = np.logspace(-5, 5, 100)
steps_ahead = 10 + 10
target_vector_train = x[round(0.05 * len(t)) + steps_ahead:round(0.55 * len(t)) + steps_ahead]
target_vector_test = x[round(0.7 * len(t)) + steps_ahead:round(0.95 * len(t)) + steps_ahead]

nrmse_train = []
nrmse_test = []

# Calculate NRMSE for different alpha values
for alpha in alpha_values:
    S_transpose_S = np.matmul(np.transpose(S_matrix_training), S_matrix_training)
    identity_matrix = np.identity(S_transpose_S.shape[0])
    STS_p_lambda_I = S_transpose_S + alpha * identity_matrix

    inverse = np.linalg.inv(STS_p_lambda_I)
    weights = np.matmul(np.matmul(inverse, np.transpose(S_matrix_training)), target_vector_train)
    predictions_train = np.matmul(S_matrix_training, weights)
    predictions_test = np.matmul(S_matrix_testing, weights)

    # Calculate NRMSE for training data
    sum_1 = sum((target_vector_train[i] - predictions_train[i]) ** 2 for i in range(len(target_vector_train)))
    NRMSE_train = np.sqrt(sum_1 / (len(target_vector_train) * np.var(target_vector_train)))
    nrmse_train.append(NRMSE_train)

    # Calculate NRMSE for testing data
    sum_2 = sum((target_vector_test[i] - predictions_test[i]) ** 2 for i in range(len(target_vector_test)))
    NRMSE_test = np.sqrt(sum_2 / (len(target_vector_test) * np.var(target_vector_test)))
    nrmse_test.append(NRMSE_test)

# Plot NRMSE vs. alpha
plt.figure(figsize=(10, 6))
plt.plot(alpha_values, nrmse_train, label='NRMSE Train')
plt.plot(alpha_values, nrmse_test, label='NRMSE Test')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('NRMSE')
plt.title('NRMSE vs. Alpha')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('nrmse_vs_alpha.pdf')
plt.show()











def process_V0(V0, initial_conditions ):
    params = {
        'a': 0.0039, 'q_e': 1.6e-19, 'k_B': 1.38e-23, 'T': 300,
        'b': 0.05, 'c': 0.0874, 'd': 0.0073, 'n1': 0.0352, 'n2': 0.0031, 'h': 0.0367,
        'R': 0.1, 'kappa': 1.1e-7, 'gamma_m': 1e7, 'gamma_l': 1e9, 'gamma_nr': 2e9,
        'eta': 1, 'J': 210e-6, 'N0': 5e5, 'tau_p': 5e-13, 'mu': 0.001, 'V0': V0
    }
    solver = DifferentialEquationSolver(params)

    V_values, I_values = solver.solve(initial_conditions, 0, 500, 0.005)
    #print(initial_conditions)
    
    # Consider only the last 30% of the data to avoid transients
    I_values = I_values[int(9*len(I_values)/10):]

   
    
    # Find the unique maximaå
    no_maxim, unique_maxima = solver.count_unique_maxima(I_values)
    no_minim, unique_minima = solver.count_unique_minima(I_values)
    
    # Get the maximum amplitude
    if unique_maxima.size > 0:
        max_amplitude = max(unique_maxima)
    else:
        max_amplitude = I_values[-1]

    if unique_minima.size > 0:
        min_amplitude = min(unique_minima)
    else:
        min_amplitude = I_values[-1]
    
    #print(min_amplitude)
    #print(initial_conditions)
    print(max_amplitude)
    
    return max_amplitude, no_maxim, min_amplitude, no_minim, [V_values[-1], I_values[-1]]





