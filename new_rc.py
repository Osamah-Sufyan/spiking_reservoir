import numpy as np
import matplotlib.pyplot as plt
import json

class InputConstructor:
    def __init__(self, no_virtual_nodes, line_width, x_data):
        self.no_virtual_nodes = no_virtual_nodes
        self.line_width = line_width
        self.x_data = x_data
        self.random_heights = np.random.choice([0, 1], no_virtual_nodes)
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
    def get_mask_value(self, index):
        node_index = (index // self.line_width) % self.no_virtual_nodes
        return self.random_heights[node_index]
        
    def construct_input(self, current_step):
        x_index = current_step // (self.line_width * self.no_virtual_nodes)
        if x_index >= len(self.x_data):
            x_index = len(self.x_data) - 1
            
        x_value = self.x_data[x_index]
        mask_value = self.get_mask_value(current_step)
        value = mask_value * x_value
        
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        
        return value

    def normalize_value(self, value, target_min=0, target_max=56e-6):
        if self.max_val == self.min_val:
            return target_min
        normalized = (value - self.min_val) / (self.max_val - self.min_val)
        return normalized * (target_max - target_min) + target_min

class DifferentialEquationSolver:
    def __init__(self, params, input_constructor):
        # System parameters
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
        self.J = params['J']
        self.N0 = params['N0']
        self.tau_p = params['tau_p']
        
        self.input_constructor = input_constructor
        self.time_step = params['time_step']
        
        # Initialize state matrix storage
        self.state_matrix = []
        self.voltage_threshold = 3.7  # Spike threshold

    def f(self, V):
        term1 = (1 + np.exp(self.q_e * (self.b - self.c + self.n1 * V) / (self.k_B * self.T))) / \
                (1 + np.exp(self.q_e * (self.b - self.c - self.n1 * V) / (self.k_B * self.T)))
        term2 = np.pi / 2 + np.arctan((self.c - self.n1 * V) / self.d)
        term3 = self.h * (np.exp(self.q_e * self.n2 * V / (self.k_B * self.T)) - 1)
        return self.a * np.log(term1) * term2 + term3

    def Vm(self, t):
        current_step = int(t / self.time_step)
        input_value = self.input_constructor.construct_input(current_step)
        return self.input_constructor.normalize_value(input_value)

    def system_derivs(self, y, t):
        V, I = y
        fV = self.f(V)
        dVdt = (I + self.Vm(t) - fV) / self.mu
        dIdt = self.mu * (2.26 - V - self.R * I)
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
        
        # For state matrix collection
        steps_per_mask = self.input_constructor.line_width
        current_mask_voltages = []
        mask_counter = 0
        current_row = []  # To store binary values for current time step
        
        for step in range(num_steps):
            results.append(y)
            I_values.append(y[1])
            V_values.append(y[0])
            
            # Collect voltage values within current mask
            current_mask_voltages.append(y[0])
            
            # Check if we're at the end of a mask period
            if step % steps_per_mask == steps_per_mask - 1:
                # Check if voltage exceeded threshold during this mask period
                spike_occurred = int(max(current_mask_voltages) > self.voltage_threshold)
                current_row.append(spike_occurred)
                current_mask_voltages = []  # Reset for next mask
                mask_counter += 1
                
                # If we've collected states for all virtual nodes
                if mask_counter == self.input_constructor.no_virtual_nodes:
                    self.state_matrix.append(current_row)
                    current_row = []
                    mask_counter = 0
            
            y = self.rk4_step(y, t, dt)
            t += dt
            
        return np.array(results), np.array(I_values), np.array(V_values), np.array(self.state_matrix)

def main():
    # Load Mackey-Glass data
    with open('../mackey_glass_data.json', 'r') as jsonfile:
        data = json.load(jsonfile)
        t = np.array(data['t'])
        x = np.array(data['y'])

    # Parameters
    t_end_mg = 40
    x = x[:t_end_mg]
    t = t[:t_end_mg]
    
    # Normalize Mackey-Glass data
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    
    # Input construction parameters
    no_virtual_nodes = 10
    line_width = 1000
    time_step = 0.01
    
    # Calculate simulation length
    total_steps = len(x) * no_virtual_nodes * line_width
    t_end = total_steps * time_step
    
    # Initialize input constructor
    input_constructor = InputConstructor(no_virtual_nodes, line_width, x)
    
    # System parameters
    params = {
        'a': 0.0039, 'q_e': 1.6e-19, 'k_B': 1.38e-23, 'T': 300,
        'b': 0.05, 'c': 0.0874, 'd': 0.0073, 'n1': 0.0352, 'n2': 0.0031, 'h': 0.0367,
        'R': 0.1, 'kappa': 1.1e-7, 'gamma_m': 1e7, 'gamma_l': 1e9, 'gamma_nr': 2e9,
        'eta': 1, 'J': 210e-6, 'N0': 5e5, 'tau_p': 5e-13, 'mu': 0.001,
        'time_step': time_step
    }
    
    # Initialize solver and run simulation
    initial_conditions = [2.26, 0.028]
    solver = DifferentialEquationSolver(params, input_constructor)
    results, I_values, V_values, state_matrix = solver.solve(initial_conditions, 0, t_end, time_step)
    
    # Print state matrix information
    print(f"State Matrix Shape: {state_matrix.shape}")
    print(f"Number of time steps: {len(state_matrix)}")
    print(f"Number of virtual nodes per time step: {len(state_matrix[0])}")
    
    # Plot results including binary state matrix visualization
    plt.figure(figsize=(15, 10))
    
    # Voltage time series
    plt.subplot(2, 1, 1)
    plt.plot(V_values)
    plt.axhline(y=3.7, color='r', linestyle='--', label='Spike Threshold')
    plt.title('Voltage over time')
    plt.xlabel('Time step')
    plt.ylabel('Voltage')
    plt.legend()
    
    # Binary state matrix visualization
    plt.subplot(2, 1, 2)
    plt.imshow(state_matrix.T, aspect='auto', cmap='binary')
    plt.title('Binary State Matrix (White = Spike)')
    plt.xlabel('Time step')
    plt.ylabel('Virtual node')
    plt.colorbar(label='Spike occurred')
    
    plt.tight_layout()
    plt.savefig('spiking_rc.png')
    
    # Print some statistics
    spike_rate = np.mean(state_matrix)
    print(f"Overall spike rate: {spike_rate:.2%}")
    print(f"Total number of spikes: {np.sum(state_matrix)}")

if __name__ == "__main__":
    main()