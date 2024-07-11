import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess
import re
import sys
import random
from collections import deque
import time

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class GateSelectionEnv(gym.Env):
    """Gate selection environment for reinforcement learning"""
    metadata = {'render.modes': ['human']}

    def __init__(self, genlib_origin, lib_path, design, total_gates, sample_gate, max_delay, max_area):
        super().__init__()
        self.genlib_origin = genlib_origin
        self.lib_path = lib_path
        self.design = design
        self.total_gates = total_gates
        self.sample_gate = sample_gate
        self.max_delay = max_delay
        self.max_area = max_area
        self.action_space = spaces.Discrete(self.total_gates)  # Select one gate at a time
        self.observation_space = spaces.MultiBinary(self.total_gates)  # Binary flags for each gate
        self.state = np.zeros(self.total_gates, dtype=int)
        self.selection_count = 0

    def step(self, action):
        if self.state[action] == 0 and self.selection_count < self.sample_gate:
            self.state[action] = 1
            self.selection_count += 1

        done = self.selection_count == self.sample_gate
        reward = 0
        delay = 1
        area = 1
        next_state = self.state.copy()

        if done:
            # Evaluate the selected gates only once all required selections are made
            delay, area = self.technology_mapper(list(np.where(self.state == 1)[0]))
            reward = self.calculate_reward(delay, area)
            next_state = self.reset()  # Get the new state after reset for the next episode
        return next_state, reward, done, delay, area

    def technology_mapper(self, partial_cell_library):
        # Read the original library file and filter gates
        with open(self.genlib_origin, 'r') as f:
            f_lines = [line.strip() for line in f if line.startswith("GATE") and not any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
            #f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
        f.close()
        with open(self.genlib_origin, 'r') as f:
            f_keep = [line.strip() for line in f if any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
            #f_keep = [line.strip() for line in f if line.startswith("GATE BUF") or line.startswith("GATE INV") or line.startswith("GATE sky130_fd_sc_hd__buf") or line.startswith("GATE sky130_fd_sc_hd__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
        f.close()

        # Generate the partial library for the selected gates
        #print(len(f_lines))
        #print(len(partial_cell_library))
        lines_partial = [f_lines[i] for i in partial_cell_library] + f_keep
        output_genlib_file = self.lib_path + self.design + "_" + str(len(lines_partial)) + "_samplelib.genlib"
        lib_origin = self.genlib_origin[:-7] + '.lib'
        temp_blif = "/export/mliu9867/LibGame/test_dqn_" + self.genlib_origin[:-7] + "/temp_blifs/" + self.design[:-5] + "_temp.blif"
        with open(output_genlib_file, 'w') as out_gen:
            for line in lines_partial:
                out_gen.write(line + '\n')

        # Execute the mapping command using ABC
        abc_cmd = f"abc -c 'read {output_genlib_file}; read {self.design}; map -a; write {temp_blif}; read {lib_origin}; read -m {temp_blif}; ps; topo; upsize; dnsize; stime;'"
        try:
            res = subprocess.check_output(abc_cmd, shell=True, text=True)
            match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", res)
            match_a = re.search(r"Area\s*=\s*([\d.]+)", res)
            delay = float(match_d.group(1)) if match_d else float('inf')
            area = float(match_a.group(1)) if match_a else float('inf')
        except subprocess.CalledProcessError as e:
            print("Failed to execute ABC:", e)
            delay, area = float('inf'), float('inf')

        return delay, area

    def calculate_reward(self, delay, area):
        if delay == float('inf') or area == float('inf'):
            return float('-inf')
        normalized_delay = delay / self.max_delay
        normalized_area = area / self.max_area
        return -np.sqrt(normalized_delay * normalized_area)

    def reset(self):
        self.state = np.zeros(self.total_gates, dtype=int)
        self.selection_count = 0
        return self.state

    def render(self, mode='human'):
        print(f"Selected Gates: {np.where(self.state == 1)[0]}")

    def close(self):
        pass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        # Define the network layers
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        # Pass the state through the network
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = 0.99

    def select_action(self, state, epsilon=0.2):
        if np.random.rand() < epsilon:
            return np.random.randint(0, len(state))
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.model(state)
                return q_values.argmax().item()
    def update_batch(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)  # Action indices need to be in a column for gather
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # Current Q values from model's prediction
        current_qs = self.model(states).gather(1, actions).squeeze(1)
        
        # Next Q values from model's prediction on next states
        next_qs = self.model(next_states).max(1)[0]  # Get max Q value for each next state
        expected_qs = rewards + self.gamma * (1 - dones) * next_qs  # Bellman update rule

        # Compute loss
        loss = F.mse_loss(current_qs, expected_qs)
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train_agent(num_episodes, agent, env, batch_size, buffer_size):
    replay_buffer = deque(maxlen=buffer_size)
    highest_reward = float('-inf')

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, delay, area = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            if done and reward > highest_reward:
                highest_reward = reward
                best_result = (delay, area)
                print('Current Best Result: ', best_result)

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                agent.update_batch(batch)  # Process batch update

        print(f"Episode {episode + 1}, Highest Reward = {highest_reward}")


genlib_origin = sys.argv[-1]
lib_origin = genlib_origin[:-7] + '.lib'
#print(lib_origin)
design = sys.argv[-2]
sample_gate = int(sys.argv[-3])
temp_blif = "/export/mliu9867/LibGame/test_dqn_" + genlib_origin[:-7] + "/temp_blifs/" + design[:-5] + "_temp.blif"
lib_path = "/export/mliu9867/LibGame/test_dqn_" + genlib_origin[:-7] + "/gen_newlibs/"
#lib_origin = '7nm.lib'

#genlib_origin = '7nm.genlib'
#lib_path = 'test_dqn/'
#design = 'bfly.abc.blif'
abc_cmd = "read %s;read %s; map -a; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (genlib_origin, design, temp_blif, lib_origin, temp_blif)
res = subprocess.check_output(('abc', '-c', abc_cmd))
match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
# Baseline
max_delay = float(match_d.group(1))
max_area = float(match_a.group(1))
print('Baseline Delay: ', max_delay)
print('Baseline Area: ', max_area)
with open(genlib_origin, 'r') as f:
        f_lines = [line.strip() for line in f if line.startswith("GATE") and not any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
        #f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
f.close()
total_gates = len(f_lines)
#sample_gate = 80
state_size = total_gates
action_size = total_gates
num_episodes = 5000
batch_size = 10
buffer_size = 10000
env = GateSelectionEnv(genlib_origin, lib_path, design, total_gates, sample_gate, max_delay, max_area)
agent = DQNAgent(state_size, action_size)
#train_agent(num_episodes=100, agent=agent, env=env, sample_gate=sample_gate)
start=time.time()
train_agent(num_episodes, agent, env, batch_size, buffer_size)
end=time.time()
runtime=end-start
print('Total time: ', runtime)
