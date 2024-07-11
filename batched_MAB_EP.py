import random
import sys
import os 
import numpy as np
import subprocess
from subprocess import PIPE
import re
import time

genlib_origin = sys.argv[-1]
lib_origin = genlib_origin[:-7] + '.lib'
design = sys.argv[-2]
sample_gate = int(sys.argv[-3])
temp_blif = "/export/mliu9867/LibGame/temp_blifs_" + genlib_origin[:-7] + "/" + design[:-5] + "_temp.blif"
lib_path = "/export/mliu9867/LibGame/gen_newlibs_" + genlib_origin[:-7] + "/"

start=time.time()
abc_cmd = "read %s;read %s; map -a; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (genlib_origin, design, temp_blif, lib_origin, temp_blif)
res = subprocess.check_output(('abc', '-c', abc_cmd))
match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
# Baseline
max_delay = float(match_d.group(1))
max_area = float(match_a.group(1))

print("Baseline Delay:", max_delay)
print("Baseline Area:", max_area)

# Mapper call
def technology_mapper(genlib_origin, partial_cell_library):
    with open(genlib_origin, 'r') as f:
        f_lines = [line.strip() for line in f if line.startswith("GATE") and not any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
        #f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    f.close()
    with open(genlib_origin, 'r') as f:
        f_keep = [line.strip() for line in f if any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
        #f_keep = [line.strip() for line in f if line.startswith("GATE BUF") or line.startswith("GATE INV") or line.startswith("GATE sky130_fd_sc_hd__buf") or line.startswith("GATE sky130_fd_sc_hd__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    f.close()
    lines_partial = [f_lines[i] for i in partial_cell_library]
    lines_partial = lines_partial + f_keep

    output_genlib_file = lib_path + design + "_" + str(len(lines_partial)) + "_samplelib.genlib"
    with open(output_genlib_file, 'w') as out_gen:
        for line in lines_partial:
            out_gen.write(line + '\n')
    out_gen.close() 

    abc_cmd = "read %s;read %s; map -a; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (output_genlib_file, design, temp_blif, lib_origin, temp_blif)
    res = subprocess.check_output(('abc', '-c', abc_cmd))
    match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
    match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
    if match_d and match_a:
        delay = float(match_d.group(1))
        area = float(match_a.group(1))
    else:
        delay, area = float("NaN"),float("NaN")
    return delay, area

# Reward calculation
def calculate_reward(max_delay, max_area, delay, area):
    normalized_delay = delay / max_delay
    normalized_area = area / max_area

    return -np.sqrt(normalized_delay * normalized_area) 

class EpsilonGreedyMAB:
    def __init__(self, num_arms, epsilon, sample_gate, batch_size):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.q_values = [0.0] * num_arms
        self.counts = [0] * num_arms
        self.sample_gate = sample_gate
        self.batch_size = batch_size

    def select_batch_actions(self):
        batches = []
        for _ in range(self.batch_size):
            selected_cells = set()
            while len(selected_cells) < self.sample_gate:
                if random.random() > self.epsilon:
                    select = np.argmax(self.q_values)
                else:
                    select = random.randint(0, self.num_arms - 1)
                selected_cells.add(select)
            batches.append(list(selected_cells))
        return batches

    def update_batch(self, batch_actions, rewards):
        for selected_arm, reward in zip(batch_actions, rewards):
            for arm in selected_arm:
                self.counts[arm] += 1
                self.q_values[arm] = (self.q_values[arm] * (self.counts[arm] - 1) + reward) / self.counts[arm]

# Main batched MAB loop
batch_size = 10 


# Initialization
num_cells_select = sample_gate
with open(genlib_origin, 'r') as f:
        f_lines = [line.strip() for line in f if line.startswith("GATE") and not any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
        #f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
f.close()
num_arms=len(f_lines)
mab = EpsilonGreedyMAB(num_arms, 0.2, sample_gate, batch_size)
best_cells = None
best_result = (float('inf'), float('inf'))  
best_reward = -float('inf')  # Track best reward

# Main Loop
num_iterations = 1000  

for i in range(num_iterations):
    print(f"Batch iteration: {i}")
    batch_actions = mab.select_batch_actions()
    batch_rewards = []
    for selected_cells in batch_actions:
        delay, area = technology_mapper(genlib_origin, selected_cells)
        if np.isnan(delay) or np.isnan(area):
            reward = -float('inf')
        else:
            reward = calculate_reward(max_delay, max_area, delay, area)
        if reward > best_reward:
            best_reward = reward
            best_result = (delay, area)
            best_cells = selected_cells

        batch_rewards.append(reward)
    print("Current best reward: ", best_reward)
    print("Current best result: ", best_result)
        # Update best results tracking here as needed
    mab.update_batch(batch_actions, batch_rewards)
end=time.time()
runtime=end-start

#print("Best Cells:", best_cells)
print("Best Delay:", best_result[0])
print("Best Area:", best_result[1])
print("Best Reward:", best_reward)
print("Total time:", runtime)
