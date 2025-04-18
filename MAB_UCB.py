import random
import math
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
temp_blif = "temp_blifs/" + design[:-5] + "_ucb_temp.blif"
lib_path = "gen_newlibs/"

start=time.time()
abc_cmd = "read %s;read %s; map; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (genlib_origin, design, temp_blif, lib_origin, temp_blif)
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
        #f_lines = [line.strip() for line in f if line.startswith("GATE") and not any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
        f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    f.close()
    with open(genlib_origin, 'r') as f:
        #f_keep = [line.strip() for line in f if any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
        f_keep = [line.strip() for line in f if line.startswith("GATE BUF") or line.startswith("GATE INV") or line.startswith("GATE sky130_fd_sc_hd__buf") or line.startswith("GATE sky130_fd_sc_hd__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    f.close()
    lines_partial = [f_lines[i] for i in partial_cell_library]
    lines_partial = lines_partial + f_keep
    output_genlib_file = lib_path + design + "_" + str(len(lines_partial)) + "_ucb_samplelib.genlib"
    with open(output_genlib_file, 'w') as out_gen:
        for line in lines_partial:
            out_gen.write(line + '\n')
    out_gen.close() 

    abc_cmd = "read %s;read %s; map; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (output_genlib_file, design, temp_blif, lib_origin, temp_blif)
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

# UCB MAB Class
class UCB_MAB:
  def __init__(self, num_arms, c, sample_gate):
    self.num_arms = num_arms
    self.c = c  # Exploration parameter for UCB
    self.q_values = [0.0] * num_arms
    self.counts = [0] * num_arms
    self.sample_gate = sample_gate
  
  def select_action(self):
    selected_cells = set()

    # Exploration (ensure each arm is tried at least once)
    for arm in range(self.num_arms):
        if self.counts[arm] == 0:
            selected_cells.add(arm)
            if len(selected_cells) == self.sample_gate:
                break

    # Exploitation (choose the remaining based on UCB)
    remaining_cells = [arm for arm in range(self.num_arms) if arm not in selected_cells]
    total_counts = sum(self.counts)
    #print(total_counts)
    ucb_values = [0.0] * self.num_arms
    for arm in remaining_cells:
        if self.counts[arm] >0:
            average_reward = self.q_values[arm]
            ucb_values[arm] = average_reward + self.c * math.sqrt(math.log(total_counts) / self.counts[arm])
    while len(selected_cells) < self.sample_gate:
        if all(math.isinf(val) or math.isnan(val) for val in ucb_values):
            selected_cell = random.choice(remaining_cells)
        else:
            selected_cell = ucb_values.index(max(ucb_values))
        if selected_cell not in selected_cells:
            selected_cells.add(selected_cell)
            ucb_values[selected_cell] = float('-inf')
        
    return list(selected_cells)

  def update(self, selected_arm, reward):
    for arm in selected_arm:
        self.counts[arm] += 1
        self.q_values[arm] = (self.q_values[arm] * self.counts[arm] + reward) / self.counts[arm]


# Initialization
num_cells_select = sample_gate
with open(genlib_origin, 'r') as f:
        #f_lines = [line.strip() for line in f if line.startswith("GATE") and not any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
        f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
f.close()
num_arms=len(f_lines)
mab = UCB_MAB(num_arms, c=2, sample_gate=num_cells_select)  
best_cells = None
best_result = (float('inf'), float('inf'))  
best_reward = -float('inf') 

# Main Loop
num_iterations = 300  

for i in range(num_iterations):
  print("Iteration: ", i)
  selected_cells = mab.select_action()
  try:
      delay, area = technology_mapper(genlib_origin, selected_cells)
      if delay == float("NaN") or area == float("NaN"): 
          reward = -float('inf') 
      else:
          reward = calculate_reward(max_delay, max_area, delay, area)
  except Exception: 
      reward = -float('inf')
  if reward > best_reward:
      best_reward = reward
      print("Current best reward: ", best_reward)
      best_result = (delay, area)
      print("Current best result: ", best_result)
      best_cells = selected_cells
  mab.update(selected_cells, reward)

end=time.time()
runtime=end-start

print("Best Cells:", best_cells)
print("Best Delay:", best_result[0])
print("Best Area:", best_result[1])
print("Best Reward:", best_reward)
print("Total time:", runtime)
