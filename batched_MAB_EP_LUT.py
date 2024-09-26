import random
import sys
import os
import numpy as np
import subprocess
from subprocess import PIPE
import re
import time
import argparse
import logging
from datetime import datetime

# Set up logging configuration dynamically based on the design and store in two different files
def setup_logging(design, sample_gate, klut, log_dir="logs"):
    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create the log filenames dynamically
    timestamp = "" 
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{design[:-5]}_sample{sample_gate}_klut{klut}_{timestamp}.log"
    best_cells_filename = f"{design[:-5]}_sample{sample_gate}_klut{klut}_best_cells_{timestamp}.log"
    
    log_filepath = os.path.join(log_dir, log_filename)
    best_cells_filepath = os.path.join(log_dir, best_cells_filename)

    # Set up standard logging for the primary log file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Log to stdout
            logging.FileHandler(log_filepath, mode='w')  # Log to the primary file
        ]
    )

    # Set up an additional logger for best cells
    best_cells_logger = logging.getLogger("best_cells_logger")
    best_cells_logger.setLevel(logging.INFO)

    # Create a handler for the best cells log file
    best_cells_handler = logging.FileHandler(best_cells_filepath, mode='w')
    best_cells_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    # Add the handler to the best cells logger
    best_cells_logger.addHandler(best_cells_handler)

    logging.info(f"Log file created at: {log_filepath}")
    logging.info(f"Best cells file created at: {best_cells_filepath}")

    return best_cells_logger  # Return the best cells logger for later use

# Argument parser setup
def parse_arguments():
    parser = argparse.ArgumentParser(description="Technology Mapping using Epsilon-Greedy MAB")

    # Required arguments
    parser.add_argument('design', type=str, help="Path to the design file (e.g., .blif or .v)")
    parser.add_argument('genlib_origin', type=str, help="Path to the original .genlib file")
    parser.add_argument('sample_gate', type=int, help="Number of gates to sample for the MAB algorithm")
    
    # Optional arguments
    parser.add_argument('--klut', type=int, default=6, help="Maximum number of LUT inputs (default: 6)")
    parser.add_argument('--num_iterations', type=int, default=5, help="Number of MAB iterations (default: 5)")
    parser.add_argument('--batch_size', type=int, default=10, help="Batch size for MAB (default: 10)")
    parser.add_argument('--log_dir', type=str, default="logs", help="Directory where logs will be saved (default: logs)")

    return parser.parse_args()

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

# Mapper call
def technology_mapper(genlib_origin, partial_cell_library, write_output="", design="", klut=6, lib_path="gen_newlibs/"):
    with open(genlib_origin, 'r') as f:
        f_lines = [line.strip() for line in f if line.startswith("GATE") and not any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
    with open(genlib_origin, 'r') as f:
        f_keep = [line.strip() for line in f if any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
    
    lines_partial = [f_lines[i] for i in partial_cell_library]
    lines_partial = lines_partial + f_keep

    output_genlib_file = lib_path + design + "_" + str(len(lines_partial)) + "_bs_ep_samplelib.genlib"
    with open(output_genlib_file, 'w') as out_gen:
        for line in lines_partial:
            out_gen.write(line + '\n')

    if write_output == "":
        abc_cmd = f"read {output_genlib_file}; read {design}; st; dch; map -a; st; dch; st; if -K {klut}; ps"
    else:
        abc_cmd = f"read {output_genlib_file}; read {design}; st; dch; map -a; st; dch; st; if -K {klut}; ps; write_blif {write_output}"
    
    try:
        res = subprocess.check_output(('abc', '-c', abc_cmd))
    except subprocess.CalledProcessError as e:
        logging.error(f"ABC tool execution failed: {e}")
        return float("NaN"), float("NaN")

    match_d = re.search(r"lev\s*=\s*([\d.]+)\s*", str(res))
    match_a = re.search(r"nd\s*=\s*([\d.]+)", str(res))
    
    if match_d and match_a:
        delay = float(match_d.group(1))
        area = float(match_a.group(1))
    else:
        delay, area = float("NaN"), float("NaN")
    
    return delay, area

# Main function
def main():
    args = parse_arguments()

    design = args.design
    genlib_origin = args.genlib_origin
    sample_gate = args.sample_gate
    kLUT = args.klut
    num_iterations = args.num_iterations
    batch_size = args.batch_size
    log_dir = args.log_dir

    # Setup logging with dynamic filename, return the best cells logger
    best_cells_logger = setup_logging(design, sample_gate, kLUT, log_dir)

    # Log all the parameters for tracking purposes
    logging.info(f"Starting technology mapping with the following parameters:")
    logging.info(f"Design file: {design}")
    logging.info(f"Genlib file: {genlib_origin}")
    logging.info(f"Sample gate: {sample_gate}")
    logging.info(f"kLUT: {kLUT}")
    logging.info(f"Number of iterations: {num_iterations}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Log directory: {log_dir}")

    best_lut_blif = f"best_blifs/{design[:-5]}_bs_ep_temp_LUT{kLUT}.blif"
    
    # Start timing
    start = time.time()

    # Baseline ABC call
    abc_cmd = f"read {genlib_origin}; read {design}; st; resyn; dch; st; resyn; dch; if -K {kLUT}; ps"
    res = subprocess.check_output(('abc', '-c', abc_cmd))

    match_d = re.search(r"lev\s*=\s*([\d.]+)\s*", str(res))
    match_a = re.search(r"nd\s*=\s*([\d.]+)", str(res))

    max_delay = float(match_d.group(1))
    max_area = float(match_a.group(1))

    logging.info(f"Baseline Delay: {max_delay}")
    logging.info(f"Baseline Area: {max_area}")

    # Initialization for MAB
    with open(genlib_origin, 'r') as f:
        f_lines = [line.strip() for line in f if line.startswith("GATE") and "BUF" not in line and "INV" not in line]
    
    num_arms = len(f_lines)
    mab = EpsilonGreedyMAB(num_arms, 0.2, sample_gate, batch_size)

    best_cells = None
    best_result = (float('inf'), float('inf'))
    best_reward = -float('inf')

    for i in range(num_iterations):
        logging.info(f"Batch iteration: {i}")
        batch_actions = mab.select_batch_actions()
        batch_rewards = []

        for selected_cells in batch_actions:
            delay, area = technology_mapper(genlib_origin, selected_cells, design=design, klut=kLUT)
            if np.isnan(delay) or np.isnan(area):
                reward = -float('inf')
            else:
                reward = calculate_reward(max_delay, max_area, delay, area)

            if reward > best_reward:
                best_reward = reward
                best_result = (delay, area)
                best_cells = selected_cells

            batch_rewards.append(reward)

        logging.info(f"Current best reward: {best_reward}")
        logging.info(f"Current best result: {best_result}")
        logging.info(f"[Iteration {i}] Best delay={best_result[0]}, area={best_result[1]}")
        best_cells_logger.info(f"Best cells (list of integers) for iteration {i}: {best_cells}")

        mab.update_batch(batch_actions, batch_rewards)

    # Final best mapping
    delay, area = technology_mapper(genlib_origin, best_cells, write_output=best_lut_blif, design=design, klut=kLUT)
    logging.info(f"[Final] Best delay={delay}, area={area}, Mapped blif saved at {best_lut_blif}")
    best_cells_logger.info(f"Final Best cells (list of integers): {best_cells}")

    # End timing
    end = time.time()
    runtime = end - start

    logging.info(f"Best Delay: {best_result[0]}")
    logging.info(f"Best Area: {best_result[1]}")
    logging.info(f"Best Reward: {best_reward}")
    logging.info(f"Total runtime: {runtime} seconds")

if __name__ == "__main__":
    main()
