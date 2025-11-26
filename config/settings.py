"""
Configuration settings for the traffic intersection optimization system.
"""

# Traffic intersection parameters
DEFAULT_CROSS_TIME = 2.0  # Time to cross intersection
DEFAULT_BUFFER = 2.0      # Safety buffer time
DEFAULT_SAT_HEAD = 2.0    # Saturation headway
DEFAULT_TRAVEL_TIME = 10.0  # Travel time to intersection

# Lane identifiers
LANE_A_ID = 1
LANE_B_ID = 2

# Optimization parameters
BIG_M_MULTIPLIER = 1.0  # Multiplier for big-M constraints
TIME_LIMIT = 60  # Gurobi time limit in seconds

# Simulation parameters
MIN_VEHICLES = 5
MAX_VEHICLES = 305
VEHICLE_STEP = 50

# R distribution parameters
TRUNCATE_LB = 1.5  # Lower bound for truncated exponential distribution

# Output settings
RESULTS_DIR = "results"
RESULTS_FILE = "results.csv"