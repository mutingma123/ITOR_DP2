import os
import sys
from unittest import runner
import numpy as np
from typing import List

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import *
from models.vehicle import Vehicle
from models.environment import Environment
from algorithms.dynamic_programming import DPAlgorithm
from algorithms.gurobi_optimizer import GurobiOptimizer
from algorithms.heuristics import HeuristicAlgorithms
from utils.csv_handler import CSVHandler
from utils.r_integration import RIntegration

class ExperimentRunner:
    """
    Main experiment runner for traffic intersection optimization.
    
    This class coordinates the execution of various optimization algorithms
    and manages the experimental parameters and result collection.
    """
    
    def __init__(self):
        """Initialize experiment runner."""
        self.csv_handler = CSVHandler(RESULTS_DIR)
        self.r_integration = RIntegration()
        self.environment = Environment(self.r_integration)
        
        # Initialize algorithm instances
        self.dp_algorithm = DPAlgorithm()
        self.gurobi_optimizer = GurobiOptimizer(TIME_LIMIT)
        self.heuristics = HeuristicAlgorithms()
    
    def run_comparison_experiments(self):
        """
        Run comprehensive comparison experiments.
        
        This method runs the main comparison between DP and other methods
        across different parameter combinations.
        """
        print("Starting traffic intersection optimization experiments...")
        
        self.csv_handler.open_files(RESULTS_FILE)
        
        try:
            # Experiment parameters (subset of original for demonstration)
            vehicle_counts = range(MIN_VEHICLES, MAX_VEHICLES + 1, VEHICLE_STEP)
            cross_times = [2.0 + i * 0.5 for i in range(5)]  # 2.0 to 4.0
            buffers = [1.0 + i * 1.0 for i in range(2)]       # 1.0 to 2.0  
            sat_heads = [2.0 + i * 1.0 for i in range(3)]     # 2.0 to 4.0
            arrival_rates = [2.0 + i * 1.0 for i in range(5)] # 2.0 to 6.0
            
            total_runs = 0
            completed_runs = 0
            
            # Count total runs for progress tracking
            for num_vehicles in vehicle_counts:
                for cross_time in cross_times:
                    for buffer in buffers:
                        for sat_head in sat_heads:
                            for arrival_rate in arrival_rates:
                                if sat_head < cross_time + buffer:
                                    total_runs += 5  # 5 runs per parameter combination
            
            print(f"Total experimental runs planned: {total_runs}")
            
            # Main experimental loop
            for num_vehicles in vehicle_counts:
                vehicle_ids = np.arange(num_vehicles)
                
                for cross_time in cross_times:
                    for buffer in buffers:
                        for sat_head in sat_heads:
                            for arrival_rate in arrival_rates:
                                
                                # Skip invalid parameter combinations
                                if sat_head >= cross_time + buffer:
                                    continue
                                
                                # Run multiple instances for statistical validity
                                for run_idx in range(5):
                                    completed_runs += 1
                                    print(f"Progress: {completed_runs}/{total_runs} - "
                                          f"Vehicles: {num_vehicles}, Run: {run_idx + 1}")
                                    
                                    self._run_single_experiment(
                                        vehicle_ids, cross_time, buffer, sat_head,
                                        arrival_rate
                                    )
            
            print("Comparison experiments completed successfully!")
            
        except Exception as e:
            print(f"Error during experiments: {e}")
            raise
        finally:
            self.csv_handler.close_files()
    
    def _run_single_experiment(self, vehicle_ids: np.ndarray, cross_time: float,
                              buffer: float, sat_head: float, arrival_rate: float):
        """
        Run a single experimental instance with all algorithms.
        
        Args:
            vehicle_ids: Array of vehicle identifiers
            cross_time: Intersection crossing time
            buffer: Safety buffer time
            sat_head: Saturation headway
            arrival_rate: Vehicle arrival rate
        """
        # Create vehicle instance
        vehicle = Vehicle(
            vehicle_ids, cross_time, buffer, sat_head,
            LANE_A_ID, LANE_B_ID, DEFAULT_TRAVEL_TIME, arrival_rate
        )
        
        # Setup environment (assign lanes and generate arrival times)
        mean = arrival_rate - TRUNCATE_LB
        self.environment.setup_intersection_scenario(vehicle, mean, TRUNCATE_LB)
        
        # Run all algorithms
        try:
            # Dynamic Programming with pruning
            result_dp = self.dp_algorithm.solve_with_pruning(vehicle)
            self.csv_handler.write_result_record(result_dp)
            
            # Dynamic Programming without pruning
            result_dp_no_prune = self.dp_algorithm.solve_without_pruning(vehicle)
            self.csv_handler.write_result_record(result_dp_no_prune)
            
            # FCFS heuristic
            result_fcfs = self.heuristics.fcfs(vehicle)
            self.csv_handler.write_result_record(result_fcfs)
            
            # Batch processing heuristic
            result_batch = self.heuristics.batch_processing(vehicle, 10)
            self.csv_handler.write_result_record(result_batch)
            
            # Gurobi makespan minimization
            result_makespan = self.gurobi_optimizer.minimize_makespan(vehicle)
            self.csv_handler.write_result_record(result_makespan)
            
        except Exception as e:
            print(f"Error in single experiment: {e}")
            # Continue with next experiment
    
