"""
Dynamic Programming algorithm for traffic intersection optimization.

This module implements the dynamic programming approach for finding optimal
vehicle scheduling at traffic intersections, reducing computational complexity
compared to brute-force optimization methods.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from models.vehicle import Vehicle
from utils.csv_handler import ResultRecord


class DPAlgorithm:
    """
    Dynamic Programming algorithm implementation.
    
    This class implements a state-space search algorithm that efficiently
    explores the solution space for vehicle scheduling at intersections.
    The algorithm maintains matrices for tracking optimal solutions and
    uses pruning strategies to reduce computational complexity.
    
    Attributes:
        mat_row (int): Number of rows in DP matrix (vehicles from lane A + 1)
        mat_col (int): Number of columns in DP matrix (vehicles from lane B + 1)
        mat_ta (np.ndarray): Time matrix for lane A solutions
        mat_tb (np.ndarray): Time matrix for lane B solutions
        mat_dpa (np.ndarray): Decision matrix for lane A
        mat_dpb (np.ndarray): Decision matrix for lane B
        mat_opa (np.ndarray): Optimal path matrix for lane A
        mat_opb (np.ndarray): Optimal path matrix for lane B
        mat_ata (np.ndarray): Arrival time matrix for lane A
        mat_atb (np.ndarray): Arrival time matrix for lane B
    """
    
    def __init__(self):
        """Initialize DP algorithm."""
        self.mat_row = 0
        self.mat_col = 0
        
        # DP matrices
        self.mat_ta: np.ndarray = np.array([])
        self.mat_tb: np.ndarray = np.array([])
        self.mat_dpa: np.ndarray = np.array([])
        self.mat_dpb: np.ndarray = np.array([])
        self.mat_opa: np.ndarray = np.array([])
        self.mat_opb: np.ndarray = np.array([])
        self.mat_ata: np.ndarray = np.array([])
        self.mat_atb: np.ndarray = np.array([])
    
    def initialize(self, vehicle: Vehicle):
        """
        Initialize DP matrices for the given vehicle configuration.
        
        Args:
            vehicle: Vehicle instance with lane assignments
        """
        self.mat_row = vehicle.num_from_a + 1
        self.mat_col = vehicle.num_from_b + 1
        
        # Initialize matrices with appropriate values
        large_value = 9999999.0
        
        self.mat_ata = np.zeros((self.mat_row, self.mat_col), dtype=int)
        self.mat_atb = np.zeros((self.mat_row, self.mat_col), dtype=int)
        self.mat_dpa = np.zeros((self.mat_row, self.mat_col), dtype=int)
        self.mat_dpb = np.zeros((self.mat_row, self.mat_col), dtype=int)
        self.mat_opa = np.zeros((self.mat_row, self.mat_col), dtype=int)
        self.mat_opb = np.zeros((self.mat_row, self.mat_col), dtype=int)
        
        self.mat_ta = np.full((self.mat_row, self.mat_col), large_value)
        self.mat_tb = np.full((self.mat_row, self.mat_col), large_value)
        
        # Set initial conditions
        self.mat_ta[0, 0] = -99999
        self.mat_tb[0, 0] = -99999
    
    def solve_with_pruning(self, vehicle: Vehicle) -> ResultRecord:
        """
        Solve using DP with conflict-aware pruning.
        
        This is the main DP algorithm that explores the state space
        while applying pruning based on conflict constraints between
        vehicles from different lanes.
        
        Args:
            vehicle: Vehicle instance to optimize
            
        Returns:
            ResultRecord with optimization results
        """
        self.initialize(vehicle)
        
        # Track computation statistics
        start_time = time.time()
        nodes_visited = 1
        
        # State tracking dictionaries
        child_nodes: Dict[int, List[int]] = {}
        child_paths: Dict[int, List[int]] = {}
        child_times: Dict[int, List[float]] = {}
        
        parent_nodes: Dict[int, List[int]] = {0: [0, 0, 99]}
        parent_paths: Dict[int, List[int]] = {0: [0]}
        parent_times: Dict[int, List[float]] = {0: [0]}
        
        try:
            # Main DP loop
            while len(parent_paths[0]) < self.mat_row + self.mat_col - 1:
                child_nodes.clear()
                child_paths.clear()
                child_times.clear()
                
                # Process each parent state
                for i in range(len(parent_nodes)):
                    if i not in parent_nodes:
                        continue
                        
                    path_id = parent_paths[i].copy()
                    time_id = parent_times[i].copy()
                    ord_a, ord_b, ord_c = parent_nodes[i]
                    
                    # Process transitions based on current state
                    self._process_state_transitions(
                        vehicle, ord_a, ord_b, ord_c, path_id, time_id,
                        child_nodes, child_paths, child_times
                    )
                
                nodes_visited += len(child_nodes)
                
                # Update for next iteration
                parent_nodes = child_nodes.copy()
                parent_paths = child_paths.copy()
                parent_times = child_times.copy()
            
            computation_time = (time.time() - start_time) * 1000  # Convert to ms
            
        except Exception as e:
            print(f"DP algorithm error: {e}")
            computation_time = (time.time() - start_time) * 1000
        
        # Extract final solution
        final_path, optimal_times = self._extract_final_solution(
            parent_paths, parent_times
        )
        
        # Calculate performance metrics
        return self._calculate_metrics(
            vehicle, final_path, optimal_times, nodes_visited, 
            computation_time, with_pruning=True
        )
    
    def solve_without_pruning(self, vehicle: Vehicle) -> ResultRecord:
        """
        Solve using DP without conflict-aware pruning.
        
        This version explores all possible transitions without applying
        the pruning strategy, useful for comparison and validation.
        
        Args:
            vehicle: Vehicle instance to optimize
            
        Returns:
            ResultRecord with optimization results
        """
        # Similar to solve_with_pruning but without conflict constraints
        # Implementation follows the same pattern but processes all transitions
        self.initialize(vehicle)
        
        start_time = time.time()
        nodes_visited = 1
        
        # State tracking (simplified version)
        child_nodes: Dict[int, List[int]] = {}
        child_paths: Dict[int, List[int]] = {}
        child_times: Dict[int, List[float]] = {}
        
        parent_nodes: Dict[int, List[int]] = {0: [0, 0, 99]}
        parent_paths: Dict[int, List[int]] = {0: [0]}
        parent_times: Dict[int, List[float]] = {0: [0]}
        
        try:
            while len(parent_paths[0]) < self.mat_row + self.mat_col - 1:
                child_nodes.clear()
                child_paths.clear()
                child_times.clear()
                
                for i in range(len(parent_nodes)):
                    if i not in parent_nodes:
                        continue
                        
                    path_id = parent_paths[i].copy()
                    time_id = parent_times[i].copy()
                    ord_a, ord_b, ord_c = parent_nodes[i]
                    
                    # Process all transitions without pruning
                    self._process_all_transitions(
                        vehicle, ord_a, ord_b, ord_c, path_id, time_id,
                        child_nodes, child_paths, child_times
                    )
                
                nodes_visited += len(child_nodes)
                
                parent_nodes = child_nodes.copy()
                parent_paths = child_paths.copy()
                parent_times = child_times.copy()
            
            computation_time = (time.time() - start_time) * 1000
            
        except Exception as e:
            print(f"DP algorithm error: {e}")
            computation_time = (time.time() - start_time) * 1000
        
        final_path, optimal_times = self._extract_final_solution(
            parent_paths, parent_times
        )
        
        return self._calculate_metrics(
            vehicle, final_path, optimal_times, nodes_visited, 
            computation_time, with_pruning=False
        )
    
    def _process_state_transitions(self, vehicle: Vehicle, ord_a: int, ord_b: int, 
                                 ord_c: int, path_id: List[int], time_id: List[float],
                                 child_nodes: Dict, child_paths: Dict, 
                                 child_times: Dict):
        """Process state transitions with conflict-aware pruning."""
        # Handle boundary conditions and conflict-aware transitions
        
        # Last vehicle from lane A
        if ord_a == self.mat_row - 1:
            self._process_lane_a_boundary(
                vehicle, ord_a, ord_b, ord_c, path_id, time_id,
                child_nodes, child_paths, child_times
            )
        
        # Last vehicle from lane B
        if ord_b == self.mat_col - 1:
            self._process_lane_b_boundary(
                vehicle, ord_a, ord_b, ord_c, path_id, time_id,
                child_nodes, child_paths, child_times
            )
        
        # Interior states with conflict checking
        if ord_a < self.mat_row - 1 and ord_b < self.mat_col - 1:
            self._process_interior_state_with_pruning(
                vehicle, ord_a, ord_b, ord_c, path_id, time_id,
                child_nodes, child_paths, child_times
            )
    
    def _process_all_transitions(self, vehicle: Vehicle, ord_a: int, ord_b: int,
                               ord_c: int, path_id: List[int], time_id: List[float],
                               child_nodes: Dict, child_paths: Dict, 
                               child_times: Dict):
        """Process all state transitions without pruning."""
        # Similar to _process_state_transitions but without conflict constraints
        
        if ord_a == self.mat_row - 1:
            self._process_lane_a_boundary(
                vehicle, ord_a, ord_b, ord_c, path_id, time_id,
                child_nodes, child_paths, child_times
            )
        
        if ord_b == self.mat_col - 1:
            self._process_lane_b_boundary(
                vehicle, ord_a, ord_b, ord_c, path_id, time_id,
                child_nodes, child_paths, child_times
            )
        
        if ord_a < self.mat_row - 1 and ord_b < self.mat_col - 1:
            self._process_interior_state_without_pruning(
                vehicle, ord_a, ord_b, ord_c, path_id, time_id,
                child_nodes, child_paths, child_times
            )
    
    def _process_lane_a_boundary(self, vehicle: Vehicle, ord_a: int, ord_b: int,
                               ord_c: int, path_id: List[int], time_id: List[float],
                               child_nodes: Dict, child_paths: Dict, 
                               child_times: Dict):
        """Process boundary condition for lane A."""
        if ord_c == 1:  # Coming from lane A
            temp = max(
                vehicle.id_arrival_time[vehicle.ids_b[ord_b]] + vehicle.travel_time,
                self.mat_ta[ord_a][ord_b] + vehicle.cross_time + vehicle.buffer
            )
        else:  # Coming from lane B
            temp = max(
                vehicle.id_arrival_time[vehicle.ids_b[ord_b]] + vehicle.travel_time,
                self.mat_tb[ord_a][ord_b] + vehicle.sat_head
            )
        
        if temp < self.mat_tb[ord_a][ord_b + 1]:
            self._update_state_b(
                vehicle, ord_a, ord_b, temp, path_id, time_id,
                child_nodes, child_paths, child_times
            )
    
    def _process_lane_b_boundary(self, vehicle: Vehicle, ord_a: int, ord_b: int,
                               ord_c: int, path_id: List[int], time_id: List[float],
                               child_nodes: Dict, child_paths: Dict, 
                               child_times: Dict):
        """Process boundary condition for lane B."""
        if ord_c == 1:  # Coming from lane A
            temp = max(
                vehicle.id_arrival_time[vehicle.ids_a[ord_a]] + vehicle.travel_time,
                self.mat_ta[ord_a][ord_b] + vehicle.sat_head
            )
        else:  # Coming from lane B
            temp = max(
                vehicle.id_arrival_time[vehicle.ids_a[ord_a]] + vehicle.travel_time,
                self.mat_tb[ord_a][ord_b] + vehicle.cross_time + vehicle.buffer
            )
        
        if temp < self.mat_ta[ord_a + 1][ord_b]:
            self._update_state_a(
                vehicle, ord_a, ord_b, temp, path_id, time_id,
                child_nodes, child_paths, child_times
            )
    
    def _process_interior_state_with_pruning(self, vehicle: Vehicle, ord_a: int, 
                                           ord_b: int, ord_c: int, 
                                           path_id: List[int], time_id: List[float],
                                           child_nodes: Dict, child_paths: Dict, 
                                           child_times: Dict):
        """Process interior state with conflict-aware pruning."""
        # Calculate potential transition times
        if ord_c == 1:  # Coming from lane A
            temp_a = max(
                vehicle.id_arrival_time[vehicle.ids_a[ord_a]] + vehicle.travel_time,
                self.mat_ta[ord_a][ord_b] + vehicle.sat_head
            )
            temp_b = max(
                vehicle.id_arrival_time[vehicle.ids_b[ord_b]] + vehicle.travel_time,
                self.mat_ta[ord_a][ord_b] + vehicle.cross_time + vehicle.buffer
            )
        else:  # Coming from lane B
            temp_a = max(
                vehicle.id_arrival_time[vehicle.ids_a[ord_a]] + vehicle.travel_time,
                self.mat_tb[ord_a][ord_b] + vehicle.cross_time + vehicle.buffer
            )
            temp_b = max(
                vehicle.id_arrival_time[vehicle.ids_b[ord_b]] + vehicle.travel_time,
                self.mat_tb[ord_a][ord_b] + vehicle.sat_head
            )
        
        # Apply conflict-aware pruning
        time_diff = temp_a - temp_b
        conflict_threshold = vehicle.buffer + vehicle.cross_time - vehicle.sat_head
        
        if round(time_diff - conflict_threshold, 3) >= 0:
            # Prefer lane B
            if temp_b < self.mat_tb[ord_a][ord_b + 1]:
                self._update_state_b(
                    vehicle, ord_a, ord_b, temp_b, path_id, time_id,
                    child_nodes, child_paths, child_times
                )
        elif round(time_diff + conflict_threshold, 3) <= 0:
            # Prefer lane A
            if temp_a < self.mat_ta[ord_a + 1][ord_b]:
                self._update_state_a(
                    vehicle, ord_a, ord_b, temp_a, path_id, time_id,
                    child_nodes, child_paths, child_times
                )
        else:
            # Explore both options
            if temp_a < self.mat_ta[ord_a + 1][ord_b]:
                self._update_state_a(
                    vehicle, ord_a, ord_b, temp_a, path_id, time_id,
                    child_nodes, child_paths, child_times
                )
            if temp_b < self.mat_tb[ord_a][ord_b + 1]:
                self._update_state_b(
                    vehicle, ord_a, ord_b, temp_b, path_id, time_id,
                    child_nodes, child_paths, child_times
                )
    
    def _process_interior_state_without_pruning(self, vehicle: Vehicle, ord_a: int,
                                              ord_b: int, ord_c: int,
                                              path_id: List[int], time_id: List[float],
                                              child_nodes: Dict, child_paths: Dict,
                                              child_times: Dict):
        """Process interior state without pruning - explore all transitions."""
        # Calculate transition times
        if ord_c == 1:
            temp_a = max(
                vehicle.id_arrival_time[vehicle.ids_a[ord_a]] + vehicle.travel_time,
                self.mat_ta[ord_a][ord_b] + vehicle.sat_head
            )
            temp_b = max(
                vehicle.id_arrival_time[vehicle.ids_b[ord_b]] + vehicle.travel_time,
                self.mat_ta[ord_a][ord_b] + vehicle.cross_time + vehicle.buffer
            )
        else:
            temp_a = max(
                vehicle.id_arrival_time[vehicle.ids_a[ord_a]] + vehicle.travel_time,
                self.mat_tb[ord_a][ord_b] + vehicle.cross_time + vehicle.buffer
            )
            temp_b = max(
                vehicle.id_arrival_time[vehicle.ids_b[ord_b]] + vehicle.travel_time,
                self.mat_tb[ord_a][ord_b] + vehicle.sat_head
            )
        
        # Explore both transitions without pruning
        if temp_a < self.mat_ta[ord_a + 1][ord_b]:
            self._update_state_a(
                vehicle, ord_a, ord_b, temp_a, path_id, time_id,
                child_nodes, child_paths, child_times
            )
        
        if temp_b < self.mat_tb[ord_a][ord_b + 1]:
            self._update_state_b(
                vehicle, ord_a, ord_b, temp_b, path_id, time_id,
                child_nodes, child_paths, child_times
            )
    
    def _update_state_a(self, vehicle: Vehicle, ord_a: int, ord_b: int, 
                       temp: float, path_id: List[int], time_id: List[float],
                       child_nodes: Dict, child_paths: Dict, child_times: Dict):
        """Update state for lane A transition."""
        self.mat_ta[ord_a + 1][ord_b] = temp
        
        if self.mat_dpa[ord_a + 1][ord_b] == 0:
            # First time reaching this state
            vec = [ord_a + 1, ord_b, 1]
            child_nodes[len(child_nodes)] = vec
            
            new_path = path_id.copy()
            new_path.append(vehicle.ids_a[ord_a])
            child_paths[len(child_paths)] = new_path
            
            self.mat_opa[ord_a + 1][ord_b] = len(child_paths) - 1
            self.mat_dpa[ord_a + 1][ord_b] = 1
            
            new_time = time_id.copy()
            new_time.append(temp)
            child_times[len(child_times)] = new_time
            
            self.mat_ata[ord_a + 1][ord_b] = len(child_times) - 1
        else:
            # Update existing state
            path_idx = self.mat_opa[ord_a + 1][ord_b]
            time_idx = self.mat_ata[ord_a + 1][ord_b]
            
            new_path = path_id.copy()
            new_path.append(vehicle.ids_a[ord_a])
            child_paths[path_idx] = new_path
            
            new_time = time_id.copy()
            new_time.append(temp)
            child_times[time_idx] = new_time
    
    def _update_state_b(self, vehicle: Vehicle, ord_a: int, ord_b: int,
                       temp: float, path_id: List[int], time_id: List[float],
                       child_nodes: Dict, child_paths: Dict, child_times: Dict):
        """Update state for lane B transition."""
        self.mat_tb[ord_a][ord_b + 1] = temp
        
        if self.mat_dpb[ord_a][ord_b + 1] == 0:
            # First time reaching this state
            vec = [ord_a, ord_b + 1, 0]
            child_nodes[len(child_nodes)] = vec
            
            new_path = path_id.copy()
            new_path.append(vehicle.ids_b[ord_b])
            child_paths[len(child_paths)] = new_path
            
            self.mat_opb[ord_a][ord_b + 1] = len(child_paths) - 1
            self.mat_dpb[ord_a][ord_b + 1] = 1
            
            new_time = time_id.copy()
            new_time.append(temp)
            child_times[len(child_times)] = new_time
            
            self.mat_atb[ord_a][ord_b + 1] = len(child_times) - 1
        else:
            # Update existing state
            path_idx = self.mat_opb[ord_a][ord_b + 1]
            time_idx = self.mat_atb[ord_a][ord_b + 1]
            
            new_path = path_id.copy()
            new_path.append(vehicle.ids_b[ord_b])
            child_paths[path_idx] = new_path
            
            new_time = time_id.copy()
            new_time.append(temp)
            child_times[time_idx] = new_time
    
    def _extract_final_solution(self, parent_paths: Dict, 
                              parent_times: Dict) -> Tuple[List[int], List[float]]:
        """Extract the best final solution from parent states."""
        if len(parent_paths) == 1:
            return parent_paths[0].copy(), parent_times[0].copy()
        else:
            # Select solution with minimum makespan
            best_idx = 0
            best_makespan = parent_times[0][-1]
            
            for i in range(1, len(parent_times)):
                if parent_times[i][-1] < best_makespan:
                    best_makespan = parent_times[i][-1]
                    best_idx = i
            
            return parent_paths[best_idx].copy(), parent_times[best_idx].copy()
    
    def _calculate_metrics(self, vehicle: Vehicle, final_path: List[int],
                          optimal_times: List[float], nodes_visited: int,
                          computation_time: float, with_pruning: bool) -> ResultRecord:
        """Calculate performance metrics for the solution."""
        max_delay = 0.0
        min_delay = float('inf')
        total_delay = 0.0
        
        # Calculate delays for each vehicle
        for i in range(len(vehicle.ids)):
            vehicle_id = final_path[i + 1]
            entry_time = optimal_times[i + 1]
            arrival_time = vehicle.id_arrival_time[vehicle_id]
            
            delay = entry_time - vehicle.travel_time - arrival_time
            max_delay = max(max_delay, delay)
            min_delay = min(min_delay, delay)
            total_delay += delay
        
        avg_delay = total_delay / len(vehicle.ids)
        makespan = optimal_times[-1]
        
        # Calculate theoretical node count for validation
        theoretical_nodes = (
            2 * vehicle.num_from_a * vehicle.num_from_b + 
            vehicle.num_from_a + vehicle.num_from_b + 1 -
            vehicle.num_platoons_a * vehicle.num_from_b -
            vehicle.num_platoons_b * vehicle.num_from_a -
            vehicle.early_sep_platoon_a * 2 * vehicle.num_from_a -
            vehicle.early_sep_platoon_b * 2 * vehicle.num_from_b
        )
        
        validation_flag = 'T' if nodes_visited == theoretical_nodes else 'F'
        
        return ResultRecord(
            nbofveh=len(vehicle.ids),
            gap=0.0,
            nbodnodes=nodes_visited,
            comptime=round(computation_time),
            nbodpla=vehicle.num_platoons_a,
            nbodplb=vehicle.num_platoons_b,
            nbofa=vehicle.num_from_a,
            nbofb=vehicle.num_from_b,
            avedelay=avg_delay,
            maxdelay=max_delay,
            mindelay=min_delay,
            makespan=makespan,
            mark="dp" if with_pruning else "nondp",
            espa=vehicle.early_sep_platoon_a,
            espb=vehicle.early_sep_platoon_b,
            nodes=theoretical_nodes,
            truefalse=validation_flag,
            crosstime=vehicle.cross_time,
            buffer=vehicle.buffer,
            sath=vehicle.sat_head,
            arrival=vehicle.arrival_rate
        )