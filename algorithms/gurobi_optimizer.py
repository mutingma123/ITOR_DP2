"""
Gurobi-based optimization algorithms for traffic intersection scheduling.

This module implements various optimization objectives using Gurobi solver:
- Makespan minimization
- Maximum delay minimization  
- Sum delay minimization
"""

import time
import gurobipy as gp
from gurobipy import GRB
from typing import List
from models.vehicle import Vehicle
from utils.csv_handler import ResultRecord


class GurobiOptimizer:
    """
    Gurobi-based optimization for traffic intersection scheduling.
    
    This class implements various optimization formulations using Gurobi
    mixed-integer programming solver. It provides different objective
    functions and constraint formulations for traffic optimization.
    """
    
    def __init__(self, time_limit: int = 60):
        """
        Initialize Gurobi optimizer.
        
        Args:
            time_limit: Time limit for optimization in seconds
        """
        self.time_limit = time_limit
    
    def minimize_makespan(self, vehicle: Vehicle) -> ResultRecord:
        """
        Minimize makespan (maximum completion time).
        
        This formulation minimizes the time when the last vehicle
        completes crossing the intersection.
        
        Args:
            vehicle: Vehicle instance to optimize
            
        Returns:
            ResultRecord with optimization results
        """
        env = gp.Env()
        model = gp.Model("makespan", env=env)
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', self.time_limit)
        
        num_vehicles = len(vehicle.ids)
        num_from_a = vehicle.num_from_a
        num_from_b = vehicle.num_from_b
        
        # Decision variables
        entry_time = model.addVars(num_vehicles, name="entry_time", lb=0)
        conflict = model.addVars(num_from_a, num_from_b, vtype=GRB.BINARY, name="conflict")
        makespan = model.addVar(name="makespan", obj=1.0)
        
        # Objective: minimize makespan
        model.setObjective(makespan, GRB.MINIMIZE)
        
        # Conflict constraints
        for i in range(num_from_a):
            for j in range(num_from_b):
                id_a = vehicle.ids_a[i]
                id_b = vehicle.ids_b[j]
                
                model.addConstr(
                    entry_time[id_a] + vehicle.cross_time + vehicle.buffer <= 
                    entry_time[id_b] + vehicle.big_m * (1 - conflict[i, j]),
                    name=f"conflict1_{i}_{j}"
                )
                model.addConstr(
                    entry_time[id_b] + vehicle.cross_time + vehicle.buffer <= 
                    entry_time[id_a] + vehicle.big_m * conflict[i, j],
                    name=f"conflict2_{i}_{j}"
                )
        
        # Car following constraints
        for i in range(1, num_from_a):
            id_curr = vehicle.ids_a[i]
            id_prev = vehicle.ids_a[i-1]
            model.addConstr(
                entry_time[id_curr] >= entry_time[id_prev] + vehicle.sat_head,
                name=f"following_a_{i}"
            )
        
        for i in range(1, num_from_b):
            id_curr = vehicle.ids_b[i]
            id_prev = vehicle.ids_b[i-1]
            model.addConstr(
                entry_time[id_curr] >= entry_time[id_prev] + vehicle.sat_head,
                name=f"following_b_{i}"
            )
        
        # Minimum arrival time constraints
        for vehicle_id in vehicle.ids:
            arrival_time = vehicle.id_arrival_time[vehicle_id]
            model.addConstr(
                entry_time[vehicle_id] >= arrival_time + vehicle.travel_time,
                name=f"min_arrival_{vehicle_id}"
            )
        
        # Makespan constraints
        for vehicle_id in vehicle.ids:
            model.addConstr(
                entry_time[vehicle_id] <= makespan,
                name=f"makespan_{vehicle_id}"
            )
        
        # Solve
        start_time = time.time()
        model.optimize()
        computation_time = (time.time() - start_time) * 1000
        
        # Extract results
        return self._extract_results(
            model, vehicle, entry_time, computation_time, "makespan"
        )
    
    def _extract_results(self, model, vehicle: Vehicle, entry_time, 
                        computation_time: float, algorithm_mark: str) -> ResultRecord:
        """Extract optimization results into ResultRecord."""
        # Check if we have a feasible solution (either optimal or incumbent from time limit)
        if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
            # Calculate delays and makespan
            max_delay = 0.0
            min_delay = float('inf')
            total_delay = 0.0
            makespan = 0.0
            
            for vehicle_id in vehicle.ids:
                entry = entry_time[vehicle_id].X
                arrival = vehicle.id_arrival_time[vehicle_id]
                delay = entry - vehicle.travel_time - arrival
                
                max_delay = max(max_delay, delay)
                min_delay = min(min_delay, delay)
                total_delay += delay
                makespan = max(makespan, entry)
            
            avg_delay = total_delay / len(vehicle.ids)
            gap = (model.MIPGap * 100) if hasattr(model, 'MIPGap') else 0.0
            node_count = model.NodeCount if hasattr(model, 'NodeCount') else 0.0
        else:
            # Infeasible or no solution found
            max_delay = min_delay = avg_delay = makespan = 0.0
            gap = 1.0
            node_count = 0.0
        
        model.dispose()
        
        return ResultRecord(
            nbofveh=len(vehicle.ids),
            gap=gap,
            nbodnodes=node_count,
            comptime=round(computation_time),
            nbodpla=vehicle.num_platoons_a,
            nbodplb=vehicle.num_platoons_b,
            nbofa=vehicle.num_from_a,
            nbofb=vehicle.num_from_b,
            avedelay=avg_delay,
            maxdelay=max_delay,
            mindelay=min_delay,
            makespan=makespan,
            mark=algorithm_mark,
            crosstime=vehicle.cross_time,
            buffer=vehicle.buffer,
            sath=vehicle.sat_head,
            arrival=vehicle.arrival_rate
        )
