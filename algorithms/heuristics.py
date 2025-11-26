"""
Heuristic algorithms for traffic intersection optimization.

This module implements heuristic approaches including:
- FCFS (First Come First Serve)
- Batch processing heuristic

These provide fast approximate solutions for comparison with optimal methods.
"""

import time
from typing import Dict, List
from models.vehicle import Vehicle
from utils.csv_handler import ResultRecord


class HeuristicAlgorithms:
    """
    Collection of heuristic algorithms for traffic intersection optimization.
    
    This class implements fast heuristic methods that provide good approximate
    solutions without the computational overhead of exact optimization methods.
    """
    
    def __init__(self):
        """Initialize heuristic algorithms."""
        pass
    
    def fcfs(self, vehicle: Vehicle) -> ResultRecord:
        """
        First Come First Serve (FCFS) heuristic.
        
        This algorithm processes vehicles in order of their arrival times,
        assigning entry times based on a simple priority rule while
        respecting safety constraints.
        
        Args:
            vehicle: Vehicle instance to process
            
        Returns:
            ResultRecord with heuristic results
        """
        start_time = time.time()
        
        # Sort vehicles by arrival time
        sorted_vehicles = sorted(
            vehicle.id_arrival_time.items(),
            key=lambda x: x[1]
        )
        
        final_times: Dict[int, float] = {}
        current_lane = None
        processing_time = -999999.0
        
        # Process vehicles in arrival order
        for vehicle_id, arrival_time in sorted_vehicles:
            vehicle_lane = vehicle.id_lane_map[vehicle_id]
            
            if current_lane == vehicle_lane:
                # Same lane: apply saturation headway
                processing_time = max(
                    processing_time + vehicle.sat_head,
                    arrival_time
                )
            else:
                # Different lane: apply crossing time + buffer
                processing_time = max(
                    processing_time + vehicle.cross_time + vehicle.buffer,
                    arrival_time
                )
            
            final_times[vehicle_id] = processing_time
            current_lane = vehicle_lane
        
        computation_time = (time.time() - start_time) * 1000
        
        # Calculate performance metrics
        return self._calculate_heuristic_metrics(
            vehicle, final_times, computation_time, "fcfs"
        )
    
    def batch_processing(self, vehicle: Vehicle, batch_size: int) -> ResultRecord:
        """
        Batch processing heuristic.
        
        This algorithm groups vehicles into batches of limited size
        and processes them with inter-batch gaps for safety.
        
        Args:
            vehicle: Vehicle instance to process
            batch_size: Maximum vehicles per batch
            
        Returns:
            ResultRecord with heuristic results
        """
        start_time = time.time()
        
        # Sort vehicles by arrival time
        sorted_vehicles = sorted(
            vehicle.id_arrival_time.items(),
            key=lambda x: x[1]
        )
        
        final_times: Dict[int, float] = {}
        current_lane = None
        processing_time = -999999.0
        vehicles_in_current_batch = 0
        lane_vehicles_remaining = {
            vehicle.lane_a: sum(1 for vid in vehicle.ids_a),
            vehicle.lane_b: sum(1 for vid in vehicle.ids_b)
        }
        
        # Create lane-specific vehicle maps
        vehicle_to_lane = {vid: vehicle.id_lane_map[vid] for vid in vehicle.ids}
        
        while sorted_vehicles:
            # Process current batch
            batch_processed = 0
            
            # Find vehicles for current lane that can form a batch
            available_vehicles = [
                (vid, arr_time) for vid, arr_time in sorted_vehicles
                if vehicle_to_lane[vid] == current_lane
            ]
            
            if not available_vehicles:
                # Switch to other lane
                remaining_lanes = set(vehicle_to_lane[vid] for vid, _ in sorted_vehicles)
                if remaining_lanes:
                    current_lane = next(iter(remaining_lanes))
                    continue
                else:
                    break
            
            # Process up to batch_size vehicles from current lane
            for i, (vehicle_id, arrival_time) in enumerate(available_vehicles):
                if batch_processed >= batch_size:
                    break
                
                if batch_processed == 0:
                    # First vehicle in batch
                    processing_time = max(
                        processing_time + vehicle.sat_head,
                        arrival_time
                    )
                else:
                    # Subsequent vehicles in batch
                    processing_time = max(
                        processing_time + vehicle.sat_head,
                        arrival_time
                    )
                
                final_times[vehicle_id] = processing_time
                sorted_vehicles.remove((vehicle_id, arrival_time))
                batch_processed += 1
                lane_vehicles_remaining[vehicle_to_lane[vehicle_id]] -= 1
            
            # Add inter-batch gap when switching lanes
            if sorted_vehicles:
                processing_time += vehicle.cross_time + vehicle.buffer - vehicle.sat_head
                
                # Switch to lane with remaining vehicles
                remaining_vehicles_by_lane = {}
                for vid, arr_time in sorted_vehicles:
                    lane = vehicle_to_lane[vid]
                    if lane not in remaining_vehicles_by_lane:
                        remaining_vehicles_by_lane[lane] = []
                    remaining_vehicles_by_lane[lane].append((vid, arr_time))
                
                if remaining_vehicles_by_lane:
                    # Choose lane with earliest next vehicle
                    next_lane_times = {
                        lane: min(arr_time for _, arr_time in vehicles)
                        for lane, vehicles in remaining_vehicles_by_lane.items()
                    }
                    current_lane = min(next_lane_times.keys(), 
                                     key=lambda x: next_lane_times[x])
        
        computation_time = (time.time() - start_time) * 1000
        
        return self._calculate_heuristic_metrics(
            vehicle, final_times, computation_time, "batch"
        )
    
    def _calculate_heuristic_metrics(self, vehicle: Vehicle, 
                                   final_times: Dict[int, float],
                                   computation_time: float, 
                                   algorithm_mark: str) -> ResultRecord:
        """
        Calculate performance metrics for heuristic solutions.
        
        Args:
            vehicle: Vehicle instance
            final_times: Dictionary of vehicle_id -> final_time
            computation_time: Computation time in milliseconds
            algorithm_mark: Algorithm identifier string
            
        Returns:
            ResultRecord with calculated metrics
        """
        if not final_times:
            return ResultRecord(
                nbofveh=len(vehicle.ids),
                mark=algorithm_mark,
                comptime=round(computation_time),
                crosstime=vehicle.cross_time,
                buffer=vehicle.buffer,
                sath=vehicle.sat_head,
                arrival=vehicle.arrival_rate
            )
        
        # Calculate delays
        max_delay = 0.0
        min_delay = float('inf')
        total_delay = 0.0
        
        for vehicle_id in vehicle.ids:
            if vehicle_id in final_times:
                entry_time = final_times[vehicle_id]
                arrival_time = vehicle.id_arrival_time[vehicle_id]
                delay = entry_time - arrival_time
                
                max_delay = max(max_delay, delay)
                min_delay = min(min_delay, delay)
                total_delay += delay
        
        avg_delay = total_delay / len(vehicle.ids)
        makespan = max(final_times.values()) + vehicle.travel_time
        
        return ResultRecord(
            nbofveh=len(vehicle.ids),
            gap=0.0,
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