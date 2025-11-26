"""
Environment model for traffic intersection simulation.

This module handles the generation of vehicle distributions and arrival times
using random distributions and R integration for statistical functions.
"""

import random
import numpy as np
from typing import List
from models.vehicle import Vehicle
from utils.r_integration import RIntegration


class Environment:
    """
    Manages the traffic intersection environment.
    
    This class is responsible for:
    1. Assigning vehicles to lanes randomly
    2. Generating arrival times using statistical distributions
    3. Counting platoons based on saturation headway
    4. Setting up the intersection scenario for optimization
    
    The environment ensures that both lanes have at least one vehicle
    and generates realistic arrival time distributions.
    """
    
    def __init__(self, r_integration: RIntegration = None):
        """
        Initialize Environment.
        
        Args:
            r_integration: R integration utility for statistical distributions
        """
        self.r_integration = r_integration or RIntegration()
        self.random = random.Random()
    
    def assign_vehicles_to_lanes(self, vehicle: Vehicle) -> None:
        """
        Randomly assign vehicles to lanes ensuring both lanes have vehicles.
        
        This method assigns each vehicle to either lane A or lane B randomly,
        but ensures that both lanes receive at least one vehicle. If the random
        assignment results in an empty lane, the assignment is repeated.
        
        Args:
            vehicle: Vehicle instance to configure
            
        Modifies:
            vehicle.ids_a: List of vehicle IDs assigned to lane A
            vehicle.ids_b: List of vehicle IDs assigned to lane B  
            vehicle.num_from_a: Count of vehicles from lane A
            vehicle.num_from_b: Count of vehicles from lane B
            vehicle.id_lane_map: Mapping of vehicle ID to lane
        """
        vehicle.num_from_a = 0
        vehicle.num_from_b = 0
        vehicle.ids_a.clear()
        vehicle.ids_b.clear()
        vehicle.id_lane_map.clear()
        
        # Repeat until both lanes have at least one vehicle
        while len(vehicle.ids_a) == 0 or len(vehicle.ids_b) == 0:
            vehicle.ids_a.clear()
            vehicle.ids_b.clear()
            vehicle.id_lane_map.clear()
            vehicle.num_from_a = 0
            vehicle.num_from_b = 0
            
            for vehicle_id in vehicle.ids:
                # Randomly assign to lane A or B
                assigned_lane = self.random.choice([vehicle.lane_a, vehicle.lane_b])
                
                if assigned_lane == vehicle.lane_a:
                    vehicle.num_from_a += 1
                    vehicle.ids_a.append(vehicle_id)
                    vehicle.id_lane_map[vehicle_id] = vehicle.lane_a
                else:
                    vehicle.num_from_b += 1
                    vehicle.ids_b.append(vehicle_id)
                    vehicle.id_lane_map[vehicle_id] = vehicle.lane_b
    
    def generate_arrival_times(self, mean: float, lower_bound: float, 
                             vehicle: Vehicle) -> None:
        """
        Generate vehicle arrival times using truncated exponential distribution.
        
        This method:
        1. Generates inter-arrival times using R's truncated exponential distribution
        2. Converts to cumulative arrival times
        3. Counts platoons based on saturation headway
        4. Updates vehicle timing parameters
        
        Args:
            mean: Mean parameter for exponential distribution
            lower_bound: Lower bound for truncation
            vehicle: Vehicle instance to configure
            
        Modifies:
            vehicle.id_arrival_time: Arrival times for each vehicle
            vehicle.num_platoons_a: Number of platoons from lane A
            vehicle.num_platoons_b: Number of platoons from lane B
            vehicle.big_m: Big-M constraint values
        """
        vehicle.early_sep_platoon_a = 0
        vehicle.early_sep_platoon_b = 0
        
        # Generate arrival times for lane A vehicles
        if vehicle.num_from_a > 0:
            inter_arrivals_a = self.r_integration.generate_truncated_exponential(
                vehicle.num_from_a, mean, lower_bound
            )
            
            # Convert to cumulative arrival times and count platoons
            self._process_lane_arrivals(
                vehicle.ids_a, inter_arrivals_a, vehicle
            )
        
        # Generate arrival times for lane B vehicles  
        if vehicle.num_from_b > 0:
            inter_arrivals_b = self.r_integration.generate_truncated_exponential(
                vehicle.num_from_b, mean, lower_bound
            )
            
            # Convert to cumulative arrival times and count platoons
            self._process_lane_arrivals(
                vehicle.ids_b, inter_arrivals_b, vehicle
            )
        
        # Update constraint parameters
        vehicle.update_big_m_values()
        vehicle.update_early_separation_indicators()
    
    def _process_lane_arrivals(self, lane_ids: List[int], 
                              inter_arrivals: List[float], 
                              vehicle: Vehicle) -> int:
        """
        Process arrival times for a specific lane and count platoons.
        
        Args:
            lane_ids: Vehicle IDs for this lane
            inter_arrivals: Inter-arrival times from distribution
            vehicle: Vehicle instance being configured
            
        Returns:
            Number of platoons formed in this lane
        """
        platoon_count = 0
        cumulative_time = 0.0
        platoon_start_time = 0.0
        vehicles_in_platoon = 1
        
        for i, vehicle_id in enumerate(lane_ids):
            cumulative_time += round(inter_arrivals[i], 2)
            vehicle.id_arrival_time[vehicle_id] = cumulative_time
            
            if i == 0:
                platoon_start_time = cumulative_time
            else:
                # Check if vehicle can join current platoon
                expected_time = (platoon_start_time + 
                               vehicle.sat_head * vehicles_in_platoon)
                
                if expected_time >= cumulative_time:
                    # Vehicle joins current platoon
                    platoon_count += 1
                    vehicles_in_platoon += 1
                else:
                    # Start new platoon
                    platoon_start_time = cumulative_time
                    vehicles_in_platoon = 1
        
        # Update platoon count for the appropriate lane
        if lane_ids == vehicle.ids_a:
            vehicle.num_platoons_a = platoon_count
        else:
            vehicle.num_platoons_b = platoon_count
        
        return platoon_count
    
    def setup_intersection_scenario(self, vehicle: Vehicle, mean: float, 
                                   lower_bound: float) -> None:
        """
        Complete setup of intersection scenario.
        
        This is the main method that orchestrates the full environment setup:
        1. Assigns vehicles to lanes
        2. Generates arrival times
        3. Configures all vehicle parameters
        
        Args:
            vehicle: Vehicle instance to configure
            mean: Mean for arrival time distribution
            lower_bound: Lower bound for arrival time distribution
        """
        self.assign_vehicles_to_lanes(vehicle)
        self.generate_arrival_times(mean, lower_bound, vehicle)