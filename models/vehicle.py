"""
Vehicle model for traffic intersection optimization.

This module contains the Vehicle class that represents vehicles approaching 
an intersection from different lanes with arrival times and constraints.
"""

import numpy as np
from typing import Dict, List
from config.settings import LANE_A_ID, LANE_B_ID, BIG_M_MULTIPLIER


class Vehicle:
    """
    Represents vehicles approaching a traffic intersection.
    
    This class manages vehicle data including arrival times, lane assignments,
    and intersection timing constraints. It serves as the core data model for
    all optimization algorithms.
    
    Attributes:
        ids (np.ndarray): Array of vehicle IDs
        cross_time (float): Time required to cross intersection
        buffer (float): Safety buffer time between conflicting vehicles
        sat_head (float): Saturation headway between vehicles from same lane
        big_m (float): Big-M value for optimization constraints
        travel_time (float): Travel time to reach intersection
        arrival_rate (float): Arrival rate parameter for distribution
        lane_a (int): Lane A identifier
        lane_b (int): Lane B identifier
        num_from_a (int): Number of vehicles from lane A
        num_from_b (int): Number of vehicles from lane B
        ids_a (List[int]): Vehicle IDs from lane A
        ids_b (List[int]): Vehicle IDs from lane B
        id_lane_map (Dict[int, int]): Mapping of vehicle ID to lane
        id_arrival_time (Dict[int, float]): Mapping of vehicle ID to arrival time
        num_platoons_a (int): Number of platoons from lane A
        num_platoons_b (int): Number of platoons from lane B
        inter_batch_gap (float): Time gap between batches
        max_batch_size (int): Maximum vehicles per batch
    """
    
    def __init__(self, vehicle_ids: np.ndarray, cross_time: float, buffer: float,
                 sat_head: float, lane_a_id: int, lane_b_id: int, 
                 travel_time: float, arrival_rate: float, 
                 inter_batch_gap: float = 0.0, max_batch_size: int = 0):
        """
        Initialize Vehicle instance.
        
        Args:
            vehicle_ids: Array of unique vehicle identifiers
            cross_time: Time to cross intersection
            buffer: Safety buffer time
            sat_head: Saturation headway
            lane_a_id: Lane A identifier
            lane_b_id: Lane B identifier  
            travel_time: Travel time to intersection
            arrival_rate: Arrival rate for distribution
            inter_batch_gap: Gap between batches (default: 0.0)
            max_batch_size: Maximum batch size (default: 0)
        """
        self.ids = vehicle_ids
        self.cross_time = cross_time
        self.buffer = buffer
        self.sat_head = sat_head
        self.travel_time = travel_time
        self.arrival_rate = arrival_rate
        self.lane_a = lane_a_id
        self.lane_b = lane_b_id
        self.inter_batch_gap = inter_batch_gap
        self.max_batch_size = max_batch_size
        
        # Initialize vehicle distribution data
        self.num_from_a = 0
        self.num_from_b = 0
        self.ids_a: List[int] = []
        self.ids_b: List[int] = []
        self.id_lane_map: Dict[int, int] = {}
        self.id_arrival_time: Dict[int, float] = {}
        
        # Platoon counting
        self.num_platoons_a = 0
        self.num_platoons_b = 0
        
        # Big-M value (will be set after arrival times are generated)
        self.big_m = 0.0
        self.big_m2 = 0.0
        self.small_m = 5.0
        
        # Objective weights (will be calculated after arrival times)
        self.weight1 = 0.0
        self.weight2 = 0.0
        
        # Special platoon indicators for early separation optimization
        self.early_sep_platoon_a = 0
        self.early_sep_platoon_b = 0
    
    def update_big_m_values(self):
        """
        Update big-M constraint values based on arrival times.
        
        Called after arrival times are set to calculate appropriate
        big-M values for optimization constraints.
        """
        if self.id_arrival_time:
            max_arrival = max(self.id_arrival_time.values())
            self.big_m = max_arrival + len(self.ids) * (self.cross_time + self.buffer)
            self.big_m2 = self.big_m
            
            # Calculate objective weights
            self.weight1 = 1.0 / self.big_m
            self.weight2 = 1.0 / len(self.ids)
    
    def update_early_separation_indicators(self):
        """
        Update early separation platoon indicators.
        
        Determines if the first vehicles from each lane can be separated
        early based on arrival time differences and timing constraints.
        """
        if (len(self.ids_a) > 0 and len(self.ids_b) > 0 and 
            self.id_arrival_time):
            
            first_a_arrival = self.id_arrival_time[self.ids_a[0]]
            first_b_arrival = self.id_arrival_time[self.ids_b[0]]
            
            time_diff_threshold = self.buffer + self.cross_time - self.sat_head
            
            if first_a_arrival - first_b_arrival >= time_diff_threshold:
                self.early_sep_platoon_a = 1
            
            if first_b_arrival - first_a_arrival >= time_diff_threshold:
                self.early_sep_platoon_b = 1
    
    def get_vehicle_count(self) -> int:
        """Get total number of vehicles."""
        return len(self.ids)
    
    def get_lane_distribution(self) -> tuple:
        """Get distribution of vehicles across lanes."""
        return self.num_from_a, self.num_from_b
    
    def get_platoon_counts(self) -> tuple:
        """Get number of platoons from each lane."""
        return self.num_platoons_a, self.num_platoons_b