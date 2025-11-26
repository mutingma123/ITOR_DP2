"""
CSV handling utilities for results export.

This module provides utilities for writing optimization results to CSV file with proper formatting and headers.
"""

import csv
import os
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ResultRecord:
    """Data class for optimization result records."""
    nbofveh: int = 0                    # Number of vehicles
    gap: float = 0.0                    # Optimization gap
    nbodnodes: float = 0.0              # Number of nodes explored
    comptime: float = 0.0               # Computation time (ms)
    nbodpla: int = 0                    # Number of platoons from lane A
    nbodplb: int = 0                    # Number of platoons from lane B
    nbofa: int = 0                      # Number of vehicles from lane A
    nbofb: int = 0                      # Number of vehicles from lane B
    avedelay: float = 0.0               # Average delay
    maxdelay: float = 0.0               # Maximum delay
    mindelay: float = 0.0               # Minimum delay
    makespan: float = 0.0               # Makespan
    mark: str = ""                      # Algorithm identifier
    espa: int = 0                       # Early separation platoon A
    espb: int = 0                       # Early separation platoon B
    nodes: int = 0                      # Theoretical node count
    truefalse: str = ""                 # Validation flag
    crosstime: float = 0.0              # Cross time parameter
    buffer: float = 0.0                 # Buffer parameter
    sath: float = 0.0                   # Saturation headway
    arrival: float = 0.0                # Arrival rate

class CSVHandler:
    """
    Handles CSV file operations for results data.
    
    This class provides methods to write optimization results to CSV files with proper headers and formatting.
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize CSV handler.
        
        Args:
            results_dir: Directory for output files
        """
        self.results_dir = results_dir
        self._ensure_results_dir()
        
        # Initialize file handles and writers
        self.results_file = None
        self.results_writer = None
    
    def _ensure_results_dir(self):
        """Ensure results directory exists."""
        os.makedirs(self.results_dir, exist_ok=True)
    
    def open_files(self, results_filename: str = "results.csv"):
        """
        Open CSV files for writing.
        
        Args:
            results_filename: Name of results file
        """
        results_path = os.path.join(self.results_dir, results_filename)
        
        self.results_file = open(results_path, 'w', newline='')
        
        # Create CSV writers
        self.results_writer = csv.DictWriter(
            self.results_file, 
            fieldnames=self._get_result_fieldnames()
        )
        
        # Write headers
        self.results_writer.writeheader()
    
    def close_files(self):
        """Close CSV files."""
        if self.results_file:
            self.results_file.close()
    
    def write_result_record(self, record: ResultRecord):
        """
        Write a result record to CSV.
        
        Args:
            record: ResultRecord instance to write
        """
        if self.results_writer and self.results_file:
            self.results_writer.writerow(record.__dict__)
            self.results_file.flush()  # Ensure data is written
    
    def _get_result_fieldnames(self) -> List[str]:
        """Get fieldnames for results CSV."""
        return [
            'nbofveh', 'gap', 'nbodnodes', 'comptime', 'nbodpla', 'nbodplb',
            'nbofa', 'nbofb', 'avedelay', 'maxdelay', 'mindelay', 'makespan',
            'mark', 'espa', 'espb', 'nodes', 'truefalse', 'crosstime', 'buffer',
            'sath', 'arrival'
        ]
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_files()