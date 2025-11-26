"""
Main entry point for traffic intersection optimization experiments.

This module orchestrates the comparison of different optimization algorithms:
- Dynamic Programming (with and without pruning)
- Gurobi-based optimization methods
- Heuristic algorithms

Results are exported to CSV files for analysis.
"""

import os
import sys
from unittest import runner
import numpy as np
from typing import List

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import *
from utils.experiment_runner import ExperimentRunner

def main():
    """Main entry point."""
    try:
        runner = ExperimentRunner()
        
        runner.run_comparison_experiments()
        
        print("All experiments completed successfully!")
        print(f"Results saved to {RESULTS_DIR}/")
       
    except Exception as e:
        print(f"Experimental error: {e}")
        raise


if __name__ == "__main__":
    main()