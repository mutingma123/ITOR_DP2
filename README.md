# Traffic Intersection Optimization with Dynamic Programming ([![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.17724051.svg)](https://doi.org/10.5281/zenodo.17724051))

This project implements traffic intersection optimization algorithms comparing dynamic programming approaches with state-of-the-art optimization methods using Gurobi.

## Project Structure

```
├── main.py                    # Main entry point and experiment runner
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── __init__.py                # Package initialization file
├── config/                    # Configuration files
│   ├── __init__.py
│   └── settings.py           # Global settings and parameters
├── models/                    # Core data models
│   ├── __init__.py
│   ├── vehicle.py            # Vehicle class with timing constraints
│   └── environment.py        # Environment setup and scenario generation
├── algorithms/                # Optimization algorithms (to be implemented)
│   ├── __init__.py
│   ├── dynamic_programming.py
│   ├── gurobi_optimizer.py
│   └── heuristics.py
├── utils/                     # Utility functions (to be implemented)
│   ├── __init__.py
│   ├── csv_handler.py        # CSV file I/O operations
│   └── r_integration.py      # Statistical distribution utilities (Python-based)
└── results/                   # Output directory (generated at runtime)
```

## Features

- Dynamic Programming algorithm for traffic intersection scheduling
- Gurobi-based optimization methods (makespan, delay minimization)
- FCFS (First Come First Serve) heuristic
- Batch processing heuristic
- Python-based statistical distribution generation using scipy

## Usage

```bash
python main.py
```

## Algorithms Implemented

1. **Dynamic Programming**: Efficient state-space search for optimal vehicle scheduling
2. **Gurobi Optimizers**: 
   - Makespan minimization
   - Maximum delay minimization
   - Sum delay minimization
3. **Heuristics**:
   - FCFS (First Come First Serve)
   - Batch processing

## Output

Results are saved to CSV files containing performance metrics including:
- Processing time
- Node count (for tree-based methods)
- Average, maximum, and minimum delays
- Makespan
- Gap (for optimization methods)
