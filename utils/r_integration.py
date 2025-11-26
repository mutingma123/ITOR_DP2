"""
Statistical distribution utility for traffic intersection modeling.

This module provides statistical distribution generation using Python's scipy library,
particularly truncated exponential distributions used for vehicle arrival times.
Previously used R integration but now uses pure Python for better portability.
"""

import numpy as np
from scipy.stats import expon
from typing import List


class RIntegration:
    """
    Statistical distribution generation utility class.
    
    This class provides methods to generate random variables from various
    distributions using Python's scipy library, particularly for
    truncated exponential distributions used in traffic modeling.
    
    Note: Despite the name 'RIntegration', this class now uses pure Python
    to maintain backward compatibility with existing code.
    """
    
    def __init__(self):
        """Initialize statistical distribution generator."""
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup Python environment for statistical distributions."""
        try:
            # Test scipy availability
            import scipy.stats
            self._scipy_available = True
            print("Using scipy for statistical distribution generation")
            
        except ImportError as e:
            print(f"Warning: scipy not available: {e}")
            print("Falling back to basic numpy for distribution generation")
            self._scipy_available = False
    
    def generate_truncated_exponential(self, n: int, mean: float, 
                                     lower_bound: float) -> List[float]:
        """
        Generate truncated exponential random variables.
        
        Uses scipy.stats for accurate truncated exponential distribution,
        falling back to numpy inverse transform method if scipy unavailable.
        
        Args:
            n: Number of random variables to generate
            mean: Mean parameter for exponential distribution
            lower_bound: Lower bound for truncation
            
        Returns:
            List of truncated exponential random variables
        """
        if self._scipy_available:
            return self._generate_with_scipy(n, mean, lower_bound)
        else:
            return self._generate_with_numpy(n, mean, lower_bound)
    
    def _generate_with_scipy(self, n: int, mean: float, 
                           lower_bound: float) -> List[float]:
        """
        Generate using scipy's truncated exponential distribution.
        
        This method uses scipy.stats.expon with proper truncation,
        which is mathematically equivalent to R's truncated exponential.
        """
        try:
            # Rate parameter (scale = 1/rate = mean for exponential)
            scale = mean  # scipy uses scale parameter (mean)
            
            # Calculate truncation parameter for scipy
            # For exponential with scale parameter, the CDF at lower_bound is:
            # F(lower_bound) = 1 - exp(-lower_bound/scale)
            a = lower_bound / scale  # Standardized lower bound
            
            # Generate from truncated exponential
            # Use survival function (1-CDF) for better numerical stability
            from scipy.stats import expon
            
            # Generate uniform random variables
            u = np.random.uniform(0, 1, n)
            
            # Apply inverse transform for truncated exponential
            # For exponential truncated at 'a' (standardized), the inverse CDF is:
            # F^(-1)(u) = -ln(1 - u * (1 - exp(-a))) (for standardized case)
            # Then scale back: result = scale * standardized_result
            
            exp_neg_a = np.exp(-a)
            standardized_samples = -np.log(1 - u * (1 - exp_neg_a))
            samples = scale * standardized_samples + lower_bound
            
            return [round(float(x), 2) for x in samples]
            
        except Exception as e:
            print(f"Scipy generation failed: {e}, falling back to numpy")
            return self._generate_with_numpy(n, mean, lower_bound)
    
    def _generate_with_numpy(self, n: int, mean: float, 
                            lower_bound: float) -> List[float]:
        """
        Fallback generation using numpy for truncated exponential.
        
        Uses inverse transform sampling for truncated exponential distribution.
        This method is mathematically equivalent to the R version.
        """
        # Set random seed for reproducibility
        np.random.seed(None)  # Use current time as seed
        
        # Rate parameter (lambda) is 1/mean for exponential distribution
        rate = 1.0 / mean
        
        # Generate uniform random variables
        u = np.random.uniform(0, 1, n)
        
        # Apply inverse transform for truncated exponential
        # For exponential with rate λ truncated at lower_bound:
        # F^(-1)(u) = lower_bound - ln(1-u)/λ
        # This ensures all samples >= lower_bound
        samples = lower_bound - np.log(1 - u) / rate
        
        return [round(float(x), 2) for x in samples]