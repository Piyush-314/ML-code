import numpy as np

def compute_norms(x):
    # x shape is (N, D)

    """
    compute_norms(x) that takes a matrix x of shape 
    (N,D)(N,D) representing a batch of N vectors, each of dimension D, and returns a dictionary with two fundamental norms:
    """
    
    # L1 Norm "l1": The L1 norm (Manhattan norm, taxicab norm) for each vector
    # Formula: Sum of absolute values
    # We apply abs(), then sum across columns (axis=1)
    l1 = np.sum(np.abs(x), axis=1)
    
    # L2 Norm "l2": The L2 norm (Euclidean norm) for each vector
    # Formula: Sqrt of sum of squares
    # 1. Square every element
    # 2. Sum across columns (axis=1)
    # 3. Take square root
    l2 = np.sqrt(np.sum(x**2, axis=1))
    
    return {
        "l1": l1, 
        "l2": l2
    }

"""
L1 Regularization (Lasso): λ∥w∥1, encourages sparse models with fewer features
    Because the L1 norm is calculated using absolute values, its derivative is constant. This creates a pressure that pushes useless weights exactly to zero.
    This is useful for Feature Selection. If you have 1,000 features but only 10 matter, L1 regularization will likely turn the other 990 weights to 0.

L2 Regularization (Ridge/Weight Decay): λ∥w∥2 prevents weights from growing too large
    The L2 norm is calculated using squares, so its derivative grows with the weight values. This creates a pressure that keeps all weights small but rarely zeroes them out completely.
    Minimize the error, but don't use massive numbers to do it.
    This is useful for preventing overfitting by keeping the model weights small and smooth.

Gradient Clipping: Clip gradients if ∥∇L∥2 > threshold to prevent exploding gradients
    We check the L2 Norm of the gradient. If norm > 1.0, we scale the whole vector down so its size is 1.0. This ensures stability.

Normalization: Normalize features or weights: x norm=x/∥x∥^2

Loss Functions: Many losses incorporate norms, e.g., mean squared error uses L2 norm
"""


# Example usage:
"""
x = [[3, 4],      # First vector: (3, 4)
     [1, -1]]     # Second vector: (1, -1)

# L1 Norm (Manhattan distance):
# Vector 0: |3| + |4| = 3 + 4 = 7
# Vector 1: |1| + |-1| = 1 + 1 = 2
l1_result = [7, 2]

# L2 Norm (Euclidean distance):
# Vector 0: sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5.0
# Vector 1: sqrt(1² + (-1)²) = sqrt(1 + 1) = sqrt(2) ≈ 1.414
l2_result = [5.0, 1.414]
"""