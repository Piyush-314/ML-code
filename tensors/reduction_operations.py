import numpy as np
from typing import Dict, Union

def tensor_reductions(x: np.ndarray, axis: int) -> Dict[str, Union[np.ndarray, float]]:
    """
    Computes sum, mean, max, argmax along axis.

    tensor_reductions(x, axis) that takes a tensor x and an axis parameter, then returns a dictionary with four essential reduction operations:
    """

    """
    The axis parameter determines which dimension to reduce:

        axis=0: Reduce along rows (first dimension)
        axis=1: Reduce along columns (second dimension)
        axis=-1: Reduce along the last dimension
        axis=None: Reduce all dimensions to a scalar

    """
    results = {}

    # SUM: Total accumulation
    # Theory: Used in gradient accumulation during backprop
    results["sum"] = np.sum(x, axis=axis)
    
    # MEAN: Average value
    # Theory: Most loss functions use mean (MSE, CrossEntropy)
    # Why not sum? Mean normalizes by batch size → stable gradients
    results["mean"] = np.mean(x, axis=axis)
    
    # MAX: Highest value
    # Theory: Used in max pooling (CNNs) and ReLU
    # Gradient: Only max element gets gradient (others get 0)
    results["max"] = np.max(x, axis=axis)
    
    # ARGMAX: Index of highest value
    # Theory: Used in classification (predicting class labels)
    # Critical for accuracy calculation
    results["argmax"] = np.argmax(x, axis=axis)
    
    
    return results
    

"""
Mean reduction in loss: Each sample contributes equally to gradient
Max reduction in pooling: Gradient only flows through max element (sparse gradient)
"""


"""
x = [[1, 2, 3], 
     [4, 5, 6]]  # Shape (2, 3)
axis = 1  # Reduce along columns (second dimension)

# Sum along rows: For each row, sum all columns
# Row 0: 1 + 2 + 3 = 6
# Row 1: 4 + 5 + 6 = 15
sum_result = [6, 15]  # Shape (2,)

# Mean along rows: Average of each row
# Row 0: (1 + 2 + 3) / 3 = 6 / 3 = 2.0
# Row 1: (4 + 5 + 6) / 3 = 15 / 3 = 5.0
mean_result = [2.0, 5.0]  # Shape (2,)

# Max along rows: Maximum value in each row
# Row 0: max(1, 2, 3) = 3
# Row 1: max(4, 5, 6) = 6
max_result = [3, 6]  # Shape (2,)

# Argmax along rows: Index of maximum in each row
# Row 0: argmax([1, 2, 3]) = 2 (index of value 3)
# Row 1: argmax([4, 5, 6]) = 2 (index of value 6)
argmax_result = [2, 2]  # Shape (2,)
"""

