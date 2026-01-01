import numpy as np
from typing import Dict, Union

def tensor_reductions(x: np.ndarray, axis: int) -> Dict[str, Union[np.ndarray, float]]:
    """
    Computes sum, mean, max, argmax along axis.

    tensor_reductions(x, axis) that takes a tensor x and an axis parameter, then returns a dictionary with four essential reduction operations:
    """

    '''
    "sum": Sum of elements along the specified axis
        Used in loss functions, gradient accumulation, and feature aggregation
        Shape: Reduces the dimension at axis to size 1 (or removes it if keepdims=False)

    "mean": Mean (average) of elements along axis
        mean= 1/n∑ximean = 1/n∑xi
        Used in loss functions (MSE, MAE), average pooling, and normalization statistics
        Shape: Same reduction as sum

    "max": Maximum value along axis
        Finds the largest element in the specified dimension
        Used in max pooling, ReLU (which is essentially max(0, x)), and attention mechanisms
        Shape: Same reduction as sum

    "argmax": Index of the maximum value along axis
        Critical for classification: converting logits to predicted class labels
        Used in finding the most activated feature, selecting the best action in reinforcement learning
        Shape: Same reduction as sum, but returns integer indices

    '''

    """
    The axis parameter determines which dimension to reduce:

        axis=0: Reduce along rows (first dimension)
        axis=1: Reduce along columns (second dimension)
        axis=-1: Reduce along the last dimension
        axis=None: Reduce all dimensions to a scalar

    """
    #
    results = {
        "sum": np.sum(x, axis=axis),
        "mean": np.mean(x, axis=axis),
        "max": np.max(x, axis=axis),
        "argmax": np.argmax(x, axis=axis)
    }
    
    return results
    


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

