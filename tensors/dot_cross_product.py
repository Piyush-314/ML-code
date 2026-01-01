import numpy as np
from typing import Dict

def vector_products(a: np.ndarray, b: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes dot and cross products for batches of 3D vectors.
    
    Args:
        a: Shape (N, 3)
        b: Shape (N, 3)
        
    Returns:
        Dict with "dot" (N,) and "cross" (N, 3)
    """
    
    # --- 1. Dot Product ---
    # Formula: sum(a_i * b_i)
    # We multiply element-wise, then sum across the columns (axis=1)
    # This gives us (N,) shape
    dot = np.sum(a*b, axis=1)


    # --- 2. Cross Product ---
    # We need to access components. 
    # a[:, 0] is all x-coords, a[:, 1] is all y-coords, etc.
    ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
    bx, by, bz = b[:, 0], b[:, 1], b[:, 2]
    
    # Apply the formula component-wise
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx

    # Stack them back together into (N, 3)
    # np.stack connects arrays along a new axis
    cross = np.stack([cx, cy, cz], axis=1)
    
    return {
        "dot": dot,
        "cross": cross
    }


"""
In Transformers, attention scores are computed as dot products:
    You have a Query (what you are looking for) and a Key (the content of the database).
    Attention Score = Dot(Query, Key).
    If the dot product is high, the network "pays attention" to that information. If it's 0 or negative, it ignores it.
    Higher dot product = stronger attention
"""