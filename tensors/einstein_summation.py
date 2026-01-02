import numpy as np
from typing import Dict, Union

def einsum_ops(A: np.ndarray, B: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
    """
    Computes basic ops using np.einsum.
    
    Args:
        A: (N, D)
        B: (D, M)
        
    Returns:
        Dict with "transpose", "sum", "row_sum", "col_sum", "matmul"
    """
    
    results = {}
    
    # 1. TRANSPOSE: Swap dimensions
    # "ij->ji": Take element (i,j) to position (j,i)
    results["transpose"] = np.einsum("ij->ji", A)
    
    # 2. SUM ALL: Complete reduction
    # "ij->": Sum over all indices i and j
    results["sum"] = np.einsum("ij->", A)
    
    # 3. ROW SUM: Reduce columns
    # "ij->i": Keep i, sum over j
    results["row_sum"] = np.einsum("ij->i", A)
    
    # 4. COL SUM: Reduce rows  
    # "ij->j": Keep j, sum over i
    results["col_sum"] = np.einsum("ij->j", A)
    
    # 5. MATRIX MULTIPLICATION: The heart of neural networks
    # "ik,kj->ij": Sum over k (contraction)
    # Theory: This is a linear layer: Y = XW
    results["matmul"] = np.einsum("ik,kj->ij", A, B)
    
    return results

"""
# B=Batch, H=Heads, T=Time(Seq), D=Dim
# Query: BHTD, Key: BHTD
# We want attention scores: BHTT (Sequence vs Sequence for each Head)
    scores = np.einsum('bhtd, bhkd -> bhtk', Q, K)
"""

"""
Neural Network Forward Pass:
Input X → Linear Layer → Activation → Output
          ↑
        WX + b = einsum("ik,kj->ij", X, W) + b
"""

