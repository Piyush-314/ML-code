import numpy as np

def reshape_and_transpose(x: np.ndarray, B: int, C: int, H: int, W: int) -> np.ndarray:
    """
    Reshapes flat x to (B, C, H, W) then transposes to (B, H, W, C).

    1) Reshape: 
        Reinterprets memory layout without moving data
        Convert the 1D vector into a 4D tensor of shape - NCHW
    2) Transpose: 
        Changes dimension order (affects memory stride)
        Rearrange dimensions to shape - NHWC

    B: Batch size (number of images)
    C: Number of channels (e.g., 3 for RGB, 1 for grayscale)
    H: Height of each image (number of rows)
    W: Width of each image (number of columns)
    """

    """
    NCHW (Channels First):
        Efficient for convolution operations (channels processed together)
        Used by PyTorch, cuDNN
        Better cache locality for channel-wise operations

    NHWC (Channels Last):
        Better for memory access (spatial operations and visualization)
        Natural for image processing workflows
        Used by TensorFlow (default), OpenCV
    """

    # 1. Validation check
    expected_size = B * C * H * W
    if x.size != expected_size:
        raise ValueError(f"Input size {x.size} does not match expected size {expected_size}")

    # Step 1: Reshape to NCHW (PyTorch style)
    # Memory interpretation: [B][C][H][W]
    # All channels for first pixel, then next pixel
    x_nchw = x.reshape(B, C, H, W)
    
    # 3. Transpose to NHWC (Batch, Height, Width, Channels)
    # Current axes: 0=B, 1=C, 2=H, 3=W
    # Target axes:  0=B, 2=H, 3=W, 1=C
    x_nhwc = x_nchw.transpose(0, 2, 3, 1)
    
    return x_nhwc


'''For a single RGB image (B=1, C=3, H=2, W=2):

NCHW: [batch][R/G/B channels][rows][columns]
NHWC: [batch][rows][columns][R/G/B channels]
The transpose operation swaps the channel dimension from position 1 to position 3.

    Example Walkthrough:Row-major vs Column-major:
        NumPy uses row-major (C-style): a[i, j] consecutive in memory
        MATLAB uses column-major (Fortran-style): a(j, i) consecutive
'''


"""
# Input: 1D vector with 24 elements
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
     13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

# Parameters
B = 1  # 1 image in batch
C = 2  # 2 channels
H = 3  # 3 rows (height)
W = 4  # 4 columns (width)

# Step 1: Reshape to (B, C, H, W) = (1, 2, 3, 4)
# This organizes data as: [batch][channel][height][width]
reshaped = [[[[1, 2, 3, 4],      # Channel 0, Row 0
              [5, 6, 7, 8],      # Channel 0, Row 1
              [9, 10, 11, 12]], # Channel 0, Row 2
             [[13, 14, 15, 16],  # Channel 1, Row 0
              [17, 18, 19, 20], # Channel 1, Row 1
              [21, 22, 23, 24]]]] # Channel 1, Row 2

# Step 2: Transpose to (B, H, W, C) = (1, 3, 4, 2)
# Rearrange: batch stays first, then height, width, channels last
result = [[[[1, 13],   [2, 14],   [3, 15],   [4, 16]],   # Row 0: all pixels with both channels
           [[5, 17],   [6, 18],   [7, 19],   [8, 20]],   # Row 1: all pixels with both channels
           [[9, 21],   [10, 22],  [11, 23],  [12, 24]]]] # Row 2: all pixels with both channels

# Final shape: (1, 3, 4, 2)
"""