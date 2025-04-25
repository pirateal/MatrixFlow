# GPU Matrix Operations Utilities
import cupy as cp

def to_low_precision(mat, fmt='fp8'):
    # convert matrix to low-precision format
    return mat.astype(cp.float16)  # placeholder

def tile_matrix(mat, block_size):
    # split into tiles
    shape = mat.shape
    return mat.reshape(shape[0]//block_size, block_size, -1, block_size)

if __name__ == "__main__":
    import numpy as np
    mat = cp.array(np.random.rand(64,64))
    print("Tiled:", tile_matrix(mat, 8).shape)
