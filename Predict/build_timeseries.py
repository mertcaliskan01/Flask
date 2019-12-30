import numpy as np

from tqdm import tqdm_notebook


def build_timeseries(matrix, y_col_index, TIME_STEPS):
    # y_col_index is the index of my output column e.g y_col_index = 3 would be a Close Price column
    # total number of time-series samples would be len(matrix) - TIME_STEPS
    dim_0 = matrix.shape[0] - TIME_STEPS     # matrix.shape[0] is how many rows
    dim_1 = matrix.shape[1]                  # matrix.shape[1] is how many columns
    x = np.zeros((dim_0, TIME_STEPS, dim_1)) # Initializing timeseries input and output 
    y = np.zeros((dim_0,))
    
    # Constructing timeseries input matrix x and output vector y
    for i in tqdm_notebook(range(dim_0)):
        x[i] = matrix[i:TIME_STEPS + i]
        y[i] = matrix[TIME_STEPS + i, y_col_index]

    print("Timeseries input shape:", x.shape, "Output shape:", y.shape)
    return x, y