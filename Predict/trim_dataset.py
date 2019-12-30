"""
    Assume our dataset has 53 samples and our batch size is 25. Then, we have to remove(trim) the remaining samples.
    This function does just that.
"""


def trim_dataset(matrix, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = matrix.shape[0] % batch_size
    if(no_of_rows_drop > 0):
        return matrix[:-no_of_rows_drop]
    else:
        return matrix