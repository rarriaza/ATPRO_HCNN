import numpy as np


def get_error(y, yh):
    # Threshold
    yht = np.zeros(np.shape(yh))
    yht[np.arange(len(yh)), yh.argmax(1)] = 1
    # Evaluate Error
    error = np.count_nonzero(np.count_nonzero(y-yht, 1))/len(y)
    return error
