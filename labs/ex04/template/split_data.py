# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    
    ! convention different from Project 1 implementation!
    """
    # set seed
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(x))
    test_set_size = int(len(x)*ratio)
    test_indices = shuffled_indices[test_set_size:]
    train_indices =shuffled_indices[:test_set_size]
    
    return x[train_indices],y[train_indices],x[test_indices],y[test_indices]
