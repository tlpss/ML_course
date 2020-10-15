# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution with MSE."""
    # ***************************************************
    # returns mse, and optimal weights
    # ***************************************************
    #w_opt = np.dot(np.dot(np.linalg.inv(np.dot(tx.T,tx)),tx.T),y.T) # cf slides 3a
    a = np.dot(tx.T,tx)
    b = np.dot(tx.T,y)
    w_opt = np.linalg.solve(a,b)
    
    return w_opt
