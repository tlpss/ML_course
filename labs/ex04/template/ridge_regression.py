# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
def compute_loss(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    n,d = tx.shape
    a = np.dot(tx.T,tx)+lambda_*2*n*np.identity(d)
    b = np.dot(tx.T,y)
    w_opt = np.linalg.solve(a,b)
    loss = compute_loss(y,tx,w_opt)
    return w_opt, loss
