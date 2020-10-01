# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    
    assumes w,y row arrays
    """
    # convert row arrays to matrices in correct shape
    w = np.matrix(w).T
    y = np.matrix(y).T
    
    # calculate e
    e =  y- np.dot(tx,w)
    
    #calculate loss
    loss =  np.dot(e.T,e)[0,0]/ 2 / y.shape[0] # MSE
    
    return loss

def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]

def grid_search(y, tx, w0, w1):
    """Algorithm for grid search.
    w0 and w1 are arrays 
    """
    losses = np.zeros((len(w0), len(w1)))
    for i0 in w0:
        for i1 in w1:
            losses[w0.tolist().index(i0),w1.tolist().index(i1)] = compute_loss(y,tx,[i0,i1])
    return losses