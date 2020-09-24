# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

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