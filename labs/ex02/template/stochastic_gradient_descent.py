# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    return compute_gradient(y,tx,w) 



def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm.
    assumes y,w are arrays
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        # create batch of random datapoints
        
        batch_indices = np.random.randint(tx.shape[0],size = batch_size)
        batch_y = y[batch_indices]
        batch_tx = tx[batch_indices,:]
        # calculate loss & gradient
        loss = compute_loss(y,tx,w)
        gradient = compute_gradient(batch_y,batch_tx,w)

        # update w by gradient
        w = w - gamma*gradient

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
