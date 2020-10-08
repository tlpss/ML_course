# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    matrix = np.ones((len(x),degree+1))
    matrix[:,1] = x
    for i in range(2,degree+1):
        matrix[:,i] = matrix[:,i-1]*matrix[:,1]
    return matrix