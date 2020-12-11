#!/usr/bin/env python
# coding: utf-8


def get_mse(pred, actual):
        # Ignore nonzero terms.
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        
        return mean_squared_error(pred, actual)
    
def get_mae(pred, actual):
        # Ignore nonzero terms.
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        
        return mean_absolute_error(pred, actual)

