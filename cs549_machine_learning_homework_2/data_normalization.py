#!/miniconda3/envs/class/bin/python

"""Main file for homework #2 for CS-549 Machine Learning."""

__author__ = "Reina Hastings"
__email__ = "reinahastings13@gmail.com"

import numpy as np

""" rescale_normalization() --- Rescale an array to a normalized range [0, 1].

Rescales each value in dataArray so that the minimum value in the array becomes 0 and the maximum value becomes 1. 
Transformation is done by subtracting the minimum value from each data point and then dividing by the range (max - min).

Parameters:
    data_array (array-like): Array of numerical data to be normalized.

Returns:
    new_values (list): A list of normalized values in the range [0, 1].
"""
def rescale_normalization(dataArray):
    min = dataArray.min()
    denom = dataArray.max() - min
    new_values = []
    for x in dataArray:
        newX = (x - min) / denom
        new_values.append(newX)
    return new_values

""" rescale_matrix --- Rescale each column in a matrix to a normalized range [0, 1].

Normalizes each column of data_matrix so that the minimum value in each column becomes 0 and the maximum becomes 1.
Normalization is done by subtracting the column minimum and dividing by the column range (max - min).

Parameters:
    data_matrix (2D array-like) - Matrix of numerical data where each column is normalized independently.

Returns:
    new_matrix (2D array-like) - Matrix with normalized values in the range [0, 1] for each column.
"""
def rescale_matrix(data_matrix):
    col_count = len(data_matrix[0])
    row_count = len(data_matrix)
    new_matrix = np.zeros(data_matrix.shape) 
    for i in range(0, col_count):
        min = data_matrix[:,i].min()
        denom = data_matrix[:,i].max() - min
        for k in range(0, row_count):
            newX = (data_matrix[k,i] - min) / denom
            new_matrix[k,i] = newX
    return new_matrix

""" mean_normalization() --- Normalize an array using mean normalization.

Applies mean normalization to data_array, transforming the values so that they are centered around zero, based on the mean of the array. Each value is adjusted by subtracting the mean and dividing by the range (max - min) of the array.

Parameters:
    data_array (1D array-like) - Array of numerical data to be normalized.

Returns:
    new_values (1D array-like) - List of normalized values where the mean is 0 and values are scaled between -1 and 1, depending on the data range.
"""
def mean_normalization(data_array):
    mean = np.mean(data_array)
    denom = data_array.max() - data_array.min()
    new_values = []
    for x in data_array:
        newX = (x - mean) / denom
        new_values.append(newX)
    return new_values
