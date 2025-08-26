#!/miniconda3/envs/class/bin/python

"""gradient_descent function for homework #2 for CS-549 Machine Learning"""

__author__ = "Reina Hastings"
__email__ = "reinahastings13@gmail.com"

import numpy as np
# X          - single array/vector
# y          - single array/vector
# theta      - single array/vector
# alpha      - scalar
# iterations - scarlar

""" gradient_descent --- Perform Gradient Descent to minimize the cost function for linear regression.

Iteratively updates the parameters (theta) of a linear regression model using gradient descent. It returns the optimized parameters and the cost values at each iteration.

Parameters:
        X (numpy.ndarray): The matrix of input features (with a bias term).
        y (numpy.ndarray): The vector of target values.
        theta (numpy.ndarray): The initial parameter values.
        alpha (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for the gradient descent algorithm.

Returns:
        tuple: 
        - theta (numpy.ndarray): The optimized parameters after gradient descent.
        - arr_cost (list): A list of cost function values at each iteration, useful for tracking convergence.
"""
def gradient_descent(X, y, theta, alpha, num_iterations):
        m = len(y)
        arr_cost =[];
        transposedX = np.transpose(X) # transpose X into a vector  -> XColCount X m matrix
        for interation in range(0, num_iterations):
                # Calculate the hypothesis
                hypothesis = X.dot(theta)  # h_theta(x) = X * theta

                # Calculate the error
                errors = hypothesis - y  # h_theta(x) - y

                # Compute the gradient
                gradient = (1 / m) * transposedX.dot(errors)  # gradient calculation

                # Update theta with the learning rate and gradient
                theta = theta - alpha * gradient

                # Calculate the current cost with the new theta
                hypothesis = X.dot(theta)

                # Calculate the errors (hypothesis - actual values)
                errors = hypothesis - y

                # Calculate the cost function (Mean Squared Error)
                atmp = (1 / (2 * m)) * np.sum(errors ** 2)

                # Print the current cost for debugging (optional)
                print(atmp)

                # Append the cost to the arr_cost list to keep track of cost for each iteration
                arr_cost.append(atmp)

                return theta, arr_cost
