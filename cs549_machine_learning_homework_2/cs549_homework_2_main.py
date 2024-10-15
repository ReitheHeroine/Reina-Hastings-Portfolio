#!/miniconda3/envs/class/bin/python

"""Main file for homework #2 for CS-549 Machine Learning"""

__author__ = "Reina Hastings"
__email__ = "reinahastings13@gmail.com"

from download_data import download_data
import numpy as np
import matplotlib.pyplot as plt
from GD import gradient_descent
from data_normalization import rescale_matrix

def main():
    # Parameters
    # Use to test multiple learning rates and report their convergence curves. 
    ALPHA = 0.5
    MAX_ITER = 50

    # Step-1: load data and divide it into two subsets, used for training and testing
    sat = download_data('sat.csv', [1, 2, 4]).values # three columns: MATH SAT, VERB SAT, UNI. GPA  # convert frame to matrix

    # Normalize data
    sat = normalize_data_standardization(sat)

    # Training data;
    satTrain = sat[0:60, :]
    # Testing data; 
    satTest = sat[60:len(sat),:]

    # Step-2: train a linear regression model using the Gradient Descent (GD) method
    # Note theta and xValues have 3 columns since have 2 features: y = (theta * x^0) + (theta * x^1) + (theta * x^2)
    theta = np.zeros(3) 

    xValues = np.ones((60, 3)) 
    xValues[:, 1:3] = satTrain[:, 0:2]
    yValues = satTrain[:, 2]
    # Call the GD algorithm - gradientDescent()
    [theta, arrCost] = gradient_descent(xValues, yValues, theta, ALPHA, MAX_ITER)
    print(arrCost)

    
    # Visualize the convergence curve
    plt.plot(range(0,len(arrCost)),arrCost);
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.title('alpha = {}  theta = {}'.format(ALPHA, theta))
    plt.show()

    # Step-3: testing
    testXValues = np.ones((len(satTest), 3)) 
    testXValues[:, 1:3] = satTest[:, 0:2]
    tVal =  testXValues.dot(theta)
    

    # Step-4: evaluation
    # Calculate average error and standard deviation
    tError = np.sqrt([x**2 for x in np.subtract(tVal, satTest[:, 2])])
    print('results: {} ({})'.format(np.mean(tError), np.std(tError)))

# Standardize the dataset (subtract the mean and divide by the standard deviation)
def normalize_data_standardization(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = (data - mean) / std_dev
    return normalized_data

if __name__ == "__main__":
        main()