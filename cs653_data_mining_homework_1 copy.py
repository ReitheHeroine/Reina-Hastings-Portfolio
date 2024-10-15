#!/anaconda3/env/hop/python

"""Homework #1 for CS-653"""

__author__ = "Reina Hastings"
__email__ = "reinahastings13@gmail.com"

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def main():
        # Part 1: Function takes positive integer N and returns the result of the factorial of N
        N = input('Part 1: Enter an integer: ')

        # Make sure N is an integer
        try:
                N = int(N)
        except:
                print('That is not an integer.')

        print(factorial(N))

        # Part 2: Function takes a sequence of integer values (seq) and determines if there is a distinct pair of numbers in the sequence whose product is odd
        seq = input('Part 2: Enter a sequence of integers with a space between each integer: ')
        print(odd_pair(seq))

        # Part 3: Function that takes an integer K (e.g., 342, -123) and returns its reverse digit (i.e., 243, -321)
        K = input('Part 3: Enter an integer: ')
        K = int(K)
        print(reverse_int(K))

        # Part 4: Function that takes a string s, representing a sentence, and returns a copy of the string with all commas removed
        s = input('Part 4: Write a sentence: ')
        print(s.replace(',',''))

        # Part 5: Function that takes a string sb, containing just the characters '(', ')', '{', '}', '[', and ']', and determine if the string is valid
        # s is considered valid if:
        #         1. open brackets must be closed by the same type of brackets
        #         2. open brackets must be closed in the correct order
        
        sb = input('Part 5: Write a string containing only brackets: ')
        if is_valid_brackets(sb) == True:
                print('String is valid.')
        elif is_valid_brackets(sb) == False:
                print('String is not valid.')

        # Part 6: Function merges two sorted lists and returns a new sorted list. Both input lists and output lists must be sorted
        list_1 = input('Enter a list of integers with a space between each integer: ')
        list_2 = input('Enter a list second of integers with a space between each integer: ')
        sort_lists(list_1, list_2)

        # Part 7: Function that visualizes given mathematical functions
        plot_functions()

# This function takes integer N and return the factorial of N
def factorial(N):
        fac_N = math.factorial(N)
        return (f'The factorial of {N} is {fac_N}.')

# This function takes a sequence of integers and returns the pairs of integers that, when multiplied together, produce an odd number
def odd_pair(seq):
        seq = seq.split(' ')
        seq_int = []

        # For loop converts seq list into a list of integers (seq_int)
        for item in seq:
                try:
                        item = int(item)
                        seq_int.append(item)
                except:
                        print('Incorrect format, no leading or trailing spaces.')

        # Output_list will print the pairs whose product is odd
        output_list = []
        for int_1 in seq_int:
                for int_2 in seq_int:
                        prod = int_1 * int_2
                        if prod % 2 != 0:

                                # Keep consistent formating to remove duplicates (ie. 3 & 5 have an odd product vs 5 & 3 have an odd product)
                                if int_1 >= int_2:
                                        output_list.append(f'{int_2} and {int_1} have an odd product.')
                                else:
                                        output_list.append(f'{int_1} and {int_2} have an odd product.')
     
        # Using set() to remove duplicates from list

        output_list = list(set(output_list))

        # Create and return output string
        out_string = ''
        for string in output_list:
                out_string = out_string + '\n' + string
        return out_string

# This function takes an integer and return the reversed integer
def reverse_int(K):
        # Neg indicates if the input is positive (0), negative (1), or zero (2)
        if K > 0:
                neg = 0
        elif K < 0:
                neg = 1
        else:
                neg = 2

        # Converts absolute value of interger to string and then reverses the string
        reversed_K = str(abs(K))[::-1]
        reversed_K = int(reversed_K)

        if neg == 0:
                return(reversed_K)
        elif neg == 1:
                return(-reversed_K)
        elif neg == 2:
                return(0)

# This function takes string s and determines if it is valid based on its proper use of brackets
def is_valid_brackets(s):

        # Initalize stack data structure to keep track of open brackets
        stack = []

        # Create bracket dictionary to match closing brackets to corresponding opening brackets
        bracket_dict = {')': '(', '}': '{', ']': '['}

        # Loop through each character in string
        for char in s:
                # Check if character is a closing bracket
                if char in bracket_dict:

                        # Check if there is a matching opening bracket on the top of the stack
                        # Dummy variable # indicates there is no matching opening bracket
                        top = stack.pop() if stack else '#'
                        
                        # If popped element doesn't match the corresponding opening bracket
                        if bracket_dict[char] != top:
                                return(False)
                else:
                        stack.append(char)                                                                                                                                             
        return not stack

# This function takes two lists, sorts them, and then merged them into one sorted list
def sort_lists(list_1_string, list_2_string):
        list_1_string = list_1_string.split(' ')
        list_1 = []

        list_2_string = list_2_string.split(' ')
        list_2 = []

        # For loop converts list_1_string into a list of integers
        for item in list_1_string:
                try:
                        item = int(item)
                        list_1.append(item)
                except:
                        print('Incorrect format, no leading or trailing spaces.')

        # For loop converts list_2_string into a list of integers
        for item in list_2_string:
                try:
                        item = int(item)
                        list_2.append(item)
                except:
                        print('Incorrect format, no leading or trailing spaces.')

        # Sort the input lists
        list_1.sort()
        print(f'Sorted list 1: {list_1}')
        list_2.sort()
        print(f'Sorted list 2: {list_2}')

        combined_list = list_1 + list_2
        combined_list.sort()

        print(f'Combined sorted list is: {combined_list}')

# This function plots a given linear function, quadratic function, log functions, and sigmoid function
def plot_functions():
        # Create a set of x values (400 evenly spaced numbers between -40 and 40)
        x = np.linspace(-40, 40, 400)

        # Create a set of x values (400 evenly spaced numbers between 0.01 and 0.99)
        # Used to avoid undefined values in logarithmic functions since log function is not defined for x = 0 or x = 1
        x_log = np.linspace(0.01, 0.99, 400)

        # Straight line
        y_straight = 30 + 0.5*x

        # Quadratic function
        y_quadratic = x**2 - 50*x + 645

        # Logarithmic functions for x between 0 and 1
        y_log_1 = -np.log(x_log)
        y_log_2 = -np.log(1 - x_log)

        # Sigmoid function - often used for machine learning classification tasks
        y_sigmoid = 1 / (1 + np.exp(-x))

        # Create 2x2 grid of sub plots for each function
        fig, axs = plt.subplots(2, 2, figsize = (12, 10))

        # Straight line plot
        # axs[0, 0].plot(x, y_straight) plots the straight line on the first subplot (top-left)
        axs[0, 0].plot(x, y_straight, label='y = 30 + 0.5x')
        axs[0, 0].set_title('Straight Line')
        axs[0, 0].set_xlabel('x')
        axs[0, 0].set_ylabel('y')

        # Adds legend to identify the line
        axs[0, 0].legend()
        axs[0, 0].grid(True) # Turn on grid

        # Quadratic function plot
        # Plots onto second sub-plot (top-right)
        # Set color to red to differentiate
        axs[0, 1].plot(x, y_quadratic, label='y = x^2 - 50x + 645', color='r')
        axs[0, 1].set_title('Quadratic Function')
        axs[0, 1].set_xlabel('x')
        axs[0, 1].set_ylabel('y')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Logarithmic functions
        # Plots onto second sub-plot (bottom-left)
        # Set color to green and blue to differentiate
        axs[1, 0].plot(x_log, y_log_1, label='y = -log(x)', color='g')
        axs[1, 0].plot(x_log, y_log_2, label='y = -log(1-x)', color='b')
        axs[1, 0].set_title('Logarithmic Functions')
        axs[1, 0].set_xlabel('x')
        axs[1, 0].set_ylabel('y')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Sigmoid function
        # Plots onto second sub-plot (bottom-right)
        # Set color to magenta to differentiate
        axs[1, 1].plot(x, y_sigmoid, label='y = 1 / (1 + e^(-x))', color='m')
        axs[1, 1].set_title('Sigmoid Function')
        axs[1, 1].set_xlabel('x')
        axs[1, 1].set_ylabel('y')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        # Save the plots
        plt.savefig('cs653_part_7.png')

if __name__ == "__main__":
        main()