#!/miniconda3/envs/class/bin/python

"""download_data file for homework #2 for CS-549 Machine Learning"""

__author__ = "Reina Hastings"
__email__ = "reinahastings13@gmail.com"

from pandas import read_table

""" download_data() --- Download data from a CSV file and return a pandas DataFrame.

Reads data from a specified CSV file and returns the selected columns in a pandas DataFrame. The file is expected to be in 'latin-1' encoding, but this can be adjusted if needed. It assumes that the first row of the CSV file contains the column headers.

Parameters:
    file_location (str) - Path to the CSV file (e.g., 'sat.csv').
    fields (list) - List of column names or indices to select from the CSV file.

Returns:
    pandas.DataFrame - DataFrame containing the data from the specified columns.
"""
def download_data(file_location, fields):

    # Downloads the data for this script into a pandas DataFrame. Uses columns indices provided

    frame = read_table(
        file_location,
        
        # Specify the file encoding
        # Latin-1 is common for data from US sources
        encoding='latin-1',
        #encoding='utf-8',  # UTF-8 is also common

        # Specify the separator in the data
        sep=',',            # comma separated values

        # Ignore spaces after the separator
        skip_initial_space=True,

        # Generate row labels from each row number
        index_col=None,

        # Generate column headers row from each column number
        header=0,          # Use the first line as headers

        use_cols=fields
    )

    # Return the entire frame
    return frame