#!/usr/bin/python3.6

"""df_builder.py: Create a Pandas dataframe from hop data input."""

__author__ = "Reina Hastings"
__email__ = "reinahastings13@gmail.com"

import pandas as pd
import argparse
import gspreadj
def main():
        parser = argparse.ArgumentParser(description='Creates Pandas d')
        parser.add_argument('-i', '--input', help='Input CSV containing gene information', default='/home/tjames/summary_hybran_1.7.1/all_gene_positions.csv')
        args = parser.parse_args()

        df = pd.read_csv(args.input)
        print(df)


if __name__ == "__main__":
        main()
