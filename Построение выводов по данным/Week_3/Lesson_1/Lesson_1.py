import pandas as pd
import numpy as np

sales = pd.read_csv('foodmart.sales.tsv', sep='\t', header=0, parse_dates=[2])
print(sales.head())

products = pd.read_csv('foodmart.products.tsv', sep='\t', header=0)
print(products.head())