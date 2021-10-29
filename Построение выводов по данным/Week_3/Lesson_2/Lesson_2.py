import pandas as pd
import numpy as np

from scipy.stats import pearsonr
from statsmodels.sandbox.stats.multicomp import multipletests
import scipy

sales = pd.read_csv('foodmart.sales.tsv', sep='\t', header=0, parse_dates=[2])
print(sales.head())

products = pd.read_csv('foodmart.products.tsv', sep='\t', header=0)
print(products.head())

sales = sales.merge(products[['product_id', 'product_name']], on=['product_id'], how='inner')
print(sales.head())