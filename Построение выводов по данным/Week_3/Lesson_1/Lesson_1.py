import pandas as pd
import numpy as np

sales = pd.read_csv('foodmart.sales.tsv', sep='\t', header=0, parse_dates=[2])
print(sales.head())

products = pd.read_csv('foodmart.products.tsv', sep='\t', header=0)
print(products.head())

sales = sales.merge(products[['product_id', 'product_name']], on=['product_id'], how='inner')
print(sales.head())

sparse_sales = pd.pivot_table(sales, values='sales', index=['date', 'store_id'],
                              columns=['product_name'], fill_value=0)
print(sparse_sales.head())

sales_correlation = sparse_sales.corr()