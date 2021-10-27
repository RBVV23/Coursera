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
print(sales_correlation.head())

product_name = 'American Chicken Hot Dogs'
print(sales_correlation[[product_name]].sort_values(product_name, ascending=True).head())

min_corr = pd.DataFrame(sales_correlation.min())
min_corr.columns = ['min']
print(min_corr.sort_values(by='min').head())

max_corr = pd.DataFrame(sales_correlation.apply(lambda x: np.max(list(filter(lambda x: x != 1., x))), axis=1))
max_corr.columns = ['max']
print(max_corr.sort_values(by='max', ascending=False).head())

product_name = 'Plato French Roast Coffee'
print(sales_correlation[[product_name]].sort_values(product_name, ascending=False).head())