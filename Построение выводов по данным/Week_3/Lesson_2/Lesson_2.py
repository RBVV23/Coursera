import pandas as pd
import numpy as np

from scipy.stats import pearsonr
from statsmodels.sandbox.stats.multicomp import multipletests
import scipy

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 25)
pd.set_option('display.max_columns', 10)


sales = pd.read_csv('foodmart.sales.tsv', sep='\t', header=0, parse_dates=[2])
print(sales.head())

products = pd.read_csv('foodmart.products.tsv', sep='\t', header=0)
print(products.head())

sales = sales.merge(products[['product_id', 'product_name']], on=['product_id'], how='inner')
print(sales.head())

sparse_sales = pd.pivot_table(sales, values='sales', index=['date', 'store_id'],
                              columns=['product_name'], fill_value=0)
print(sparse_sales.head())

corr_data = []

for i, lhs_column in enumerate(sparse_sales.columns):
    for j, rhs_column in enumerate(sparse_sales.columns):
        if i >= j:
            continue

        corr, p = pearsonr(sparse_sales[lhs_column], sparse_sales[rhs_column])
        corr_data.append([lhs_column, rhs_column, corr, p])

sales_correlation = pd.DataFrame.from_records(corr_data)
sales_correlation.columns = ['product_A', 'product_B', 'corr', 'p_value']

print(sales_correlation.head())

print('Количество отвергнутых/принятых гипотез без поправки на множественную проверку:')
print((sales_correlation.p_value < 0.05).value_counts())

reject, p_corrected, a1, a2 = multipletests(sales_correlation.p_value, alpha=0.05, method='holm')
sales_correlation['p_corrected'] = p_corrected
sales_correlation['reject'] = reject

print(sales_correlation.head())

print('Количество отвергнутых/принятых гипотез с поправки на множественную проверку методом Холма:')
print(sales_correlation.reject.value_counts())

print(sales_correlation[sales_correlation.reject == True].sort_values(by='corr', ascending=False))

reject, p_corrected, a1, a2 = multipletests(sales_correlation.p_value, alpha=0.05, method='fdr_bh')
sales_correlation['p_corrected'] = p_corrected
sales_correlation['reject'] = reject

print(sales_correlation.head())

print('Количество отвергнутых/принятых гипотез с поправки на множественную проверку методом Бенджамини-Хохберга:')
print(sales_correlation.reject.value_counts())

print(sales_correlation[sales_correlation.reject == True].sort_values(by='corr', ascending=False))

