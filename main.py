import pandas as pd

encoding = 'utf-8-sig'

pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)

DataFrame = pd.read_csv('./queries.txt', header=None)

print(DataFrame)