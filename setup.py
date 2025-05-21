import pandas as pd
from helpers import *
import xlrd
import numpy as np
import matplotlib.pyplot as plt

list_of_periods = [5, 10, 15, 20, 25, 30]
list_of_k = [1, 1.5, 2]
# Dataset_XLS_To_CSV('cumul_ohlc.xls', 'Dataset', 10)

# get the closing prices data from Dataset.csv
data = pd.read_csv('Dataset.csv')
closing_prices = np.array(data['close'].values)

list_of_profits = []
for k in list_of_k:
    for period in list_of_periods:
        list_of_profits.append(generate_list_of_profit(closing_prices, period, k))
        print(f"Profit for period {period} and k {k} is completed")

np.save('profits.npy', list_of_profits)
print("list_of_profits saved to profits.npy")
        

