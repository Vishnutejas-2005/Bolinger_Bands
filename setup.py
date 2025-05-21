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



# list_of_profits = []
# for k in list_of_k:
#     for period in list_of_periods:
#         list_of_profits.append(generate_list_of_profit(closing_prices, period, k))
#         print(f"Profit for period {period} and k {k} is completed")

# np.save('profits.npy', list_of_profits)
# print("list_of_profits saved to profits.npy")
        


list_of_profits = np.load('profits.npy', allow_pickle=True)
print("list_of_profits loaded from profits.npy")
print(list_of_profits.shape)


list_of_returns = []
for i in range(len(list_of_profits)):
    list_of_returns.append(np.diff(list_of_profits[i]))

# reshape the list_of_returns to its transpose
list_of_returns = np.array(list_of_returns).T

for i in range(list_of_returns.shape[1]):
    spa = get_spa(list_of_returns=list_of_returns, benchmark_index=i, alpha=0.05)
    significant_better_models = interpret_spa(spa, alpha=0.05, benchmark_index=i)
    print(f"\nBetter models than benchmark (strategy {i}):")
    print(significant_better_models)
    