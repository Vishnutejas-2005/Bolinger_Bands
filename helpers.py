import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def Dataset_XLS_To_CSV (path:str, Filename:str, col:int)->bool:
    '''
        Takes the path of an xls file and creates a csv file with its name as Filename.csv \\
        With the csv file containing columns 1 to "col". 
        **********
        #### INPUT
        path - a string of the path to the xls file \\
        Filename - the filename of the resulting csv file, to be saved as \\
        col - a int, where only upto column number "col" is copied 
        **********        
        #### OUTPUT
        boolean value - True => success, False => could not do the operation
    '''

    try :
        # Trying to read the xls file
        data = pd.read_excel(path,engine='xlrd')

        # Saving upto column number "col"
        required_data = data.iloc[:,:col]

        # Saving the data to a csv file
        required_data.to_csv(f"{Filename}.csv", index=False)

        return True
    
    except FileNotFoundError:
        print(f"Error: File at path '{path}' does not exist.")
        return False
    except ValueError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"Exception: {e}")
        return False
    

def get_avg_std(prices:np.ndarray,period:int)->tuple:
    '''
        Takes a numpy array of prices and gives the moving average 
        and std deviation of the prices \\
        **********
        #### INPUT
        prices - a numpy array of prices \\
        period - a int, the period for which the moving 
                average and std deviation is to be calculated
        **********
        #### OUTPUT
        tuple - a tuple of the moving average and std deviation
        of the prices
                '''
    avg = np.zeros(len(prices))
    std = np.zeros(len(prices))
    for i in range(period, len(prices)+1):
        avg[i-1] = np.mean(prices[i-period:i])
        std[i-1] = np.std(prices[i-period:i])
    return avg, std
    
def construct_upper_lower_bands(prices:np.ndarray, moving_avg:np.ndarray, moving_std:np.ndarray, period:int)->tuple:
    '''
        Takes a numpy array of prices and gives the upper and lower bands \\
        **********
        #### INPUT
        prices - a numpy array of prices \\
        moving_avg - a numpy array of moving averages \\
        moving_std - a numpy array of std deviations \\
        period - a int, the period for which the moving 
                average and std deviation is to be calculated
        **********
        #### OUTPUT
        tuple - a tuple of the upper and lower bands
    '''
    upper_band = np.zeros(len(prices))
    lower_band = np.zeros(len(prices))
    for i in range(period, len(prices)+1):
        upper_band[i-1] = moving_avg[i-1] + 2*moving_std[i-1]
        lower_band[i-1] = moving_avg[i-1] - 2*moving_std[i-1]
    return upper_band, lower_band
    
def buy_and_sell_profits_basic(closing_prices:np.ndarray, upper_band:np.ndarray, lower_band:np.ndarray, mid_band:np.ndarray):
    '''
        Takes a numpy array of closing prices and gives the profits made by buying and selling \\
            **********
            #### INPUT
            # closing_prices - a numpy array of closing prices \\
            # upper_band - a numpy array of upper bands \\
            # lower_band - a numpy array of lower bands \\
            # mid_band - a numpy array of mid bands \\
            # **********
            # #### OUTPUT
            # tuple - a tuple of the profits made by buying and selling
    '''
    # flag = 0 => no position
    # flag = 1 => long position
    # flag = -1 => short position
    # current_price = the price at which the position was taken
    flag = 0
    current_price = 0
    list_of_profit = []
    profit = 0

    for i in range(len(closing_prices)):
        if flag == 0:
            if closing_prices[i] > upper_band[i]:
                flag = -1
                current_price = closing_prices[i]
            elif closing_prices[i] < lower_band[i]:
                flag = 1
                current_price = closing_prices[i]
        elif flag == -1:
            if closing_prices[i] < mid_band[i]:
                flag = 0
                profit += current_price - closing_prices[i]
        elif flag == 1:
            if closing_prices[i] > mid_band[i]:
                flag = 0
                profit += closing_prices[i] - current_price
        list_of_profit.append(profit)
    if flag == -1:
        profit += current_price - closing_prices[-1]
    elif flag == 1:
        profit += closing_prices[-1] - current_price
    list_of_profit.append(profit)
    return list_of_profit

def plot_profit(list_of_profit:np.ndarray, period:int = 20,k:int = 2):
    '''
        Takes a numpy array of profits and plots the profits \\
            **********
            #### INPUT
            # list_of_profit - a numpy array of profits \\
            # period - a int, the period for which the moving 
                    average and std deviation is to be calculated
            # k - a int, the number of standard deviations to be used
            # **********
            # #### OUTPUT
            # None
    '''
    plt.plot(list_of_profit)
    plt.title(f"Profits made by buying and selling with period {period} and k {k}")
    plt.xlabel("Time")
    plt.ylabel("Profit")
    plt.show()


def generate_list_of_profit(closing_prices:np.ndarray, period:int = 20, k:int = 2):
    '''
        Takes a numpy array of closing prices and gives the profits made by buying and selling \\
            **********
            #### INPUT
            # closing_prices - a numpy array of closing prices \\
            # period - a int, the period for which the moving 
                    average and std deviation is to be calculated
            # k - a int, the number of standard deviations to be used
            **********
            # #### OUTPUT
            # tuple - a tuple of the profits made by buying and selling
    '''
    moving_avg, moving_std = get_avg_std(closing_prices, period=period)
    upper_band, lower_band = construct_upper_lower_bands(closing_prices, moving_avg, moving_std, period=period)
    list_of_profit = buy_and_sell_profits_basic(closing_prices, upper_band, lower_band, moving_avg)
    return list_of_profit
