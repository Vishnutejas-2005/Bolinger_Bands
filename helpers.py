import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch.bootstrap import SPA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox



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
    
def buy_and_sell_profits_basic(closing_prices:np.ndarray, upper_band:np.ndarray, lower_band:np.ndarray, mid_band:np.ndarray, stoploss:int = -1 ):
    '''
        Takes a numpy array of closing prices and gives the profits made by buying and selling \\
            **********
            #### INPUT
            # closing_prices - a numpy array of closing prices \\
            # upper_band - a numpy array of upper bands \\
            # lower_band - a numpy array of lower bands \\
            # mid_band - a numpy array of mid bands \\
            # stoploss - a int that sets the stoploss in % (if -ve then no stoploss)
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
        if flag == 0: # No Position
            if closing_prices[i] > upper_band[i]:
                flag = -1 # Shorting
                current_price = closing_prices[i]
            elif closing_prices[i] < lower_band[i]:
                flag = 1 # Longing
                current_price = closing_prices[i]

        elif flag == -1: # Closing for a Shorted Position
            Current_Profit = current_price - closing_prices[i]
            if stoploss > 0 :
                if closing_prices[i] < mid_band[i] or ((Current_Profit/closing_prices[i] * 100) < -1 * stoploss) :
                    flag = 0
                    profit += Current_Profit
            else :
                if closing_prices[i] < mid_band[i]:
                    flag = 0
                    profit += Current_Profit
        
        elif flag == 1: # Closing for a Longed Position
            Current_Profit = closing_prices[i] - current_price
            if stoploss > 0 :
                if closing_prices[i] > mid_band[i] or ((Current_Profit/closing_prices[i] * 100) < -1 * stoploss) :
                    flag = 0
                    profit += Current_Profit
            else :
                if closing_prices[i] > mid_band[i]:
                    flag = 0
                    profit += Current_Profit


        list_of_profit.append(profit)

    # Closing the trade at the end of data
    if flag == -1:
        profit += current_price - closing_prices[-1]
    elif flag == 1:
        profit += closing_prices[-1] - current_price
    
    list_of_profit.append(profit)
    return list_of_profit

##==============================================================================##

def Bollinger_Open_strat3 (mean:int, prev_mean:int, BB_up:int, BB_low:int, Hit:int = 0) -> int :
    '''
        Takes mean price of the stock at a particular minute and says if the stock has crossed the BB Bands or not \\
        If it has, it tells us if we should short, long or hold the stock 
        
        ---
        #### INPUT
        mean - mean/current value of the stock we want to trade \\
        prev_mean - mean of the stock in the previous minute \\
        BB_up - Upper Bollinger Band \\
        BB_low - Lower Bollinger Band \\
        Hit - 0 => Has not crossed the Bands, -1 => Crossed Upper Band before, 1 => Crossed lower band before
        ---
        #### OUTPUT
        If Hit = 0 \\
            Integer Value: 1 => crossed lower band; 0 => not crossed; -1 => crossed upper band \\
        If Hit = -1 or 1 \\
            Integer Value: 1 => Long; -1 => short; 0 => Do nothing
    '''       
    if Hit == 0:
        if mean < BB_low:
            signal = 1  # Hit Lower BB
        elif mean > BB_up:
            signal = -1  # Hit Upper BB
        else:
            signal = 0  # Did not Hit
    elif Hit == 1:
        if prev_mean < mean:
            signal = 1
        else: 
            signal = 0
    else:
        if prev_mean > mean:
            signal = -1
        else:
            signal = 0    
    
    return signal


def BollingerBaseline_Close (mean:int, open_price:int, BB_mid:int, Mode:str, stoploss:int = 100) -> list[int] :
    '''
        Takes current mean price and trade opening price and tells us if we should close the trade or not at the current stock price

        ---
        #### INPUT
        mean - mean/current value of the stock we want to trade \\
        open_price - the price at which we opened the trade \\
        BB_mid - Middle Bollinger Band (moving average) \\
        Mode - 'long' => we bought the stock before; 'short' => we sold the stock before \\
        lev - leverage \\
        stoploss - in percentage 

        ---
        #### OUTPUT
        List of 2 Integers: 
            List[0]: 1 => close/complete the trade; 0 => hold/do nothing
            List[1]: Net profit % 
    '''
 
    if Mode == 'long': # we bought stocks
        profit_per = (mean - open_price)/open_price * 100 # calculating PnL
        profit = (mean - open_price)
        if mean >= BB_mid: # closing when current value goes above middle band
            signal = 1 # signaling to close
            return_list = [signal, profit]
            return return_list
        else:
            if profit_per <= -stoploss : # closing at stoploss if stock falls too much
                signal = 1 # signaling to close
                return_list = [signal, profit]
                return return_list
            else:
                return [0, 0] # do nothing
            
    
    elif Mode == 'short': # we sold stocks
        profit_per = (open_price - mean)/open_price * 100 # calculating PnL
        profit = (mean - open_price)
        if mean <= BB_mid: # closing when current value goes below middle band
            signal = 1 # signaling to close
            return_list = [signal, profit]
            return return_list
        else:
            if profit_per <= -stoploss : # closing at stoploss if stock increases too much
                signal = 1 # signaling to close
                return_list = [signal, profit]
                return return_list
            else:
                return [0, 0] # do nothing
            
    else:
        print("Mode should have inputs as 'short' or 'long'; but someother input was given")
        return [0, 0]


def Bollinger_Running_strat3(closing_prices:np.ndarray, upper_band:np.ndarray, lower_band:np.ndarray, mid_band:np.ndarray, stoploss:int = -1,  output:bool = 'True') -> list[float]:
    '''
        Takes a string containing the path to a csv file that contains the data in minute-by-minute order. \\
        The data in the csv file should contain "open, high, low, close, Mean" in the respective order. \\
        It runs the BollingerBaseline algorithm on this data and returns the PnL. \\
        Assumptions - we can have only one entry before we close. And only one active position at a time. 
        
        ---
        #### INPUT
        path - a string containing the path to the csv file of stock data \\
        period - moving average for "period" minutes \\
        k - SD Multiplier (Bandwidth/2 or the distance between upper/lower band and middle band) \\
        lev - leverage \\
        stoploss - in percentage \\
        Entry_size - percentage of portfolio traded each time \\

        ---
        #### OUTPUT
        List - Per minute PnL
    '''

    # Initialize variables
    position = None  # 'long' or 'short'
    Hit = 0
    open_price = 0
    total_profit = 0
    Per_min_returns = []
    signal = 0

    # Iterate through the data
    for i in range(1, len(closing_prices)):
        Current_CP = closing_prices[i]
        Prev_CP = closing_prices[i-1]
        BB_up = upper_band[i]
        BB_low = lower_band[i]
        BB_mid = mid_band[i]

        if position is None:  # No active position
            if Hit == 0:
                Hit = Bollinger_Open_strat3(Current_CP, Prev_CP, BB_up, BB_low, Hit=0)  
            if Hit == 1:
                signal = Bollinger_Open_strat3(Current_CP, Prev_CP, BB_up, BB_low, Hit=1)
            elif Hit == -1:
                signal = Bollinger_Open_strat3(Current_CP, Prev_CP, BB_up, BB_low, Hit=-1)

            if signal == 1:  # Long entry
                if not output: print(f"Open Signal: {signal}")  # Debugging: Print entry signal
                position = 'long'
                open_price = Current_CP
            elif signal == -1:  # Short entry
                if not output:print(f"Open Signal: {signal}")  # Debugging: Print entry signal
                position = 'short'
                open_price = Current_CP

        elif position == 'long':  # Active long position
            close_signal, profit = BollingerBaseline_Close(Current_CP, open_price, BB_mid, 'long')                
            if close_signal == 1:  # Close position
                if not output: print(f"Close Signal: {close_signal}, Profit: {profit}")  # Debugging: Print close signal and profit
                total_profit += profit
                # Per_min_returns.append(profit) # appending per minute returns (in percentage)
                position = None  # Reset position
                signal = 0 # Reset signal

        elif position == 'short':  # Active short position
            close_signal, profit = BollingerBaseline_Close(Current_CP, open_price, BB_mid, 'short')            
            if close_signal == 1:  # Close position
                if not output: print(f"Close Signal: {close_signal}, Profit: {profit}")  # Debugging: Print close signal and profit
                total_profit += profit
                # Per_min_returns.append(profit) # appending per minute returns (in percentage)
                position = None  # Reset position
                signal = 0 # Reset signal
        
        Per_min_returns.append(total_profit)     
    
    return Per_min_returns  # Return net profit as a percentage of the initial portfolio and per minute returns


##==============================================================================##

def plot_profit(list_of_profit:np.ndarray, type_of_data:str = "timeseries", period:int = 20, k:int = 2):
    '''
        Takes a numpy array of profits and plots the profits \\
            **********
            #### INPUT
            # list_of_profit - a numpy array of profits \\
            # type_of_data - a string, the type of data to be plotted
            # period - a int, the period for which the moving 
                    average and std deviation is to be calculated
            # k - a int, the number of standard deviations to be used
            # **********
            # #### OUTPUT
            # None
    '''
    if type_of_data == "timeseries":
        plt.figure(figsize=(10, 5))
        plt.plot(list_of_profit)
        plt.title(f"Profits made by buying and selling with period {period} and k {k}")
        plt.xlabel("Time")
        plt.ylabel("Profit")
        plt.show()
    elif type_of_data == "profit":
        plt.plot(list_of_profit)
        plt.title(f"Profits made by buying and selling with period {period} and k {k}")
        plt.xlabel("Time")
        plt.ylabel("Profit")
        plt.show()
    return None

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

def get_spa(list_of_returns:np.ndarray, benchmark_index: int = 0, alpha:float = 0.05):
    '''
        Takes a numpy array of profits and gives the SPA test results \\
            **********
            #### INPUT
            # list_of_profit - a numpy array of profits \\
            # benchmark_index - a int, the index of the benchmark to be used
            # alpha - a float, the significance level to be used
            **********
            # #### OUTPUT
            # tuple - a tuple of the SPA test results
    '''
    benchmark_returns = list_of_returns[:, benchmark_index]
    # everything else
    stratergy_returns = np.delete(list_of_returns, benchmark_index, axis=1)
    benchmark_losses = -benchmark_returns
    stratergy_losses = -stratergy_returns
    spa = SPA(benchmark=benchmark_losses,
        models=stratergy_losses,
        reps=1000,
        block_size=100,  
        bootstrap="stationary",
        studentize=True,
        nested=False,
        seed=42
    )
    spa.compute()
    return spa

def interpret_spa(spa, alpha:float = 0.05, benchmark_index: int = 0):
    '''
        Takes the computed spa and gives the interpretation of the results \\
            **********
            #### INPUT
            # spa - the computed spa
            # alpha - a float, the significance level to be used
            # benchmark_index - a int, the index of the benchmark to be used
            **********
            # #### OUTPUT
            # indices_of_significant_models - a list of the indices of the models that are significantly different from the benchmark
    '''
    result = spa.better_models(pvalue=0.05, pvalue_type="consistent")
    # add 1 to the result whose index is more than i and do nothing for the rest
    result = [x + 1 if x > benchmark_index else x for x in result]
    return result

def ad_fuller_test(data:np.ndarray):
    '''
        Takes a numpy array of timeseries and gives the adfuller test results \\ 
         \\ preferebly returns percentage \\
            **********
            #### INPUT
            # data - a numpy array of data
            **********
            # #### OUTPUT
            # tuple - a tuple of the adfuller test results
    '''
    result = adfuller(data)
    return result

def ljung_box_test(data:pd.Series, lags:list = [10]):
    '''
        Takes a dataframe of data and gives the ljung box test results \\
            **********
            #### INPUT
            # data - a dataframe of data
            # lags - a int, the number of lags to be used
            **********
            # #### OUTPUT
            # dataframe - a dataframe of the ljung box test results
    '''
    result = acorr_ljungbox(data, lags=lags)
    return result

def plot_acf_pacf(data:pd.Series, lags:int = 10, change_the_scale:bool = False):
    '''
        Takes a numpy array of data and gives the acf and pacf plots \\
            **********
            #### INPUT
            # data - a numpy array of data
            # lags - a int, the number of lags to be used
            # change_the_scale - a boolean, if True, the scale of the plot is changed
            **********
            # #### OUTPUT
            # None
    '''
    plt.figure(figsize=(12, 6))
    plot_acf(data, lags=lags)
    if change_the_scale:
        plt.ylim(-0.1, 0.1)
    plt.title('ACF Plot')
    plt.show()
    plt.figure(figsize=(12, 6))
    plot_pacf(data, lags=lags)
    if change_the_scale:
        plt.ylim(-0.1, 0.1)
    plt.title('PACF Plot')
    plt.show()
    return None

