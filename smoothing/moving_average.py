#! /usr/bin/python
# -*- coding: UTF-8 -*-

from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt



def plot_moving_average(series, n):
    '''
    series - dataframe with timeseries
    n - rolling window size
    '''

    rolling_mean = series.rolling(window=n).mean()

    #При желании, можно строить доверительные интревалы для сглаженных значений
    #rolling_std = series.rolling(window=n).std()
    #upper_bond = rolling_mean + 1.96*rolling_std
    #upper_bond = rolling_mean - 1.96*rolling_std

    plt.figure(figsize=(15, 5))
    plt.title("Moving average/n window size = {}".format(n))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    #plt.plot(upper_bond, "r--", label="Upper Bond"
    #plt.plot(lower_bond, "r--", label="Lower Bond"
    # plt.plot(dataset[n:], label="Actual values")
    plt.plot(series[n:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()

def exponential_smoothin(series, alpha):
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha*series[n] + (1 - alpha)*result[n-1])
    return result

if __name__ == '__main__':

    dataset = read_csv('exportFile-000002-2018-01-23-15-17-41.csv', ';', index_col=['Дата'], parse_dates=['Дата'], dayfirst=True, encoding='PT154')
    money_per_day = dataset['сумма выданых наличных за сутки']
    money_per_day = money_per_day[2000:]

    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(20,8))
        for alpha in [0.3, 0.05]:
            plt.plot(exponential_smoothin(money_per_day, alpha), label="Alpha {}".format(alpha))
        plt.plot(money_per_day.values, 'c', label = 'Actual')
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Exponential Smoothing")
        plt.grid(True)
        plt.show()