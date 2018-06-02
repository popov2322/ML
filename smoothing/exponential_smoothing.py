#! /usr/bin/python
# -*- coding: UTF-8 -*-

from pandas import read_csv
import matplotlib.pyplot as plt

def exponential_smoothin(series, alpha):
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha*series[n] + (1 - alpha)*result[n-1])
    return result

if __name__ == '__main__':

    dataset = read_csv('/home/pda/PycharmProjects/Alesja_ML/exportFile-000002-2018-01-23-15-17-41.csv', ';', index_col=['Дата'], parse_dates=['Дата'], dayfirst=True, encoding='PT154')
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