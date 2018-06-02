#! /usr/bin/python
# -*- coding: UTF-8 -*-

from pandas import read_csv
import matplotlib.pyplot as plt

def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # прогнозируем
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result


if __name__ == '__main__':

    dataset = read_csv('/home/pda/PycharmProjects/Alesja_ML/exportFile-000002-2018-01-23-15-17-41.csv', ';',
                       index_col=['Дата'], parse_dates=['Дата'], dayfirst=True, encoding='PT154')
    money_per_day = dataset['сумма выданых наличных за сутки']
    money_per_day = money_per_day[1000:]

    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(20, 8))
        for alpha in [0.9, 0.02]:
            for beta in [0.9, 0.02]:
                plt.plot(double_exponential_smoothing(money_per_day, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
        plt.plot(money_per_day.values, label = "Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Double Exponential Smoothing")
        plt.grid(True)
        plt.show()