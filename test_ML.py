#! /usr/bin/python
# -*- coding: UTF-8 -*-

from pandas import read_csv
from smoothing import moving_average
from smoothing import exponential_smoothing

dataset = read_csv('/home/pda/PycharmProjects/Alesja_ML/CashStatistics/exportFile-000011-2018-01-23-15-21-32.csv',';', index_col=['Дата'], parse_dates=['Дата'], dayfirst=True, encoding='PT154')
money_per_day = dataset['сумма выданых наличных за сутки']
print(money_per_day)

moving_average.plot_moving_average(money_per_day[500:], 30*12)



















# money_per_day = money_per_day.resample('W').mean()
# money_per_day.plot(figsize=(12,6))
# plt.show()
# itog = money_per_day.describe()
# money_per_day.hist()
# print(itog)
# print('V = {func}'.format(func=itog['std']/itog['mean']))
# row =  [u'JB', u'p-value', u'skew', u'kurtosis']
# jb_test = sm.stats.stattools.jarque_bera(money_per_day)
# a = np.vstack([jb_test])
# itog = SimpleTable(a, row)
# print(itog)
# test = sm.tsa.adfuller(money_per_day)
# print('adf: ', test[0])
# print('p-value: ', test[1])
# print('Critical values: ', test[4])
# if test[0]> test[4]['5%']:
#     print('есть единичные корни, ряд не стационарен')
# else:
#     print ('единичных корней нет, ряд стационарен')