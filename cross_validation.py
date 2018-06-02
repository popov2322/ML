import numpy as np
from pandas import read_csv
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import minimize
from smoothing.Holt_Winters import HoltWinters
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def timeseriesCVscore(x):
    # вектор ошибок
    errors = []

    values = data.values
    alpha, beta, gamma = x

    # задаём число фолдов для кросс-валидации
    tscv = TimeSeriesSplit(n_splits=5)
    print('tscv.split'.format(tscv.split))

    # идем по фолдам, на каждом обучаем модель, строим прогноз на отложенной выборке и считаем ошибку
    for train, test in tscv.split(values):
        model = HoltWinters(series=values[train], slen=7, alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
        model.triple_exponential_smoothing()

        predictions = model.result[-len(test):]
        actual = values[test]
        error = mean_squared_error(predictions, actual)
        errors.append(error)
    return np.mean(np.array(errors))

def plotHoltWinters():
    Anomalies = np.array([np.NaN]*len(data))
    Anomalies[data.values<model.LowerBond] = data.values[data.values<model.LowerBond]
    plt.figure(figsize=(25, 10))
    plt.plot(model.result, label = "Model")
    plt.plot(model.UpperBond, "r--", alpha=0.5, label = "Up/Low confidence")
    plt.plot(model.LowerBond, "r--", alpha=0.5)
    plt.fill_between(x=range(0,len(model.result)), y1=model.UpperBond, y2=model.LowerBond, alpha=0.5, color = "grey")
    plt.plot(data.values, label = "Actual")
    plt.plot(Anomalies, "o", markersize=10, label = "Anomalies")
    plt.axvspan(len(data)-128, len(data), alpha=0.5, color='lightgrey')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc="best", fontsize=13)


if __name__ == '__main__':

    dataset = read_csv('/home/pda/PycharmProjects/Alesja_ML/CashStatistics/exportFile-000010-2018-01-23-15-21-05.csv', ';', index_col=['Дата'], parse_dates=['Дата'], dayfirst=True, encoding='PT154')
    data = dataset['сумма выданых наличных за сутки']
    data = data[1476::2]
    print(data.values)

    # инициализируем значения параметров
    x = [0,0,0]

    # Минимизируем функцию потерь с ограничениями на параметры
    opt = minimize(timeseriesCVscore, x0=x, method="TNC", bounds=((0, 1), (0, 1), (0, 1)))

    # Из оптимизатора берем оптимальное значение параметров
    alpha_final, beta_final, gamma_final = opt.x
    print(alpha_final, beta_final, gamma_final)

    print('начал')
    model = HoltWinters(data[:-300], slen=7, alpha=alpha_final, beta=beta_final, gamma=gamma_final, n_preds=300,
                        scaling_factor=3)
    model.triple_exponential_smoothing()

    plotHoltWinters()
    plt.show()