#!/usr/bin/python
# from __future__ import division, print_function

from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.table import SimpleTable
from sklearn.metrics import r2_score
import ml_metrics as metrics