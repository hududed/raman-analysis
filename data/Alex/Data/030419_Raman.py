# 2019-03-04
# Hud Wahab
# ---------Raman Analysis ---------#

import pandas as pd  # python data manipulation and analysis library
import numpy as np  # Library with large collection of high-level mathematical functions to operate on arrays
import matplotlib.pyplot as plt  # python plotting library

from scipy.optimize import curve_fit
import csv

# Lorentzian functions to which baseline subtracted data is fitted
# Learn more: https://lmfit.github.io/lmfit-py/builtin_models.html
def lorentzian_fcn(x, I, x0, gamma):
    return I * ((gamma ** 2) / (((x - x0) ** 2) + gamma ** 2))

def two_lorentzian(x, I1, x1, gamma1, I2, x2, gamma2, y0):
    return lorentzian_fcn(x, I1, x1, gamma1) + lorentzian_fcn(x, I2, x2, gamma2) + y0

def to_wavenumber(x): return 1e7/532 - 1e7/x

def fit_report(results):
    print("The G/D ratio is %.2f\n...D-intensity is %.2f at %.2f cm-1\nG-intensity is %.2f at %.2f cm-1"
          % ((results[0]/results[3]), results[0], to_wavenumber(results[1]),
             results[3], to_wavenumber(results[4])))


# -------- Reading data from .csv files
path = './'
fname = '2019-02-17  GO20_s1_lp400mW_dur600ms_p70psi.csv'
bgrfname = '2019-02-18  GO20_bgr.csv'
datafn = path + fname
bgrfn = path + bgrfname
data = pd.read_csv(datafn, header=0, index_col=0, names=['W', 'I'])
bgr = pd.read_csv(bgrfn, header=0, index_col=0, names=['W', 'I'])

data_proc = (data.I.values - bgr.I.values)

data_index = data.index.values
data_proc = pd.DataFrame({'I': data_proc}, index=data_index)
data_proc = data_proc[569:585]

lowval, hival = data_proc[data_proc.index.min():data_proc.index.min() + 2].values.mean(), data_proc[
                                                                                          data_proc.index.max() - 2:data_proc.index.max()].values.mean()
low, hi = data_proc[data_proc.index.min():data_proc.index.min() + 2].index.values.mean(), data_proc[
                                                                                          data_proc.index.max() - 2:data_proc.index.max()].index.values.mean()

y = [lowval, hival]
x = [low, hi]
m, b = np.polyfit(x, y, 1)

data_index = data_proc.index.values
data_proc = data_proc.I.values - (data_proc.index.values * m + b)
data_proc = pd.DataFrame({'I': data_proc}, index=data_index)

prms = [2000, 573, 1, 6000, 581.5, 0.5, 5000]  # prms = [I1, x1, gamma1, I2, x2, gamma2, y0]

# Optimal values for the prms are returned in array form via popt after lorentzian curve_fit
popt, pcov = curve_fit(two_lorentzian, data_proc.index.values, data_proc.I.values, p0=prms)

# Fit data is computed by passing optimal prms and x-values to two_lorentzian function
data_proc['fit'] = two_lorentzian(data_proc.index, *popt)

plt.plot(data_proc)
plt.plot(data_proc.fit)

# print G/D, D and G intensities and centroids
fit_report(popt)

# input variables to be appended in csv file
power = input('Power: ')
time = input('time: ')
pressure = input('Pressure: ')

# redefining outputs
D = popt[0]
G = popt[3]
ratio = popt[0] / popt[3]
list1 = [power, time, pressure, D, G, ratio]  # to be appended

# append outputs
with open("output_.csv", "a") as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(list1)
