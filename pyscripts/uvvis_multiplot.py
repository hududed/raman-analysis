from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from lmfit.lineshapes import *
from lmfit import Parameters, fit_report, minimize


def load_data(parent, child):
    # loads data only if x columns are identical!
    dfs = []
    for j, k in zip(parent, child):
        dfs.append(pd.read_csv(j / k, names=['Energy',
                                             k], header=None, delimiter='\t'))
    return pd.concat(dfs, axis=1).T.drop_duplicates().T  # care: works only when all E column values are the same


def normalise(x, y):
    # min-max normalisation
    return (y - y.min()) / (y.max() - y.min())


def line(x, slope, intercept):
    """a line"""
    return slope * x + intercept


def residual(pars, x, data=None, eps=None):  # Function definition
    # unpack parameters, extract .value attribute for each parameter
    a1 = pars['a1'].value
    f1 = pars['f1'].value
    l1 = pars['l1'].value
    m = pars['m'].value
    c = pars['c'].value

    # Using the Gaussian model function from rampy
    peak1 = gaussian(x, a1, f1, l1)
    #     peak3 = lorentzian(x,a3,f3,l3)
    line1 = line(x, m, c)

    model = peak1 + line1  # The global model is the sum of the Gaussian peaks

    if data is None:  # if we don't have data, the function only returns the direct calculation
        return model, peak1, line1
    if eps is None:  # without errors, no ponderation
        return (model - data)
    return (model - data) / eps  # with errors, the difference is ponderated


p = Path('../data/UV_-VIS/')

# look for case-insensitive csv, in subfolders as well
parent = [x.parents[0] for x in p.rglob("*.[cC][sS][vV]")]
child = [x.name for x in p.glob("*.[cC][sS][vV]")]

df = pd.read_csv(parent[0] / child[0], header=[0], delimiter=',', skiprows=[1], skipfooter=1198, engine='python')

# need to shift headers 1 to the right, care: wavelength columns NOT identical
cols = df.columns[:-2].insert(loc=0, item='Wavelength')
df = df.drop('Unnamed: 68', 1)
df.columns = cols
df = df.drop(df.columns[2::2], 1)

df.plot(x='Wavelength', y=['coal oxidizer2', 'biochar hummers', 'coal hummers', 'coal char hummers'])

# normalising min-max
results = [normalise(df['Wavelength'], val) for cols, val in df.iteritems()]

df2 = pd.DataFrame(results[1:]).T
df2.columns = [str(col) + '_norm' for col in df.columns[1:]]

# add normalised columns to dataframe df
dfs = pd.concat([df, df2], axis=1)

# normalised plots
dfs.plot(x='Wavelength', y=['coal oxidizer2_norm', 'graphite2_norm', 'coal hummers_norm', 'biochar oxidizer2_norm'])

# fitting
pars = Parameters()

x = dfs['Wavelength'].values
y = dfs['coal oxidizer2_norm'].values

#            (Name,  Value,  Vary,   Min,  Max,  Expr)
pars.add_many(('a1', 10, True, 0, None, None),
              ('f1', 200, False, 190, 205, None),
              ('l1', 20, True, 0, 30, None),
              ('m', 0, True, 0, None, None),
              ('c', 0, True, 0, None, None))

result = minimize(residual, pars, method='leastsq', args=(x, y))

pars['f1'].vary = True
result2 = minimize(residual, pars, method='leastsq', args=(x, y))

yout, peak1, line1 = residual(result2.params, x)

print(fit_report(result2))

f3 = plt.figure(3, figsize=(20, 5))

plt.plot(x, y, 'bo')
plt.plot(x, yout, 'r-', label='coal oxidizer2', linewidth=3)
plt.plot(x, peak1, 'b-', label='gaussian')
plt.plot(x, line1, 'k-', label='baseline')
plt.legend()
plt.show()


#
# df = load_data(parent, child)
#
# # f1 = plt.figure(1, figsize=(20, 5))
#
# fig, axes = plt.subplots(nrows=4, ncols=1,
#                          sharex=True, sharey=True, figsize=(8, 8))
#
# df.plot(x='Energy', y='biochar oxidizer 2.txt', ax=axes[0])
# df.plot(x='Energy', y='biochar hummers.txt', ax=axes[1])
# df.plot(x='Energy', y='CoalChar_HNO3_33.8laser_30inttime.txt', ax=axes[2])
# df.plot(x='Energy', y='CoalChar_Hummers_33.8laser_20inttime.txt', ax=axes[3])
# fig.tight_layout()
#
# # Bring subplots close to each other and positioned further to the right
# fig.subplots_adjust(hspace=0, left=0.1, bottom=0.08)
#
# # Hide x labels and tick labels for all but bottom plot.
# for ax in axes:
#     ax.xaxis.label.set_visible(False)
#     ax.tick_params(axis='both', which='both', direction='in',top=True)
#                     #     ax.label_outer()
#
# fig.suptitle('Raman Rawdata GO', y=1.02, fontsize=20)
# fig.text(0.5, 0.01, r'Wavenumber [$cm^{-1}$]', fontsize=20, ha='center')
# fig.text(0, 0.5, 'Intensity a.u.', fontsize=20, va='center', rotation='vertical')

plt.show()

input()
