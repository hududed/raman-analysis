import lmfit
from pathlib import Path
import matplotlib.pyplot as plt
import rampy as rp
import pandas as pd


def load_data(parent, child):
    dfs = []
    for j, k in zip(parent, child):
        dfs.append(pd.read_csv(j / k, names=['Energy',
                                             k], header=None, delimiter='\t'))
    return pd.concat(dfs, axis=1).T.drop_duplicates().T  # care: works only when all E column values are the same


p = Path('../data/UV_-VIS/')

parent=[x.parents[0] for x in p.rglob("*.[cC][sS][vV]")]
child=[x.name for x in p.glob("*.[cC][sS][vV]")]

df = pd.read_csv(parent[0]/child[0], header=[0], delimiter = ',', skiprows=[1], skipfooter=1198, engine='python')

# need to shift headers 1 to the right, care: wavelength columns NOT identical
cols = df.columns[:-2].insert(loc=0, item='Wavelength')
df = df.drop('Unnamed: 68', 1)
df.columns = cols
df = df.drop(df .columns[2::2], 1)

df.plot(x='Wavelength', y=['graphite','biochar hummers', 'coal hummers', 'coal char hummers'])


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
