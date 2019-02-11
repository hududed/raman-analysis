# coding: utf-8

# # Starting comments
# Charles Le Losq, Geophysical Laboratory, Carnegie Institution for Science. 7 April 2015.
# 
# This IPython notebook is aimed to show how you can easily fit a Raman spectrum with Python tools, for free and,
# in my opinion, in an elegant way.
# 
# This fitting procedure is much less "black-box" than existing GUI softwares. It probably is a little bit harder to
# learn for the newcomer, but rewards are much greater since you can control all the procedure in every single detail.
# 
# In this example, we will fit the 850-1300 cm$^{-1}$ portion of a Raman spectrum of a lithium tetrasilicate glass
# Li$_2$Si$_4$O$_9$, the name will be abbreviated LS4 in the following.
# 
# For further references for fitting Raman spectra of glasses, please see for instance: Virgo et al., 1980,
# Science 208, p 1371-1373; Mysen et al., 1982, American Mineralogist 67, p 686-695; McMillan, 1984,
# American Mineralogist 69, p 622-644; Mysen, 1990, American Mineralogist 75, p 120-134; Le Losq et al., 2014,
# Geochimica et Cosmochimica Acta 126, p 495-517 and Neuville et al., 2014, Reviews in Mineralogy and Geochemistry 78.
# 
# We will use the optimization algorithms of Scipy together with the library lmfit (http://lmfit.github.io/lmfit-py/)
# that is extremely useful to add constrains to the fitting procedure.

# # Importing libraries
# So the first part will be to import a bunch of libraries for doing various things

import lmfit
from fastai.basics import *
from lmfit.lineshapes import *
import rampy as rp  # Charles' libraries and functions

# # Importing and looking at the data
# Let's first have a look at the spectrum

path = Path('../data/Raman Ana/1_16_19')
# path.ls()

# get the spectrum to deconvolute, with skipping header and footer comment lines from the spectrometer
inputsp = np.genfromtxt(path / "graphene oxide from graphenea company 2.txt")

x = inputsp[:, 0]
y = inputsp[:, 1]

# create a new plot for showing the spectrum
f1 = plt.figure(1, figsize=(20, 5))

plt.plot(x, y, 'k.', markersize=1)
plt.xticks(np.arange(min(x), max(x), 100))
plt.xlabel("Raman shift, cm$^{-1}$", fontsize=12)
plt.ylabel("Normalized intensity, a. u.", fontsize=12)
plt.title("Fig. 1: the raw data", fontsize=12, fontweight="bold")

# We are interested in fitting the 870-1300 cm$^{-1}$ portion of this spectrum, which can be assigned to the various
# symmetric and assymetric stretching vibrations of Si-O bonds in the SiO$_2$ tetrahedra present in the glass network
# (see the above cited litterature for details).

# # Baseline Removal
# 
# First thing we notice in Fig. 1, we have to remove a baseline because this spectrum is shifted from 0 by some
# "background" scattering. For that, we can use the rp.baseline() function

bir = np.array([(1000, 1100), (1800, 1900)])  # The regions where the baseline will be fitted
y_corr, y_base = rp.baseline(x, y, bir, 'poly', polynomial_order=2)  # We fit a polynomial background.

f2 = plt.figure(2, figsize=(10, 10))
plt.plot(x, y_corr)

# Now we will do some manipulation to have the interested portion of spectrum in a single variable. We will assume
# that the errors have not been drastically affected by the correction process (in some case it can be, but this one
# is quite straightforward), such that we will use the initial relative errors stored in the "ese0" variable.

# signal selection
lb = 1100  # The lower boundary of interest
hb = 1800  # The upper boundary of interest

x_fit = x[np.where((x > lb) & (x < hb))]
y_fit = y_corr[np.where((x > lb) & (x < hb))]

ese0 = np.sqrt(abs(y_fit[:, 0])) / abs(y_fit[:, 0])  # the relative errors after baseline subtraction
y_fit[:, 0] = y_fit[:, 0] / np.amax(y_fit[:, 0]) * 10  # normalise spectra to maximum intensity, easier to handle
sigma = abs(ese0 * y_fit[:, 0])  # calculate good ese

# # And let's plot the portion of interest before and after baseline subtraction:

# create a new plot for showing the spectrum
f3 = plt.figure(3)
plt.subplot(1, 2, 1)
inp = plt.plot(x, y, 'k-', label='Original')
corr = plt.plot(x, y_corr, 'b-',
                label='Corrected')  # we use the sample variable because it is not already normalized...
bas = plt.plot(x, y_base, 'r-', label='Baseline')
plt.xlim(lb, hb)
plt.ylim(0, 10000)
plt.xlabel("Raman shift, cm$^{-1}$", fontsize=14)
plt.ylabel("Normalized intensity, a. u.", fontsize=14)
plt.legend()
plt.title('A) Baseline removal')

plt.subplot(1, 2, 2)
plt.plot(x_fit, y_fit, 'k.')
plt.xlabel("Raman shift, cm$^{-1}$", fontsize=14)
plt.title('B) signal to fit')
# plt.tight_layout()
plt.suptitle('Figure 2', fontsize=14, fontweight='bold')


# # Last steps before fitting the spectrum
# 
# So here we are. We have the corrected spectrum in the sample variable. 
# 
# But before going further away,we need to write a function for the optimisation. It will return the difference
# between the calculated and measured spectrum, following the guideline provived by lmfit (
# http://lmfit.github.io/lmfit-py/) Please note that I do the fitting this way because it gives a pretty good control
# of the entire process, but you can use directly the builtin models of lmfit (
# http://lmfit.github.io/lmfit-py/builtin_models.html) for fitting the spectrum. Doing so, your code will be
# different from this one and you don't need to define a residual function. In such case, you want to look at the
# example 3 on the page http://lmfit.github.io/lmfit-py/builtin_models.html. But let's just pretend we want to write
# our own piece of code and use the Gaussian function implemented in Rampy.
# 
# The shape of the spectrum suggests that at least three peaks are present, because of the two obvious bands near 950
# and 1080 cm$^{-1}$ and a slope break near 1200 cm $^{-1}$. From previous works, we actually know that we have two
# additional peaks (See Mysen, 1990 or Le Losq et al., 2014) in this spectral region located near 1050 and 1150 cm$^{
# -1}$. So we have to fit 5 peaks, and hence, we have 5 intensities variables a1 to a5, 5 frequencies f1 to f5,
# and 5 half width at half peak maximum l1 to l5. This makes a total of 15 parameters. Those variables will be stored
# in the Parameters() object created by the lmfit software (see http://lmfit.github.io/lmfit-py/parameters.html),
# we will go back on this one latter. For now, let just say that the Parameters() object is called "pars" and
# contains the various a1-a5, f1-f5 and l1-l5 parameters, such that we can have their values with using a1 = pars[
# 'a1'].value for instance.
# 
# So let's go. We create the function "residual" with arguments pars (the Parameters() object), the x axis, and,
# in option, the y axis as data and the errors.

def residual(pars, x_data, data=None, eps=None):  # Function definition
    # unpack parameters, extract .value attribute for each parameter
    a1 = pars['a1'].value
    a2 = pars['a2'].value
    a3 = pars['a3'].value
    #     a4 = pars['a4'].value
    #     a5 = pars['a5'].value

    fr1 = pars['f1'].value
    fr2 = pars['f2'].value
    fr3 = pars['f3'].value
    #     f4 = pars['f4'].value
    #     f5 = pars['f5'].value

    l1 = pars['l1'].value
    l2 = pars['l2'].value
    l3 = pars['l3'].value
    #     l4 = pars['l4'].value
    #     l5 = pars['l5'].value

    # Using the Gaussian model function from rampy
    p1 = rp.gaussian(x_data, a1, fr1, l1)
    p2 = rp.gaussian(x_data, a2, fr2, l2)
    p3 = rp.gaussian(x_data, a3, fr3, l3)
    #     peak4 = rp.gaussian(x,a4,f4,l4)
    #     peak5 = rp.gaussian(x,a5,f5,l5)

    p_model = p1 + p2 + p3  # The global model is the sum of the Gaussian peaks

    if data is None:  # if we don't have data, the function only returns the direct calculation
        return p_model, p1, p2, p3
    if eps is None:  # without errors, no ponderation
        return p_model - data
    return (p_model - data) / eps  # with errors, the difference is ponderated


# Note that in the above function, I did not applied the square to (model - data). This is implicitely done by lmfit
# (see http://lmfit.github.io/lmfit-py/fitting.html#fit-func-label for further information on function writting).
# 
# # Fitting
# 
# Ok, we have our optimisation function. So we can go forward and fit the spectrum... 
# 
# We need five Guassians at 950, 1050, 1100, 1150 and 1200 cm$^{-1}$. We set their half-width at half-maximum at the
# same value.
#
params = lmfit.Parameters()
# #               (Name,  Value,  Vary,   Min,  Max,  Expr)
# params.add_many(('a1',   5,   True,  0,      10,  None),
#                 ('f1',   1355,   True, 1350,    1360,  None),
#                 ('l1',   45,   True,  40,      60,  None),
#                 ('a2',   0.5,   True,  0,      4,  None),
#                 ('f2',   1450,  True, 1420,   1480,  None),
#                 ('l2',   40,   True,  25,   50,  None),
#                 ('a3',   5,   True,  0,      None,  None),
#                 ('f3',   1590,  True, 1580,   1595,  None),
#                 ('l3',   40,   True,  25,   50,  None))
# #                 ('a3',   8.5,    True,    0,      None,  None),
# #                 ('f3',   1590,  True, 1580,   1600,  None),
# #                 ('l3',   20,   True,  10,   30,  None),
# #                 ('a4',   1.,   True,  0,      None,  None),
# #                 ('f4',   1620,  True, 1600,    1640,  None),
# #                 ('l4',   20,   True,  10,   30,  None))
# #                 ('a5',   2.,   True,  0,      None,  None),
# #                 ('f5',   1211,  True, 1180,   1220,  None),
# #                 ('l5',   28,   True,  20,   45,  None))
#
#
# # For further details on the Parameters() object, I invite you to look at this page:
# http://lmfit.github.io/lmfit-py/parameters.html . But from the above piece of code, you can already guess that you
# can make specific parameters that vary or not, you can fixe Min or Max values, and you can even put some contrains
# between parameters (e.g., "l1 = l2') using the last "Expr" column. # # You can remark that we applied some
# boundaries for the peak positions, but also for peak widths. This is based on previous fits made for this kind of
# compositions. Typically, in such glass, peaks from Si-O stretch vibrations do not present half-width greater than
# 50 cm$^{-1}$ or smaller than 20 cm$^{-1}$. For instance, the 1080 cm$^{-1}$ peak typically present a half-width of
# ~ 30 cm$^{-1}$ ± 5 cm$^{-1}$ in silica-rich silicate glasses, such that we can apply a tighter constrain there.
# Following such ideas, I put bonds for the parameter values for the half-width of the peaks. This avoid fitting
# divergence. Furthermore, we know approximately the frequencies of the peaks, such that we can also apply bondaries
# for them. This will help the fitting, since in this problem, we have five peaks in a broad envelop that only
# present two significant features at ~950, ~1080 cm$^{-1}$ as well as two barely visible shoulders near 1050 and
# 1200 cm$^{-1}$. But this is a simple case. For some more complex (aluminosilicate) glasses, this 850-1300 cm$^{-1}$
# frequency envelop is even less resolved, such that applying reasonable constrains become crucial for any
# quantitative Raman fitting. # # For starting the fit, as we suppose we have a not bad knowledge of peak frequencies
# (see the discussion right above), a good thing to do is to fix for the first fit the frequencies of the peaks:
#
# # we constrain the positions
# params['f1'].vary = False
# params['f2'].vary = False
# params['f3'].vary = False
# # params['f4'].vary = False
# # params['f5'].vary = False
#
#
# # This avoids any divergence of the fitting procedure regarding the hald-width, because with free frequencies and
# badly estimated half-width and intensities, the fitting procedure always tends to extremely broaden the peaks and
# put them at similar frequencies, with strong overlapping. Starting the fitting procedure by fixing the parameter we
# know the best, i.e. the frequencies, avoid such complications. # # Then, we need to use a large-scale algorithm
# quite robust for fitting. The levenberg-marquart algorithm fails on such fitting problem in my experience. Let's
# choose the Nelder and Mead algorithm for this example: (http://comjnl.oxfordjournals.org/content/7/4/308.short) :
#
# algo = 'nelder'
#
# result = lmfit.minimize(residual, params, method = algo, args=(x_fit, y_fit[:,0])) # fit data with  nelder model from scipy
#
#
# # And now we release the frequencies:
#
#
# # we release the positions but contrain the FWMH and amplitude of all peaks
# params['f1'].vary = True
# params['f2'].vary = True
# params['f3'].vary = True
# # params['f4'].vary = True
# # params['f5'].vary = True
#
# #we fit twice
# result2 = lmfit.minimize(residual, params,method = algo, args=(x_fit, y_fit[:,0])) # fit data with leastsq model from scipy
#
#
# # We can now extract the various things generated by lmfit as well as the peaks:
#
# # In[230]:
#
#
# model = lmfit.fit_report(result2.params)
# yout, peak1,peak2,peak3 = residual(result2.params,x_fit) # the different peaks
# rchi2 = (1/(float(len(y_fit))-15-1))*np.sum((y_fit - yout)**2/sigma**2) # calculation of the reduced chi-square

# print(model)
#
# # And let's have a look at the fitted spectrum:
#
# ##### WE DO A NICE FIGURE THAT CAN BE IMPROVED FOR PUBLICATION
# plt.plot(x_fit,y_fit,'k-')
# plt.plot(x_fit,yout,'r-')
# plt.plot(x_fit,peak1,'b-')
# plt.plot(x_fit,peak2,'b-')
# plt.plot(x_fit,peak3,'b-')
# # plt.plot(x_fit,peak4,'b-')
# # plt.plot(x_fit,peak5,'b-')
#
# plt.xlim(lb,hb)
# plt.ylim(-0.5,10.5)
# plt.xlabel("Raman shift, cm$^{-1}$", fontsize = 14)
# plt.ylabel("Normalized intensity, a. u.", fontsize = 14)
# plt.title("Fig. 3: Fit of the D- and G-bands\n in GO Graphenea with \nthe Nelder Mead algorithm ",fontsize = 14,fontweight = "bold")
# print("rchi-2 = \n"+str(rchi2))
# plt.subplots_adjust(top=0.78)
# plt.savefig(path/'fig3.png', format='png', dpi=300)
#
#
# # Ok, we can test to change the algorithm and use the Levenberg-Marquart one which is well-used for simple problems by a lot of people. We will re-initialize the Params() object and run the entire code written above again.
#
#
#
# algo = 'leastsq' # We will use the Levenberg-Marquart algorithm
# #               (Name,  Value,  Vary,   Min,  Max,  Expr) Here I directly initialize with fixed frequencies
#
# params.add_many(('a1',   5,   True,  0,      10,  None),
#                 ('f1',   1355,   True, 1350,    1360,  None),
#                 ('l1',   45,   True,  40,      60,  None),
#                 ('a2',   0.5,   True,  0,      4,  None),
#                 ('f2',   1450,  True, 1420,   1480,  None),
#                 ('l2',   40,   True,  25,   50,  None),
#                 ('a3',   5,   True,  0,      None,  None),
#                 ('f3',   1590,  True, 1580,   1595,  None),
#                 ('l3',   40,   True,  25,   50,  None))
# #                 ('a2',   0.5,   True,  0,      None,  None),
# #                 ('f2',   1380,  True, 1350,   1410,  None),
# #                 ('l2',   39,   True,  20,   55,  None),
# #                 ('a3',   8.5,    True,    0,      None,  None),
# #                 ('f3',   1590,  True, 1580,   1600,  None),
# #                 ('l3',   20,   True,  10,   30,  None),
# #                 ('a4',   1.,   True,  0,      None,  None),
# #                 ('f4',   1620,  True, 1600,    1640,  None),
# #                 ('l4',   20,   True,  10,   30,  None))
# #                 ('a5',   2.,   True,  0,      None,  None),
# #                 ('f5',   1211,  True, 1180,   1220,  None),
# #                 ('l5',   28,   True,  20,   45,  None))
#
# result = lmfit.minimize(residual, params, method = algo, args=(x_fit, y_fit[:,0]))
# # we release the positions but contrain the FWMH and amplitude of all peaks
# params['f1'].vary = True
# params['f2'].vary = True
# params['f3'].vary = True
# # params['f4'].vary = True
# # params['f5'].vary = True
#
# result2 = lmfit.minimize(residual, params,method = algo, args=(x_fit, y_fit[:,0]))
# model = lmfit.fit_report(result2.params) # the report
# yout, peak1,peak2,peak3 = residual(result2.params,x_fit) # the different peaks
# rchi2 = (1/(float(len(y_fit))-15-1))*np.sum((y_fit - yout)**2/sigma**2) # calculation of the reduced chi-square
#
# ##### WE DO A NICE FIGURE THAT CAN BE IMPROVED FOR PUBLICATION
# plt.plot(x_fit,y_fit,'k-')
# plt.plot(x_fit,yout,'r-')
# plt.plot(x_fit,peak1,'b-')
# plt.plot(x_fit,peak2,'b-')
# plt.plot(x_fit,peak3,'b-')
# # plt.plot(x_fit,peak4,'b-')
# # plt.plot(x_fit,peak5,'b-')
#
# plt.xlim(lb,hb) plt.ylim(-0.5,10.5) plt.xlabel("Raman shift, cm$^{-1}$", fontsize = 14) plt.ylabel("Normalized
# intensity, a. u.", fontsize = 14) plt.title("Fig. 4: Fit of the Si-O stretch vibrations\n in LS4 with \nthe
# Levenberg-Marquardt (LM) algorithm",fontsize = 14,fontweight = "bold") print("rchi-2 = \n"+str(rchi2))
#
#
# # The comparison of Fig. 3 and 4 shows small differences. In this case, and because we have a good error model,
# the LM algorithm converges toward results similar to those of the Nelder-Mead algorithm. You can try to run again
# the calculation with removing the "sigma" input in the "minimize" function used above. You will see that the
# results will diverge much more than in this case. # # A convenient thing about the LM algorithm is that it allows
# to estimate the errors on the fitting parameters. This is not possible with gradient-less algorithms such as the
# Nelder-Mear or the Powell algorithms. For the latters, I will give a piece of code at the end of this notebook that
# allows to estimate good errors on parameters through bootrapping. # # The downside of the LM algorithm is that,
# in my experience, it fails if the envelop of bands to fit is broader than the one used in this example, because it
# seachs at all costs to fit the spectrum as good as possible... This typically results in extrem broadening and
# overlapping of the peaks you try to fit. # # A way to resolve this issue if the use of the LM algorithm is really
# needed is to put tigther constrains on the peak half-widths. # # But another way is to use a more global algorithm
# less prone to diverge from the initial estimations. The Nelder-Mead, Powell (Powell, 1964, Computer Journal 7 (2):
# 155-62) or the COBYLA (see Powell, 2007 Cambridge University Technical Report DAMTP 2007) algorithms can give good
# results for complex problems. Also, the Conjugate Gradient algorithm may be suitable (Wright & Nocedal, “Numerical
# Optimization”, 1999, pp. 120-122). Let's try the latter for now:
#
# # In[214]:
#
#
# algo = 'cg' # We will use the Conjugate Gradient algorithm
# #               (Name,  Value,  Vary,   Min,  Max,  Expr) Here I directly initialize with fixed frequencies
# params.add_many(('a1',   5,   True,  0,      10,  None),
#                 ('f1',   1355,   True, 1350,    1360,  None),
#                 ('l1',   45,   True,  40,      60,  None),
#                 ('a2',   0.5,   True,  0,      4,  None),
#                 ('f2',   1450,  True, 1420,   1480,  None),
#                 ('l2',   40,   True,  25,   50,  None),
#                 ('a3',   5,   True,  0,      None,  None),
#                 ('f3',   1590,  True, 1580,   1595,  None),
#                 ('l3',   40,   True,  25,   50,  None))
# #                 ('a2',   0.5,   True,  0,      None,  None),
# #                 ('f2',   1380,  True, 1350,   1410,  None),
# #                 ('l2',   39,   True,  20,   55,  None),
# #                 ('a3',   8.5,    True,    0,      None,  None),
# #                 ('f3',   1590,  True, 1580,   1600,  None),
# #                 ('l3',   20,   True,  10,   30,  None),
# #                 ('a4',   1.,   True,  0,      None,  None),
# #                 ('f4',   1620,  True, 1600,    1640,  None),
# #                 ('l4',   20,   True,  10,   30,  None))
# #                 ('a5',   2.,   True,  0,      None,  None),
# #                 ('f5',   1211,  True, 1180,   1220,  None),
# #                 ('l5',   28,   True,  20,   45,  None))
#
# result = lmfit.minimize(residual, params, method = algo, args=(x_fit, y_fit[:,0]))
# # we release the positions but contrain the FWMH and amplitude of all peaks
# params['f1'].vary = True
# params['f2'].vary = True
# params['f3'].vary = True
# # params['f4'].vary = True
# # params['f5'].vary = True
#
# result2 = lmfit.minimize(residual, params,method = algo, args=(x_fit, y_fit[:,0]))
# model = lmfit.fit_report(result2.params) # the report
# yout, peak1,peak2,peak3 = residual(result2.params,x_fit) # the different peaks
# rchi2 = (1/(float(len(y_fit))-15-1))*np.sum((y_fit - yout)**2/sigma**2) # calculation of the reduced chi-square
#
# ##### WE DO A NICE FIGURE THAT CAN BE IMPROVED FOR PUBLICATION
# plt.plot(x_fit,y_fit,'k-')
# plt.plot(x_fit,yout,'r-')
# plt.plot(x_fit,peak1,'b-')
# plt.plot(x_fit,peak2,'b-')
# plt.plot(x_fit,peak3,'b-')
# # plt.plot(x_fit,peak4,'b-')
# # plt.plot(x_fit,peak5,'b-')
#
# plt.xlim(lb,hb)
# plt.ylim(-0.5,10.5)
# plt.xlabel("Raman shift, cm$^{-1}$", fontsize = 14)
# plt.ylabel("Normalized intensity, a. u.", fontsize = 14)
# plt.title("Fig. 5: Fit of the Si-O stretch vibrations\n in LS4 with \nthe Conjugate Gradient (CG) algorithm",fontsize = 14,fontweight = "bold")
# print("rchi-2 = \n"+str(rchi2))


# The CG algorithm returns a result close to the Nelder-Mead and the LM algorithms. A bad thing about the CG algorithm is that it is extremely slow in the Scipy implementation... It is (nearly) acceptable for one fit, but for bootstrapping 100 spectra, it is not a good option at all.
# 
# As a last one, we can see what the results look like with the Powell algorithm:

# In[215]:
algo = 'leastsq'  # We will use the Powell algorithm
#               (Name,  Value,  Vary,   Min,  Max,  Expr) Here I directly initialize with fixed frequencies
params.add_many(('a1', 5, True, 0, 10, None),
                ('f1', 1355, False, 1350, 1360, None),
                ('l1', 45, True, 40, 60, None),
                ('a2', 0.5, True, 0, 4, None),
                ('f2', 1450, False, 1420, 1480, None),
                ('l2', 40, True, 25, 50, None),
                ('a3', 5, True, 0, None, None),
                ('f3', 1590, False, 1580, 1595, None),
                ('l3', 40, True, 25, 50, None))
#                 ('a2',   0.5,   True,  0,      None,  None),
#                 ('f2',   1380,  True, 1350,   1410,  None),
#                 ('l2',   39,   True,  20,   55,  None),  
#                 ('a3',   8.5,    True,    0,      None,  None),
#                 ('f3',   1590,  True, 1580,   1600,  None),
#                 ('l3',   20,   True,  10,   30,  None),  
#                 ('a4',   1.,   True,  0,      None,  None),
#                 ('f4',   1620,  True, 1600,    1640,  None),
#                 ('l4',   20,   True,  10,   30,  None))  
#                 ('a5',   2.,   True,  0,      None,  None),
#                 ('f5',   1211,  True, 1180,   1220,  None),
#                 ('l5',   28,   True,  20,   45,  None))

result = lmfit.minimize(residual, params, method=algo, args=(x_fit, y_fit[:, 0]))

# we release the positions but contrain the FWMH and amplitude of all peaks 
params['f1'].vary = True
params['f2'].vary = True
params['f3'].vary = True
# params['f4'].vary = True
# params['f5'].vary = True

result2 = lmfit.minimize(residual, params, method=algo, args=(x_fit, y_fit[:, 0]))
model = lmfit.fit_report(result2.params)  # the report
yout, peak1, peak2, peak3 = residual(result2.params, x_fit)  # the different peaks
rchi2 = (1 / (float(len(y_fit)))) * np.sum((y_fit - yout) ** 2 / sigma ** 2)  # calculation of the reduced chi-square

### WE DO A NICE FIGURE THAT CAN BE IMPROVED FOR PUBLICATION
f4 = plt.figure(4)
plt.plot(x_fit, y_fit, 'k-')
plt.plot(x_fit, yout, 'r-')
plt.plot(x_fit, peak1, 'b-')
plt.plot(x_fit, peak2, 'b-')
plt.plot(x_fit, peak3, 'b-')
# plt.plot(x_fit,peak4,'b-')
# plt.plot(x_fit,peak5,'b-')

plt.xlim(lb, hb)
plt.ylim(-0.5, 10.5)
plt.xlabel("Raman shift, cm$^{-1}$", fontsize=14)
plt.ylabel("Normalized intensity, a. u.", fontsize=14)
plt.title("Fig. 6: Fit of the Si-O stretch vibrations\n in LS4 with \nthe Powell algorithm", fontsize=14,
          fontweight="bold")
print("rchi-2 = \n" + str(rchi2))

plt.show()

input()
# You see in Fig. 6 that the results are, again, close to those of the other algorithms, at the exception of the two last peaks. The intensity and the frequency of the peak near 1200 cm$^{-1}$ is higher in this fit than in the others.
# 
# So one important thing that has to be remembered is that, with the same parameter inputs, you will obtain different
# results with using different fitting algorithms. The above results are close because the fitting example is quite
# simple. Actually, all the results given above seem reasonable. The experience with other spectra from other
# silicate and aluminosilicate glasses is that the Nelder-Mead and Powell algorithms will provide the most robust
# ways to fit the spectra.


# # Error estimations
# 
# Errors can be estimated with using the "confidence" function if you used the Levenberg-Marquardt algorithm. See the
# examples here: http://lmfit.github.io/lmfit-py/confidence.html .
# 
# If you use a large-scale gradient-less algorithm such as the Nelder-Mead or the Powell algorithms, you cannot do
# that. Thus, to calculate the errors on the parameters that those algorithms provide as well as the error introduced
# by choosing one or the other algorithm, we can use the bootstrapping technic. Several descriptions on the internet
# are available for this technic, so I will skip a complete description here.
# 
# A quick overview is to say that we have datapoints Yi affected by errors e_Yi. We assume that the probability
# density function of the Yi points is Gaussian. According to the Central Theorem Limit, this probably is a good
# assumption. Therefore, for each frequency in the spectrum of Fig.1, we have points that are probably at an
# intensity of Yi but with an error of e_Yi. To estimate how this uncertainties affect our fitting results,
# we can pick new points in the Gaussian distribution of mean Yi with a standard deviation e_Yi, and construct whole
# new spectra that we will fit. We will repeat this procedure N times.
# 
# In addition to that, we can also randomly choose between the Nelder-Mead or the Powell algorithm during the new
# fits, such that we will take also into account our arbitrary choice in the fitting algorithm for calculating the
# errors on the estimated parameters.
# 
# A last thing would be to randomly change a little bit the initial values of the parameters, but this is harder to
# implement so we will not do it for this example.
# 
# First of all, we have to write a Python function that will randomly sample the probability density functions of the
# points of the spectrum of Fig. 1. Here is the piece of code I wrote for doing so:

# #### Bootstrap function
# def bootstrap(data, ese,nbsample):
#     # Bootstrap of Raman spectra. We generate new datapoints with the basis of existing data and their standard deviation
#     N = len(data)
#     bootsamples = np.zeros((N,nbsample))
#
#     for i in range(nbsample):
#         for j in range(N):
#             bootsamples[j,i] = np.random.normal(data[j], ese[j], size=None)
#     return bootsamples
#
#
# # Now we will define how much new spectra we want to generate (the nbsample option of the bootstrap function), and we will run the previous function.
#
# get_ipython().run_cell_magic('time', '', 'nboot = 10 # Number of bootstrap samples, I set it to a low value for the example but usually you want thousands there\ndata_resampled = bootstrap(y_fit[:,0],sigma,nboot)# resampling of data + generate the output parameter tensor')
#
#
# # Now, we will create a loop which is going to look at each spectrum in the data_resampled variable, and to fit them with the procedure already described.
# #
# # For doing so, we need to declare a couple of variables to record the bootstrap mean fitting error, in order to see if we generated enought samples to obtain a statistically representative bootstrapping process, and to record each set of parameters obtained for each bootstrapped spectrum.
#
# para_output = np.zeros((5,3,nboot)) # 5 x 3 parameters x N boot samples
# bootrecord = np.zeros((nboot)) # For recording boot strap efficiency
#
# for nn in range(nboot):
#     algos = ['powell','nelder']
#     algo = random.choice(algos) # We randomly select between the Powell or Nelder_mear algorithm
#     params = lmfit.Parameters()
#     #               (Name,  Value,  Vary,   Min,  Max,  Expr) Here I directly initialize with fixed frequencies
#     params.add_many(('a1',   5,   True,  0,      10,  None),
#                 ('f1',   1355,   True, 1350,    1360,  None),
#                 ('l1',   45,   True,  40,      60,  None),
#                 ('a2',   0.5,   True,  0,      4,  None),
#                 ('f2',   1450,  True, 1420,   1480,  None),
#                 ('l2',   40,   True,  25,   50,  None),
#                 ('a3',   5,   True,  0,      None,  None),
#                 ('f3',   1590,  True, 1580,   1595,  None),
#                 ('l3',   40,   True,  25,   50,  None))
# #                 ('a5',   2.,   True,  0,      None,  None),
# #                 ('f5',   1211,  True, 1180,   1220,  None),
# #                 ('l5',   28,   True,  20,   45,  None))
#
# #     params.add_many(('a1',   24,   True,  0,      None,  None),
# #                 ('f1',   946,   True, 910,    970,  None),
# #                 ('l1',   26,   True,  20,      50,  None),
# #                 ('a2',   35,   True,  0,      None,  None),
# #                 ('f2',   1026,  True, 990,   1070,  None),
# #                 ('l2',   39,   True,  20,   55,  None),
# #                 ('a3',   85,    True,    70,      None,  None),
# #                 ('f3',   1082,  True, 1070,   1110,  None),
# #                 ('l3',   31,   True,  25,   35,  None),
# #                 ('a4',   22,   True,  0,      None,  None),
# #                 ('f4',   1140,  True, 1110,    1160,  None),
# #                 ('l4',   35,   True,  20,   50,  None),
# #                 ('a5',   4,   True,  0,      None,  None),
# #                 ('f5',   1211,  True, 1180,   1220,  None),
# #                 ('l5',   28,   True,  20,   45,  None))
#
#     result = lmfit.minimize(residual, params, method = algo, args=(x_fit, data_resampled[:,nn],sigma))
#     # we release the positions but contrain the FWMH and amplitude of all peaks
#     params['f1'].vary = True
#     params['f2'].vary = True
#     params['f3'].vary = True
# #     params['f4'].vary = True
# #     params['f5'].vary = True
#
#     result2 = lmfit.minimize(residual, params,method = algo, args=(x_fit, data_resampled[:,nn], sigma))
#
#     vv = result2.params.valuesdict()
#     para_output[0,0,nn] = vv['a1']
#     para_output[1,0,nn] = vv['a2']
#     para_output[2,0,nn] = vv['a3']
# #     para_output[3,0,nn] = vv['a4']
# #     para_output[4,0,nn] = vv['a5']
#
#     para_output[0,1,nn] = vv['f1']
#     para_output[1,1,nn] = vv['f2']
#     para_output[2,1,nn] = vv['f3']
# #     para_output[3,1,nn] = vv['f4']
# #     para_output[4,1,nn] = vv['f5']
#
#     para_output[0,2,nn] = vv['l1']
#     para_output[1,2,nn] = vv['l2']
#     para_output[2,2,nn] = vv['l3']
# #     para_output[3,2,nn] = vv['l4']
# #     para_output[4,2,nn] = vv['l5']
#
# para_mean = np.mean(para_output,axis=2)
# para_ese = np.std(para_output,axis=2)
# for kjy in range(nboot):
#     if kjy == 0:
#         bootrecord[kjy] = 0
#     else:
#         bootrecord[kjy] = np.sum(np.std(para_output[:,:,0:kjy],axis=2))
#
#
# # We can have a view at the mean values and standard deviation of the parameters that have been generated by the bootstrapping:
#
# # In[220]:
#
#
# para_mean
#
#
# # In[221]:
#
#
# para_ese
#
#
# # Those errors are probably the best estimates of the errors that affect your fitting parameters. You can add another bootstrapping function for changing of, saying, 5 percents the initial estimations of the parameters, and you will have a complete and coherent estimation of the errors affecting the fits. But for most cases, the errors generated by this above bootstrapping technic are already quite robust.
# #
# # We can see if we generated enought samples to have valid bootstrap results by looking at how the mean value of the parameters and their error converge. To do a short version of such thing, we can also look at how the summation of the errors of the parameters change with the iteration number. If the summation of errors becomes constant, we can say that we have generated enought bootstrap samples to have a significant result, statistically speaking.
#
# # In[222]:
#
#
# plt.plot(np.arange(nboot)+1,bootrecord,'ko')
# plt.xlim(0,nboot+1)
# plt.xlabel("Number of iterations",fontsize = 14)
# plt.ylabel("Summed errors of parameters",fontsize = 14)
# plt.title("Fig. 7: Bootstrap iterations for convergence",fontsize = 14, fontweight = 'bold')
#
#
# # We see from the above figure that the algorithm seems to have converged after 70 iterations. Therefore, we need to generate at least 70 spectra with the bootstrap function to obtain a good estimate of the errors that affect the parameters.
# #
# # # Conclusion
# # This IPython notebook showed how spectra can be corrected from any baseline, how it is possible to use lmfit to fit it with Gaussian peaks, how changing the optimisation algorithm can change the results, and how we can estimate the errors on the calculated parameters with using the bootstrapping technic.
# #
# # Several peak models are defined in the Rampy toolbox, or directly in lmfit. You can look at the relevant instructions for both software to use other peak models in the above calculation. Results can be saved in textfiles, through using the Python output functions like np.savetxt or other ways. See the relevant python documentation for doing so. Any comments on this program will be welcome.
#
# # In[260]:
#
#
# get_ipython().system('jupyter nbconvert --to script Raman-Ana.ipynb')
