import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import optimize
import fitting_scripts
import new_err

model_c='da2014'
basedir='/Users/christophermanser/Storage/PhD_files/DESI/WDFitting'
c = 299792.458 # Speed of light in km/s
plot = True

# Loads the input spectrum as sys.argv[1], first input
spec1,spec2,spec3=np.loadtxt(sys.argv[1],usecols=(0,1,2),unpack=True)
spec1=spec1[np.isnan(spec2)==False]
spec3=spec3[(np.isnan(spec2)==False) | (spec1>3500)]
spec2=spec2[(np.isnan(spec2)==False) | (spec1>3500)]
spec1=spec1[(spec1>3500)]
spectra=np.dstack((spec1,spec2,spec3))[0]

#load lines to fit
line_crop = np.loadtxt(basedir+'/line_crop.dat')
#fit entire grid to find good starting point
best=fitting_scripts.fit_line(spectra,model_in=None,quick=True,model=model_c)
first_T, first_g = best[2][0], best[2][1]
all_chi, all_TL  = best[4], best[3]

#------find starting point for secondary solution
if first_T < 13000.:
    other_TL=all_TL[all_TL[:,0]>13000.]
    other_chi=all_chi[all_TL[:,0]>13000.]
    other_sol= other_TL[other_chi==np.min(other_chi)]
elif first_T > 13000.:
    other_TL=all_TL[all_TL[:,0]<13000.]
    other_chi=all_chi[all_TL[:,0]<13000.]
    other_sol= other_TL[other_chi==np.min(other_chi)]

# Calculate a cropped line list for new_err & normalises spectrum
l_crop = line_crop[(line_crop[:,0]>spec1.min()) & (line_crop[:,1]<spec1.max())]
spec_n, cont_flux = fitting_scripts.norm_spectra(spectra)
# Find best fitting model and then calculate error
new_best=optimize.fmin(new_err.fit_func_test,(first_T,first_g,10.),args=(spec_n,l_crop,model_c,0),disp=0,xtol=1.,ftol=1.,full_output=1)
other_T=optimize.fmin(new_err.err_t,(first_T,first_g),args=(new_best[0][2],new_best[1],spec_n,l_crop,model_c),disp=0,xtol=1.,ftol=1.,full_output=0)
best_T, best_g, best_rv = new_best[0][0], new_best[0][1], new_best[0][2]
print("First solution")
print("T = ", best_T, abs(best_T-other_T[0]))
print("logg = ", best_g/100, abs(best_g-other_T[1])/100)
print("rv=",best_rv)
# get and save best model
lines_s,lines_m,model_n=new_err.fit_func_test((best_T,best_g,best_rv),spec_n,l_crop,models=model_c,mode=1)

#repeat fit for secondary solution
second_best=optimize.fmin(new_err.fit_func_test,(other_sol[0][0],other_sol[0][1],best_rv),args=(spec_n,l_crop,model_c,0),disp=0,xtol=1.,ftol=1.,full_output=1)
other_T2=optimize.fmin(new_err.err_t,(other_sol[0][0],other_sol[0][1]),args=(best_rv,second_best[1],spec_n,l_crop,model_c),disp=0,xtol=1.,ftol=1.,full_output=0)
s_best_T, s_best_g, s_best_rv = second_best[0][0], second_best[0][1], second_best[0][2]
print("\nSecond solution")
print("T = ", s_best_T, abs(s_best_T-other_T2[0]))
print("logg = ", s_best_g/100, abs(s_best_g-other_T2[1])/100)
# get and save best model
lines_s_o,lines_m_o,model_n_o=new_err.fit_func_test((s_best_T,s_best_g,s_best_rv),spec_n,l_crop,models=model_c,mode=1)

#=======================plotting===============================================
if plot == True:
    fig=plt.figure(figsize=(8,5))
    ax1 = plt.subplot2grid((1,4), (0, 3))
    step = 0
    for i in range(1,6): # plots Halpha (i=0) to H6 (i=5)
        min_p   = lines_s[i][:,0][lines_s[i][:,1]==np.min(lines_s[i][:,1])][0]
        min_p_o = lines_s_o[i][:,0][lines_s_o[i][:,1]==np.min(lines_s_o[i][:,1])][0]
        ax1.plot(lines_s[i][:,0]-min_p,lines_s[i][:,1]+step,color='k')
        ax1.plot(lines_s[i][:,0]-min_p,lines_m[i]+step,color='r')
        ax1.plot(lines_s_o[i][:,0]-min_p_o,lines_m_o[i]+step,color='g')
        step+=0.5
    xticks = ax1.xaxis.get_major_ticks()
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    ax2 = plt.subplot2grid((1,4), (0, 0),colspan=3)
    ax2.plot(spectra[:,0],spectra[:,1],color='k')
    # Adjust the flux of models to match the spectrum
    model_n[np.isnan(model_n)], model_n_o[np.isnan(model_n_o)] = 0.0, 0.0
    check_f_spec=spectra[:,1][np.logical_and(spectra[:,0]>4500., spectra[:,0]<4700.)]
    check_f_model=model_n[:,1][np.logical_and(model_n[:,0]>4500., model_n[:,0]<4700.)]
    adjust=np.average(check_f_model)/np.average(check_f_spec)
    ax2.plot(model_n[:,0]*(best_rv+c)/c,model_n[:,1]/adjust,color='r')
    check_f_model_o=model_n_o[:,1][np.logical_and(model_n_o[:,0]>4500., model_n_o[:,0]<4700.)]
    adjust_o=np.average(check_f_model_o)/np.average(check_f_spec)
    ax2.plot(model_n_o[:,0]*(best_rv+c)/c,model_n_o[:,1]/adjust_o,color='g')

    ax2.set_ylabel(r'F$_{\lambda}$ [erg cm$^{-2}$ s$^{-1} \AA^{-1}$]',fontsize=12)#labelpad=24
    ax2.set_xlabel(r'Wavelength $(\AA)$',fontsize=12)
    plt.show()
    plt.close()
