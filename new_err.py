import numpy as np
import fitting_scripts
from scipy import interpolate

def tmp_func(_T, _g, _rv, _s, _l, _m):
    c = 299792.458 # Speed of light in km/s 
    model_test=fitting_scripts.interpolating_model_DA(_T,(_g/100),mod_type=_m)
    try: model_w, model_f = model_test[:,0], model_test[:,1]
    except:
        print("Could not load the model")
        return 1
    else:
        #Load Balmer line and Check if lines are inside spectra l 
        # => normalise spectra and interpolate onto same wavelength scale
        norm_model, m_cont_flux=fitting_scripts.norm_spectra(model_test)
        spec_n, cont_flux = fitting_scripts.norm_spectra(_s)
        m_wave_n, m_flux_n, sn_w =norm_model[:,0]*(_rv+c)/c, norm_model[:,1], spec_n[:,0]
        m_flux_n_i = interpolate.interp1d(m_wave_n,m_flux_n,kind='linear')(sn_w)
        l_crop = _l[(_l[:,0]>sn_w.min()) & (_l[:,1]<sn_w.max())]
        #Initialise: normalised models and spectra in line region, and chi2
        tmp_lines_m, lines_s, sum_l_chi2 = [], [], 0
        for i in range(len(l_crop)):
            #for each line, crop model and spectra to line only if line 
            #region is entirely covered by spectrum
            l_c0, l_c1 = l_crop[i,0], l_crop[i,1]
            if (l_c1 < sn_w.max()) & (l_c0 > sn_w.min()):
                l_m = m_flux_n_i.transpose()[(sn_w>=l_c0)&(sn_w<=l_c1)].transpose()
                l_s = spec_n[(sn_w>=l_c0)&(sn_w<=l_c1)]
                #renormalise models to spectra in line region & calculate chi2+sum
                l_m = l_m*np.sum(l_s[:,1])/np.sum(l_m)
                sum_l_chi2 += np.sum(((l_s[:,1]-l_m)/l_s[:,2])**2)
                tmp_lines_m.append(l_m)
                lines_s.append(l_s)
        model_test=np.dstack((model_w, model_f))[0]
        return lines_s, tmp_lines_m, model_test, sum_l_chi2
        
def fit_func_test(x,spec,linee,models='da2014',mode=0):
    """Requires: x - initial guess of T, g, and rv
       spec - spectrum to fit
       linee - list of lines to fit
       mode=0 is for finding bestfit, mode=1 for fitting & retriving specific model """
    tmp = tmp_func(x[0], x[1], x[2], spec, linee, models)
    if tmp == 1: pass
    elif mode==0: return tmp[3] #this is the quantity that gets minimized
    elif mode==1: return tmp[0], tmp[1], spec, tmp[2]

def err_t(x,rv,valore,spec,linee,models='da2014'):
    """Script finds errors by minimising function at chi+1 rather than chi
       Requires: x; rv - initial guess of T, g; rv
       valore - the chi value of the best fit
       spec - spectrum to fit
       linee - list of lines to fit """
    tmp = tmp_func(x[0], x[1], rv, spec, linee, models)
    if tmp != 1: return abs(tmp[3]-(valore+1.)) #this is quantity that gets minimized 
    else: pass
