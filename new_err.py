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
        norm_model, m_cont_flux=fitting_scripts.norm_spectra(model_test)
        #Load Balmer line and Check if lines are inside spectra l, then normalise spectra
        l_crop = _l[(_l[:,0]>_s[:,0].min()) & (_l[:,1]<_s[:,0].max())]
        spec_n, cont_flux = fitting_scripts.norm_spectra(_s)
        #Interpolate spectrum and model onto same resolution
        m_wave_n, m_flux_n =norm_model[:,0]*(_rv+c)/c, norm_model[:,1]
        tck_l_m = interpolate.interp1d(m_wave_n,m_flux_n,kind='linear')
        spec_n_w=spec_n[:,0][spec_n[:,0]>np.min(m_wave_n)]
        m_flux_n_i = tck_l_m(spec_n[:,0])

        #Initialise: normalised models and spectra in line region, and chi2
        tmp_lines_m, lines_s, sum_l_chi2 = [], [], 0
        for i in range(len(l_crop)):
            #for each line, crop model and spectra to line only if line 
            #region is entirely covered by spectrum ...spec_n[:,0] --> spec_n_w
            if (l_crop[i,1] < _s[:,0].max()) & (l_crop[i,0] > _s[:,0].min()):
                l_m = m_flux_n_i.transpose()[(spec_n[:,0]>=l_crop[i,0])&(spec_n[:,0]<=l_crop[i,1])].transpose()
                l_s = spec_n[(spec_n[:,0]>=l_crop[i,0])&(spec_n[:,0]<=l_crop[i,1])]
                lines_s.append(l_s)
                #renormalise models to spectra in line region
                l_m = l_m*np.sum(l_s[:,1])/np.sum(l_m)
                tmp_lines_m.append(l_m)
                #calculate chi2
                sum_l_chi2 += np.sum(((l_s[:,1]-l_m)/l_s[:,2])**2)
        #mean chi2 over lines
        model_test=np.dstack((model_w, model_f))[0]
        return lines_s, tmp_lines_m, model_test, sum_l_chi2
        
def fit_func_test(x,spec,linee,models='da2014',mode=0):
    """Requires: x - initial guess of T, g, and rv
       spec - spectrum to fit
       linee - list of lines to fit
       mode=0 is for finding bestfit, mode=1 for fitting & retriving specific model """
    T,g,rv=x
    spectra=spec
    tmp = tmp_func(T, g, rv, spectra, linee, models)
    if tmp == 1: pass
    elif mode==0: return tmp[3] #this is the quantity that gets minimized
    elif mode==1: return tmp[0], tmp[1], spectra, tmp[2]

def err_t(x,rv,valore,spec,linee,models='da2014'):
    """Script finds errors by minimising function at chi+1 rather than chi
       Requires: x - initial guess of T, g
       rv - rv value from best fit
       valore - the chi value of the best fit
       spec - spectrum to fit
       linee - list of lines to fit """
    T,g=x
    spectra=spec
    tmp = tmp_func(T, g, rv, spectra, linee, models)
    if tmp != 1: return abs(tmp[3]-(valore+1.)) #this is quantity that gets minimized 
    else: pass
