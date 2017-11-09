import numpy as np
import fitting_scripts
from scipy import interpolate

def tmp_func(_T, _g, _rv, _spectra, _linee, _models):
    c = 299792.458 # Speed of light in km/s 
    model_test=fitting_scripts.interpolating_model_DA(_T,(_g/100),mod_type=_models)
    try: model_w, model_f =model_test[:,0], model_test[:,1]
    except:
        print("Could not load the model")
        return 1
    else:
        norm_model, m_cont_flux=fitting_scripts.norm_spectra(model_test)
        #Load Balmer line and Check if lines are inside spectra l
        line_crop = _linee[(_linee[:,0]>_spectra[:,0].min()) & (_linee[:,1]<_spectra[:,0].max())]
        #Normalize the spectrum
        spectra_n, cont_flux = fitting_scripts.norm_spectra(_spectra)
        #Interpolate spectrum and model onto same resolution
        m_wave_n, m_flux_n =norm_model[:,0]*(_rv+c)/c, norm_model[:,1]
        tck_l_m = interpolate.interp1d(m_wave_n,m_flux_n,kind='linear')
        spectra_n_w=spectra_n[:,0][spectra_n[:,0]>np.min(m_wave_n)]
        m_flux_n_i = tck_l_m(spectra_n[:,0])

        #Initialise: normalised models and spectra in line region, and chi2
        tmp_lines_m, lines_s, l_chi2 = [], [], []
        list=np.array([])
        for c in xrange(len(line_crop)):
            #for each line, crop model and spectra to line only if line 
            #region is entirely covered by spectrum ...spectra_n[:,0] --> spectra_n_w
            if (line_crop[c,1] < _spectra[:,0].max()) & (line_crop[c,0] > _spectra[:,0].min()):
                l_m = m_flux_n_i.transpose()[(spectra_n[:,0]>=line_crop[c,0])&(spectra_n[:,0]<=line_crop[c,1])].transpose()
                l_s = spectra_n[(spectra_n[:,0]>=line_crop[c,0])&(spectra_n[:,0]<=line_crop[c,1])]
                lines_s.append(l_s)
                #renormalise models to spectra in line region
                l_m = l_m*np.sum(l_s[:,1])/np.sum(l_m)
                tmp_lines_m.append(l_m)
                #calculate chi2
                l_chi2.append( np.sum(((l_s[:,1]-l_m)/l_s[:,2])**2))
                list=np.concatenate((list,((l_s[:,1]-l_m)/l_s[:,2])**2),axis=0)
        #mean chi2 over lines
        l_chi2 = np.array(l_chi2)
        model_test=np.dstack((model_w, model_f))[0]
        return lines_s, tmp_lines_m, model_test, np.sum(l_chi2)
        
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
    elif mode==1: return tmp[0], tmp[1], spectra, tmp[2]#,l_chi2#, x_list 

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
