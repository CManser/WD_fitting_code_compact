import numpy as np
import os
import matplotlib.pyplot as plt
basedir='/Users/christophermanser/Storage/PhD_files/DESI/WDFitting'

def air_to_vac(wavin):
    """Converts spectra from air wavelength to vacuum to compare to models"""
    return wavin/(1.0 + 2.735182e-4 + 131.4182/wavin**2 + 2.76249e8/wavin**4)


def _band_limits(band):
    """ give mag band eg "sdss_r" & return outer limits for use in models etc"""
    mag = np.loadtxt(basedir+'/sm/'+band+'.dat')
    mag = mag[mag[:,1]>0.05]
    return [mag[:,0].min(),mag[:,0].max()]


def norm_models(quick=True, model='da2014', testing=False):
    """ Import Normalised WD Models
    Optional arguments:
        quick=True   : Use presaved model array. Check is up to date
        model='da2014': Which model grid to use: List shown below in mode_list
        testing=False      : plot testing image
    Return [model_list,model_param,orig_model_wave,orig_model_flux,tck_model,r_model]
    """
    if quick: # Use preloaded tables
        model_list = ['da2014','pier','pier3D','pier3D_smooth','pier_rad','pier1D',
                      'pier_smooth','pier_rad_smooth','pier_rad_fullres','pier_fullres']
        if model not in model_list: raise wdfitError('Unknown "model" in norm_models')
        fn, d = '/wdfit.'+model+'.lst', '/WDModels_Koester.'+model+'_npy/'
        model_list  = np.loadtxt(basedir+fn, usecols=[0], dtype=np.string_).astype(str)
        model_param = np.loadtxt(basedir+fn, usecols=[1,2])
        m_spec = np.load(basedir+d+model_list[0])
        m_wave = m_spec[:,0]
        out_m_wave = m_wave[(m_wave>=3400)&(m_wave<=13000)]
        norm_m_flux = np.load(basedir+'/norm_m_flux.'+model+'.npy')
        #
        if out_m_wave.shape[0] != norm_m_flux.shape[1]:
            raise wdfitError('l and f arrays not correct shape in norm_models')
    #Calculate
    else:
        from scipy import interpolate
        #Load models from models()
        model_list,model_param,m_wave,m_flux,r_model = models(quick=quick, model=model)
        #Range over which DA is purely continuum
        norm_range = np.loadtxt(basedir+'/wide_norm_range.dat', usecols=[0,1])
        norm_range_s = np.loadtxt(basedir+'/wide_norm_range.dat', usecols=[2], dtype='string')
        #for each line, for each model, hold l and f of continuum
        cont_l = np.empty([len(m_flux), len(norm_range)])
        cont_f = np.empty([len(m_flux), len(norm_range)])
        #for each zone
        
        for j in range(len(norm_range)):
            if (norm_range[j,0] < m_wave.max()) & (norm_range[j,1] > m_wave.min()):
                #crop
                _f = m_flux.transpose()[(m_wave>=norm_range[j,0])&(m_wave<=norm_range[j,1])].transpose()
                _l = m_wave[(m_wave>=norm_range[j,0])&(m_wave<=norm_range[j,1])]
                
                #interpolate region
                print(norm_range[j,0], norm_range[j,1])
                print(np.size(_l))
                
                tck = interpolate.interp1d(_l,_f,kind='cubic')
                #interpolate onto 10* resolution
                l = np.linspace(_l.min(),_l.max(),(len(_l)-1)*10+1)
                f = tck(l)
                #print f
                #find maxima and save
                if norm_range_s[j]=='P':
                    for k in range(len(f)):
                        cont_l[k,j] = l[f[k]==f[k].max()][0]
                        cont_f[k,j] = f[k].max()
                #find mean and save
                elif norm_range_s[j]=='M':
                    for k in range(len(f)):
                        cont_l[k,j] = np.mean(l)
                        cont_f[k,j] = np.mean(f[k])
                else:
                    print('Unknown norm_range_s, ignoring')
        #Continuum
        if (norm_range.min()>3400) & (norm_range.max()<13000):
            out_m_wave = m_wave[(m_wave>=3400)&(m_wave<=13000)]
        else:
            raise wdfitError('Normalised models cropped to too small a region')
        cont_m_flux = np.empty([len(m_flux),len(out_m_wave)])
        for i in range(len(m_flux)):
            #not suitable for higher order fitting
            tck = interpolate.splrep(cont_l[i],cont_f[i], t=[3885,4340,4900,6460], k=3)
            cont_m_flux[i] = interpolate.splev(out_m_wave,tck)
        #Normalised flux
        norm_m_flux = m_flux.transpose()[(m_wave>=3400)&(m_wave<=13000)].transpose()/cont_m_flux
        np.save(basedir+'/norm_m_flux.'+model+'.npy', norm_m_flux)
        #
        #testing
        if testing:
            import matplotlit.pyplot as plt
            def p():
                plt.figure(figsize=(7,9))
                ax1 = pl.subplot(211)
                llst = [3885, 4340, 4900, 6460]
                for num in llst: plt.axvline(num, color='g',zorder=1)
                plt.plot(m_wave,m_flux[i], color='grey', lw=0.8, zorder=2)
                plt.plot(out_m_wave,cont_m_flux[i], 'b-', zorder=3)
                plt.scatter(cont_l[i], cont_f[i], edgecolors='r', facecolors='none', zorder=20)
                ax2 = plt.subplot(212, sharex=ax1)
                plt.axhline([1], color='g')
                plt.plot(out_m_wave, norm_m_flux[i], 'b-')
                plt.ylim([0,2])
                plt.xlim([3400,13000])
                plt.show()
                return
            for i in np.where(model_param[:,1]==8.0)[0][::8]: p()
    return [out_m_wave,norm_m_flux,model_param]


def norm_spectra(spectra, add_infinity=True):
    """
    Normalised spectra by DA WD continuum regions
    spectra of form array([wave,flux,error]) (err not necessary)
    only works on SDSS spectra region
    Optional:
        EDIT wide_norm_range to change whether the region is fitted for a peak or mean'd
        add_infinity=False : add a spline point at [infinity,0]
    returns spectra, cont_flux
    """
    from scipy import interpolate
    start_n=np.array([3770.,3796.,3835.,3895.,3995.,4130.,4490.,4620.,5070.,5200.,
                      6000.,7000.,7550.,8400.])
    end_n=np.array([3795.,3830.,3885.,3960.,4075.,4290.,4570.,4670.,5100.,5300.,
                    6100.,7050.,7600.,8450.])
    norm_range_s=np.array(['P','P','P','P','P','P','M','M','M','M','M','M','M','M'])
    if len(spectra[0])>2:
        s_nr = np.zeros([len(start_n),3])
        spectra[:,2][spectra[:,2]==0.] = spectra[:,2].max()
    wave, flux, err
    else: s_nr = np.zeros([len(start_n),2])
    for j in range(len(start_n)):
        if (start_n[j] < spectra[:,0].max()) & (end_n[j] > spectra[:,0].min()):
            #crop
            _s = spectra[(spectra[:,0]>=start_n[j])&(spectra[:,0]<=end_n[j])]
            #Avoids gappy spectra
            #More points than k
            k=3
            if len(_s)>k:
                #interpolate onto 10* resolution
                l = np.linspace(_s[:,0].min(),_s[:,0].max(),(len(_s)-1)*10+1)
                if len(spectra[0])>2:
                    tck = interpolate.splrep(_s[:,0],_s[:,1],w=1/_s[:,2], s=1000)
                    #median errors for max/mid point
                    s_nr[j,2] = np.median(_s[:,2]) / np.sqrt(len(_s[:,0]))
                else: tck = interpolate.splrep(_s[:,0],_s[:,1],s=0.0)
                f = interpolate.splev(l,tck)
                #find maxima and save
                if norm_range_s[j]=='P': s_nr[j,0], s_nr[j,1] = l[f==f.max()][0], f.max()
                #find mean and save
                elif norm_range_s[j]=='M': s_nr[j,0:2] = np.mean(l), np.mean(f)
                else: print('Unknown norm_range_s, ignoring')
    s_nr = s_nr[ s_nr[:,0] != 0 ]
    #
    #t parameter chosen by eye. Position of knots. need more near blue end
    #allow for short spectra though not as effective
    if s_nr[:,0].max() < 4901: knots=None
    elif s_nr[:,0].max() < 6460: knots = [3000,4900,4100,4340,4860,int(s_nr[:,0].max()-5)]
    else: knots = [3885,4340,4900,6460]
    if s_nr[:,0].min() > 3885:
        print('Warning: knots used for spline normalisation not suitable for high order fitting')
        knots=knots[1:]
    if s_nr[:,0].min() > 4340:
        print('Warning: knots used for spline normalisation probably bad')
        knots=None
    # add a point at infinity, 0 for spline to fit too. error = mean of other errors.
    if add_infinity:
        if s_nr.shape[1] > 2:
            s_nr = np.vstack([ s_nr, np.array([90000.,0., np.mean(s_nr[:,2]) ]) ])
            s_nr = np.vstack([ s_nr, np.array([100000.,0., np.mean(s_nr[:,2]) ]) ])
        else:
            s_nr = np.vstack([ s_nr, np.array([90000.,0.]) ])
            s_nr = np.vstack([ s_nr, np.array([100000.,0.]) ])
    #weight by errors
    try:
        if len(spectra[0])>2: tck = interpolate.splrep(s_nr[:,0],s_nr[:,1], w=1/s_nr[:,2], t=knots, k=3)
        else: tck = interpolate.splrep(s_nr[:,0],s_nr[:,1], t=knots, k=3)
    except ValueError:
        knots=None
        if len(spectra[0])>2: tck = interpolate.splrep(s_nr[:,0],s_nr[:,1], w=1/s_nr[:,2], t=knots, k=3)
        else: tck = interpolate.splrep(s_nr[:,0],s_nr[:,1], t=knots, k=3)
    cont_flux = interpolate.splev(spectra[:,0],tck).reshape(spectra[:,0].size, 1)
    spectra_ret = np.copy(spectra)
    spectra_ret[:,1:] = spectra_ret[:,1:]/cont_flux
    return spectra_ret, cont_flux


def models(quick=True, quiet=True, band='sdss_r', model='da2014'):
    """Import WD Models
    Optional:
        quick=True   : Use presaved model array. Check is up to date
        quiet=True   : verbose
        band='sdss_r': which mag band to calculate normalisation over (MEAN, not folded)
        model: Which model grid to use
    Return [model_list,model_param,orig_model_wave,orig_model_flux,tck_model,r_model]
    """
    from scipy import interpolate
    model_list = ['da2014','pier','pier3D','pier3D_smooth','pier_rad','pier1D','pier_smooth','pier_rad_smooth','pier_rad_fullres','pier_fullres']
    if model not in model_list: raise wdfitError('Unknown "model" in models')
    else: fn, d = '/wdfit.'+model+'.lst', '/WDModels_Koester.'+model+'/'
    #Load in table of all models
    model_list = np.loadtxt(basedir+fn, usecols=[0], dtype=np.string_).astype(str)
    model_param = np.loadtxt(basedir+fn, usecols=[1,2])
    orig_model_wave = np.loadtxt(basedir+d+model_list[0], usecols=[0])
    #
    if not quiet: print('Loading Models')
    if quick:
        orig_model_flux = np.load(basedir+'/orig_model_flux.'+model+'.npy')
        if orig_model_wave.shape[0] != orig_model_flux.shape[1]:
            raise wdfitError('l and f arrays not correct shape in models')
    else:
        orig_model_flux = np.empty([len(model_list),len(orig_model_wave)])
        if model != 'db':
            for i in range(len(model_list)):
                if not quiet: print(i)
                print(basedir+d+model_list[i])
                orig_model_flux[i] = np.loadtxt(basedir+d+model_list[i],usecols=[1])
        else:
            from jg import spectra as _s
            for i in range(len(model_list)):
                if not quiet: print(i)
                tmp = _s.spectra(fn = basedir+d+model_list[i], usecols=[0,1])
                #Not uniform wavelength grid argh!
                tmp.interpolate(orig_model_wave, kind='linear', save_res=True)
                orig_model_flux[i] = tmp.f()
        np.save(basedir+'/orig_model_flux.'+model+'.npy',orig_model_flux)
    #Linearly interpolate Model onto Spectra points
    tck_model = interpolate.interp1d(orig_model_wave,orig_model_flux,kind='linear')
    #Only calculate r model once
    band_lims = _band_limits(band)
    r_model = np.mean(orig_model_flux.transpose()[((orig_model_wave>=band_lims[0])&(orig_model_wave<=band_lims[1]))].transpose(),axis=1)
    #
    print(orig_model_wave)
    print(orig_model_flux)
    return [model_list,model_param,orig_model_wave,orig_model_flux,r_model]


def corr3d(temperature,gravity,ml2a=0.8,testing=False):
    """ Determines the 3D correction (Tremblay et al. 2013, 559, A104)
	  from their atmospheric parameters
    Optional, measure in parsecs
    ML2/alpha = 0.8 is implemented
    """	
    temperature, gravity = float(temperature), float(gravity)
    if temperature > 14500. or temperature < 6000. or gravity > 9.0 or gravity < 7.0:
        #print "no correction"
        return 0.,0.
    if ml2a==0.8:
        print("NEED TO CHANGE PATH AND GET DATA.")
        teff_corr = csv2rec('/storage/astro2/phsmav/data2/models/3dcorr/ml2a_08/teff_corr.csv')
        logg_corr = csv2rec('/storage/astro2/phsmav/data2/models/3dcorr/ml2a_08/logg_corr.csv')
        teff, logg = teff_corr['teff'], teff_corr['logg']
        # indentifying ranges Teff_corr#
        t1,t2 = np.max(teff[teff<=temperature]),np.min(teff[teff>=temperature])
        g1,g2 = np.max(logg[logg<=gravity]),np.min(logg[logg>=gravity])
        if t1!=t2:
            t1,t2 = float(t1), float(t2)
            t = (temperature-t1)/(t2-t1)
        else: t=0.
        if np.isnan(t)==True: t=0.
        if g1!=g2:
            g1,g2 = float(g1), float(g2)
            g = (gravity-g1)/(g2-g1)
        else: g=0
        if np.isnan(g)==True: g=0		
        m11 = teff_corr[logical_and(teff==t1,logg==g1)]['teff_corr'][0]
        m12 = teff_corr[logical_and(teff==t1,logg==g2)]['teff_corr'][0]
        m21 = teff_corr[logical_and(teff==t2,logg==g1)]['teff_corr'][0]
        m22 = teff_corr[logical_and(teff==t2,logg==g2)]['teff_corr'][0]
        teff_3dcorr = (1-t)*(1-g)*m11+t*(1-g)*m21+t*g*m22+(1-t)*g*m12
        
        m11 = logg_corr[logical_and(teff==t1,logg==g1)]['logg_corr'][0]
        m12 = logg_corr[logical_and(teff==t1,logg==g2)]['logg_corr'][0]
        m21 = logg_corr[logical_and(teff==t2,logg==g1)]['logg_corr'][0]
        m22 = logg_corr[logical_and(teff==t2,logg==g2)]['logg_corr'][0]
        logg_3dcorr = (1-t)*(1-g)*m11+t*(1-g)*m21+t*g*m22+(1-t)*g*m12
		
        if testing:	
            plt.fig = figure(figsize=(7,2.5))
            plt.fig.subplots_adjust(left=0.10,bottom=0.15,right=0.98,top=0.98,wspace=0.35)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            tmp_g = [7.0, 7.5, 8.0, 8.5, 9.0]
            tmp_s = ['b-', 'r-', 'g-', 'c-', 'm-']
            axes = [ax1, ax2]
            tmp_t = ['teff_corr', 'logg_corr']
            for i in range(len(axes)):
                for j in range(len(tmp_g)): axes[i].plot(teff[where(logg==tmp_g[j])],teff_corr[tmp_t[i]][where(logg==tmp_g[j])],tmp_s[j],label='logg = %.1f'%(tmp_g[j]),lw=0.5)
                axes[i].legend(numpoints=1,ncol=1,loc=3, fontsize=7)
                axes[i]._set_xlabel('Teff')
                axes[i]._set_ylabel(tmp_t[i])
                axes[i].set_xlim([6000,14500])    
            ax1.plot(temperature,teff_3dcorr,'ro',ms=2)
            ax2.plot(temperature,logg_3dcorr,'ro',ms=2)
            plt.show()
            plt.close()
        return np.round(teff_3dcorr,0),np.round(logg_3dcorr,2)
    elif ml2a==0.8: print("to be implemented")
    elif ml2a==0.7: print("to be implemented")

def interpolating_model_DA(temp,grav,m_type='da2014'):
    """Interpolate model atmospheres given an input Teff and logg
    models are saved as a numpy array to increase speed"""
    # PARAMETERS # 
    dir_models = basedir + '/WDModels_Koester.'+m_type+'_npy/'
    if m_type=="pier" or m_type=="pier_fullres":
        teff=np.array([1500.,1750.,2000.,2250.,2500.,2750.,3000.,3250.,3500.,
                       3750.,4000.,4250.,4500.,4750.,5000.,5250.,5500.,6000.,
                       6500.,7000.,7500.,8000.,8500.,9000.,9500.,10000.,10500.,
                       11000.,11500.,12000.,12500.,13000.,13500.,14000.,14500.,
                       15000.,15500.,16000.,16500.,17000.,20000.,25000.,30000.,
                       35000.,40000.,45000.,50000.,55000.,60000.,65000.,70000.,
                       75000.,80000.,85000.,90000.])
        logg=np.array([6.50,7.00,7.50,7.75,8.00,8.25,8.50,9.00,9.50])
    elif m_type=="pier3D_smooth":	
        teff=np.array([1500.,1750.,2000.,2250.,2500.,2750.,3000.,3250.,3500.,
                       3750.,4000.,4250.,4500.,4750.,5000.,5250.,5500.,6000.,
                       6500.,7000.,7500.,8000.,8500.,9000.,9500.,10000.,10500.,
                       11000.,11500.,12000.,12500.,13000.,13500.,14000.,14500.,
                       15000.,15500.,16000.,16500.,17000.,20000.,25000.,30000.,
                       35000.,40000.,45000.,50000.,55000.,60000.,65000.,70000.,
                       75000.,80000.,85000.,90000.])
        logg=np.array([7.00,7.50,8.00,8.50,9.00])
    elif m_type=="pier_smooth" or m_type=="pier_rad" or m_type=="pier_rad_smooth":	
        teff=np.array([6000.,6500.,7000.,7500.,8000.,8500.,9000.,9500.,10000.,
                       10500.,11000.,11500.,12000.,12500.,13000.,13500.,14000.,
                       14500.,15000.])
        logg=np.array([7.00,7.50,8.00,8.50,9.00])
    elif m_type=="da2014":
        teff=np.array([6000.,6250.,6500.,6750.,7000.,7250.,7500.,7750.,8000.,
                       8250.,8500.,8750.,9000.,9250.,9500.,9750.,10000.,10100.,
                       10200.,10250.,10300.,10400.,10500.,10600.,10700.,10750.,
                       10800.,10900.,11000.,11100.,11200.,11250.,11300.,11400.,
                       11500.,11600.,11700.,11750.,11800.,11900.,12000.,12100.,
                       12200.,12250.,12300.,12400.,12500.,12600.,12700.,12750.,
                       12800.,12900.,13000.,13500.,14000.,14250.,14500.,14750.,
                       15000.,15250.,15500.,15750.,16000.,16250.,16500.,16750.,
                       17000.,17250.,17500.,17750.,18000.,18250.,18500.,18750.,
                       19000.,19250.,19500.,19750.,20000.,21000.,22000.,23000.,
                       24000.,25000.,26000.,27000.,28000.,29000.,30000.,35000.,
                       40000.,45000.,50000.,55000.,60000.,65000.,70000.,75000.,
                       80000.,90000.,100000.])
        logg=np.array([4.00,4.25,4.50,4.75,5.00,5.25,5.50,5.75,6.00,6.25,6.50,
                       6.75,7.00,7.25,7.50,7.75,8.00,8.25,8.50,8.75,9.00,9.25,
                       9.50])
    if (m_type=='pier3D') & (temp<6000. or temp>90000. or grav<6.5 or grav>9.): return [],[]
    elif (m_type=='pier_rad' or m_type=='pier_smooth') & (temp<6000. or temp>15000. or grav<7.0 or grav>9.): return [],[]
    elif (m_type=='da2014') & (temp<6000. or temp>100000. or grav<4.0 or grav>9.5): return [],[]
    # INTERPOLATION #
    g1,g2 = np.max(logg[logg<=grav]),np.min(logg[logg>=grav])
    if g1!=g2: g = (grav-g1)/(g2-g1)
    else: g=0
    t1,t2 = np.max(teff[teff<=temp]),np.min(teff[teff>=temp])
    if t1!=t2: t = (temp-t1)/(t2-t1)          
    else: t=0	
    if m_type =='da2014': models = ['da%06d_%d_2.7.npy'%(i, j*100) for i in [t1,t2] for j in [g1,g2]]
    else: models = ['WD_%.2f_%d.0.npy'%(j, i) for i in [t1,t2] for j in [g1,g2]]
    try:
        m11, m12 = np.load(dir_models+models[0]), np.load(dir_models+models[1])	
        m21, m22 = np.load(dir_models+models[2]), np.load(dir_models+models[3])	
        flux_i = (1-t)*(1-g)*m11[:,1]+t*(1-g)*m21[:,1]+t*g*m22[:,1]+(1-t)*g*m12[:,1]
        return np.dstack((m11[:,0], flux_i))[0]
    except: return [],[]


def fit_line(_sn, l_crop, model_in=None, quick=True, model='sdss'):
    """
    Use norm models - can pass model_in from norm_models() with many spectra
    Input _sn, l_crop <- Normalised spectrum & a cropped line list
    Optional:
        model_in=None   : Given model array
        quick=True      : Use presaved model array. Check is up to date
        model='da2014'  : 'da2014' or 'pier'
    Calc and return chi2, list of arrays of spectra, and scaled models at lines
    """
    from scipy import interpolate
    #load normalised models and linearly interp models onto spectrum wave
    if model_in==None: m_wave,m_flux_n,m_param = norm_models(quick=quick, model=model)
    else: m_wave,m_flux_n,m_param = model_in
    sn_w = _sn[:,0]
    m_flux_n_i = interpolate.interp1d(m_wave,m_flux_n,kind='linear')(sn_w)
    #Crops models and spectra in a line region, renorms models, calculates chi2
    tmp_lines_m, lines_s, l_chi2 = [],[],[]
    for i in range(len(l_crop)):
        l_c0,l_c1 = l_crop[i,0],l_crop[i,1] 
        l_m = m_flux_n_i.transpose()[(sn_w>=l_c0)&(sn_w<=l_c1)].transpose()
        l_s = _sn[(sn_w>=l_c0)&(sn_w<=l_c1)]
        l_m = l_m*np.sum(l_s[:,1])/np.sum(l_m,axis=1).reshape([len(l_m),1])
        l_chi2.append( np.sum(((l_s[:,1]-l_m)/l_s[:,2])**2,axis=1) )
        tmp_lines_m.append(l_m)
        lines_s.append(l_s)
    #mean chi2 over lines and stores best model lines for output
    lines_chi2, lines_m = np.sum(np.array(l_chi2),axis=0), []
    is_best = lines_chi2==lines_chi2.min()
    for i in range(len(l_crop)): lines_m.append(tmp_lines_m[i][is_best][0])
    best_TL = m_param[is_best][0]
    return  lines_s,lines_m,best_TL,m_param,lines_chi2
    

def tmp_func(_T, _g, _rv, _sn, _l, _m):
    from scipy import interpolate
    c = 299792.458 # Speed of light in km/s 
    model=interpolating_model_DA(_T,(_g/100),m_type=_m)
    try: norm_model, m_cont_flux=norm_spectra(model)
    except:
        print("Could not load the model")
        return 1
    else:
        #interpolate normalised model and spectra onto same wavelength scale
        m_wave_n, m_flux_n, sn_w = norm_model[:,0]*(_rv+c)/c, norm_model[:,1], _sn[:,0]
        m_flux_n_i = interpolate.interp1d(m_wave_n,m_flux_n,kind='linear')(sn_w)
        #Initialise: normalised models and spectra in line region, and chi2
        lines_m, lines_s, sum_l_chi2 = [],[],0
        for i in range(len(_l)):
            # Crop model and spec to line
            l_c0,l_c1 = _l[i,0],_l[i,1]
            l_m = m_flux_n_i.transpose()[(sn_w>=l_c0)&(sn_w<=l_c1)].transpose()
            l_s = _sn[(sn_w>=l_c0)&(sn_w<=l_c1)]
            #renormalise models to spectra in line region & calculate chi2+sum
            l_m = l_m*np.sum(l_s[:,1])/np.sum(l_m)
            sum_l_chi2 += np.sum(((l_s[:,1]-l_m)/l_s[:,2])**2)
            lines_m.append(l_m)
            lines_s.append(l_s)
        return lines_s, lines_m, model, sum_l_chi2
        
        
def fit_func(x,specn,lcrop,models='da2014',mode=0):
    """Requires: x - initial guess of T, g, and rv
       specn/lcrop - normalised spectrum / list of cropped lines to fit
       mode=0 is for finding bestfit, mode=1 for fitting & retriving specific model """
    tmp = tmp_func(x[0], x[1], x[2], specn, lcrop, models)
    if tmp == 1: pass
    elif mode==0: return tmp[3] #this is the quantity that gets minimized
    elif mode==1: return tmp[0], tmp[1], tmp[2]


def err_func(x,rv,valore,specn,lcrop,models='da2014'):
    """Script finds errors by minimising function at chi+1 rather than chi
       Requires: x; rv - initial guess of T, g; rv
       valore - the chi value of the best fit
       specn/lcrop - normalised spectrum / list of cropped lines to fit"""
    tmp = tmp_func(x[0], x[1], rv, specn, lcrop, models)
    if tmp != 1: return abs(tmp[3]-(valore+1.)) #this is quantity that gets minimized 
    else: pass
