
#return where x in range of [a,b]
from cmath import exp

from numpy import Infinity


def wherein(x,rangea,rangeb):
    import numpy as np
    y=np.where((x >= rangea) & (x < rangeb))
    return y

# return Spearman's rank coefficient    
def cor(x,y):
    from scipy.stats import spearmanr 
    a=spearmanr(x,y)
    return a.correlation

#The formula of universal IRX relation in Qin+2019a
def uir_formula(logzoh,loglir,logr50,logar,p):
    y=p[0]+p[1]*(logzoh-8.69)+(p[2]+p[3]*(logzoh-8.69))*(loglir-10)\
        -1*(p[4]+p[5]*(logzoh-8.69))*logr50 -(p[6]+p[7]*(logzoh-8.69))*logar
    return y

def uir_formula_cor(logzoh,loglir,logr50,logar,p):
    import numpy as np
    alpha=p[2]+p[3]*(logzoh-8.69)
    beta=p[4]+p[5]*(logzoh-8.69)
    gamma=p[6]+p[7]*(logzoh-8.69)
    a=np.where(alpha<0) 
    if np.size(a)>0:alpha[a]=0
    a=np.where(beta<0)
    if np.size(a)>0:beta[a]=0
    a=np.where(gamma<0)
    if np.size(a)>0:gamma[a]=0
    y=p[0]+p[1]*(logzoh-8.69)+(alpha)*(loglir-10)\
        -1*(beta)*logr50 -(gamma)*logar
    return y

def uir_formula_withmass(logmstar,logzoh,loglir,logr50,logar,p):
    y=p[0]+p[1]*(logzoh-8.69)+(p[2]+p[3]*(logzoh-8.69))*(loglir-10)\
        -1*(p[4]+p[5]*(logzoh-8.69))*logr50 -(p[6]+p[7]*(logzoh-8.69))*logar\
             +(p[8]+p[9]*(logzoh-8.69))*(logmstar-10)
    return y

def uir_formula_usesfr(logzoh,logsfr,logr50,logar,p):
    y=p[0]+p[1]*(logzoh-8.69)+(p[2]+p[3]*(logzoh-8.69))*(logsfr)\
        -1*(p[4]+p[5]*(logzoh-8.69))*logr50 -(p[6]+p[7]*(logzoh-8.69))*logar
    return y

def sersic_fun(I0,r,rs,n):
    import numpy as np
    Ir=I0*np.exp(-(np.abs(r)/rs)**(1/n))
    return Ir

def sersic_fun_re(Ie,r,re,n):
    import numpy as np
    bn=2*n-1/3.0
    Ir=Ie*np.exp(-bn*((np.abs(r)/re)**(1/n)-1))
    return Ir

def alam_exp_disk_with_bc(aRs,tau_bc,fbc,fRs,fH0,sigma_mdust,klam):
    from scipy import integrate
    import numpy as np
    sigma_mdust_rs=sigma_mdust*2.82 # convert sigma_mdust_re to sigma_mdust_rs
    f = lambda Y,X: \
        0.5/(1-(1+aRs)*np.exp(-aRs))*X* \
        np.exp(- klam*(1-fbc)*sigma_mdust_rs/4/(1-(1+fRs*aRs)*np.exp(-fRs*aRs)) \
        *(np.exp(-fRs*X)-fH0*Y))
    res=integrate.dblquad(f, 0, aRs, lambda X: -np.exp(-X), lambda X: np.exp(-X))
    return (-np.log(res[0])*1.086)+tau_bc

def alam_exp_disk(aRs,fRs,fH0,sigma_mdust,klam):
    from scipy import integrate
    import numpy as np
    sigma_mdust_rs=sigma_mdust*2.82 # convert sigma_mdust_re to sigma_mdust_rs
    f = lambda Y,X: \
        0.5/(1-(1+aRs)*np.exp(-aRs))*X* \
        np.exp(- klam*(1-0)*sigma_mdust_rs/4/(1-(1+fRs*aRs)*np.exp(-fRs*aRs)) \
        *(np.exp(-fRs*X)-fH0*Y))
    res=integrate.dblquad(f, 0, aRs, lambda X: -np.exp(-X), lambda X: np.exp(-X))
    return (-np.log(res[0])*1.086)

# This is used to 
def alam_2exp_disk(fRs,fHs,sigma_mdust,klam):
    from scipy import integrate
    import numpy as np
    sigma_mdust_rs=sigma_mdust*2.82 # convert sigma_mdust_re to sigma_mdust_rs
    # two part function
    f=lambda Y,X: np.piecewise(X, [Y < 0, Y >= 0], [0.5*X*np.exp(-klam*sigma_mdust_rs/4*np.exp(-fRs*X)*(2-np.exp(fHs*Y))-X+Y),  0.5*X*np.exp(-klam*sigma_mdust_rs/4*np.exp(-fRs*X-fHs*Y)-X-Y)])
    int=integrate.dblquad(f, 0, 20, -20,20)
    return (-np.log(int[0])*1.086)

def alam_2exp_disk_nint(fRs,fHs,sigma_mdust,klam):
    #与上面不同的是，这里不用numpy自带的积分，而是自己分网格进行数值积分，可以省点时间，计算结果二者差不多。仅在计算量很大很耗时时使用（例如thickness_effect_incl.ipynb）,其它程序都用默认的积分方法。
    from scipy import integrate
    import numpy as np
    sigma_mdust_rs=sigma_mdust*2.82 # convert sigma_mdust_re to sigma_mdust_rs
    # two part function
    f=lambda X,Y: np.piecewise(X, [Y < 0, Y >= 0], [0.5*X*np.exp(-klam*sigma_mdust_rs/4*np.exp(-fRs*X)*(2-np.exp(fHs*Y))-X+Y),  0.5*X*np.exp(-klam*sigma_mdust_rs/4*np.exp(-fRs*X-fHs*Y)-X-Y)])
    int=integrate2d(f, 0, 10, -10,10, 0.2,0.2)
    return (-np.log(int)*1.086)

#====================================
def sigmadustdiff2afuv_withfZs(logsigma_mdust,fbc,fRs,fZs):
    #Require: tau_diff_ngrid.ipynb
    import numpy as np
    from scipy import interpolate
    maxarray=logsigma_mdust*0+fbc*0+fRs*0+fZs*0
    logsigma_mdust,fbc,fRs,fZs=maxarray+logsigma_mdust,maxarray+fbc,maxarray+fRs,maxarray+fZs,
    cat=np.load('./tau_diff_4paras.npz')
    res=interpolate.interpn((cat['logsigma_mdust_arr'],cat['fbc_arr'],cat['fRs_arr'],cat['fZs_arr']),cat['afuv_mod_grid'],np.transpose([logsigma_mdust,fbc,fRs,fZs]),bounds_error=False, fill_value=0,method='linear')
    return res

def irx_model_withfZs(logsigma_mdust,fbc,fRs,fZs,tau_bc,Cbc,klam=0.67*2.5/1e5):
    #Require: tau_diff_ngrid.ipynb
    import numpy as np
    afuv= -np.log(1-Cbc+Cbc*np.exp(-tau_bc))*1.086+sigmadustdiff2afuv_withfZs(logsigma_mdust,fbc,fRs,fZs)*klam/(0.67*2.5/1e5)
    irx=afuv2irx(afuv)
    return irx


#====================================================
def sigmadustdiff2afuv(logsigma_mdust,fbc,fRs):
    import numpy as np
    from scipy import interpolate
    maxarray=logsigma_mdust*0+fbc*0+fRs*0
    cat=np.load('./tau_diff_3paras.npz')
    logsigma_mdust,fbc,fRs=maxarray+logsigma_mdust,maxarray+fbc,maxarray+fRs
    res=interpolate.interpn((cat['logsigma_mdust_arr'],cat['fbc_arr'],cat['fRs_arr']),cat['afuv_mod_grid'],np.transpose([logsigma_mdust,fbc,fRs]),bounds_error=False, fill_value=0,method='linear')
    return res

def irx_model(logsigma_mdust,fbc,fRs,tau_bc,Cbc,klam=0.67*2.5/1e5):
    #主程序
    import numpy as np
    afuv= -np.log(1-Cbc+Cbc*np.exp(-tau_bc))*1.086+sigmadustdiff2afuv(logsigma_mdust,fbc,fRs)*klam/(0.67*2.5/1e5)
    irx=afuv2irx(afuv)
    return irx
#============================================
def irx_model_overlap(logsigma_mdust,fbc,fRs,tau_bc,Cbc,fol):
    #Require: tau_diff.ipynb
    import numpy as np
    logsigma_mdust_bc=logsigma_mdust+np.log10(fbc)
    afuv= -np.log(1-Cbc+Cbc*np.exp(-tau_bc*(1+10**((logsigma_mdust_bc-5)-fol))))*1.086+sigmadustdiff2afuv(logsigma_mdust,fbc,fRs)#*klam/(0.67*2.5/1e5)
    irx=afuv2irx(afuv)
    return irx

def afuv_model(logsigma_mdust,fbc,fRs,tau_bc,Cbc):
    #Require: tau_diff.ipynb
    import numpy as np
    afuv= -np.log(1-Cbc+Cbc*np.exp(-tau_bc))*1.086+sigmadustdiff2afuv(logsigma_mdust,fbc,fRs)#*klam/(0.67*2.5/1e5)
    return afuv

def afuv2irx_b05(afuv):# according to Buat+2005
    import numpy as np
    from scipy import interpolate
    irxt=np.arange(-1,10,0.1)
    afuvt=(-0.0333*irxt**3 + 0.3522*irxt**2 + 1.1960*irxt + 0.4967)
    f = interpolate.interp1d(afuvt, irxt, bounds_error=False, fill_value=0)
    irx_mod=f(afuv)
    return irx_mod

def irx2afuv_b05(irx):# according to Buat+2005
    afuv=(-0.0333*irx**3 + 0.3522*irx**2 + 1.1960*irx + 0.4967)
    return afuv


def irx2afuv(irx): # according to Hao+2011
    import numpy as np
    afuv=2.5*np.log10(1+0.48*10**irx)
    return afuv

def afuv2irx(afuv):# according to Hao+2011
    import numpy as np
    from scipy import interpolate
    irxt=np.arange(-19,19,0.05)
    afuvt=2.5*np.log10(1+0.48*10**irxt)
    f = interpolate.interp1d(afuvt, irxt, bounds_error=False, fill_value=19)
    irx_mod=f(afuv)
    return irx_mod



def fun_Rs_t2d(fbc,fRs):
    #APPENDIX A: DETERMINING THE NUMERICAL SOLUTION of fRs_tot(fRs,fbc)
    import numpy as np
    from scipy.optimize import curve_fit
    def exp_profile(r, rho0,Rs):
        return rho0*np.exp(-r/Rs)
    ######################
    xdata=np.arange(0.01,5,0.01)
    Rs_t2d=[]
    if np.ndim(fbc)==0:
        fbc_arr=np.array([fbc]); fRs_arr=np.array([fRs])
    else:
        fbc_arr=np.array(fbc); fRs_arr=np.array(fRs)
    for i,ifbc in enumerate(fbc_arr):
        ydata=(1-fbc_arr[i])*np.exp(-xdata)+fbc_arr[i]/fRs_arr[i]*np.exp(-xdata/fRs_arr[i])
        a=np.where(ydata >= ydata[0]*np.exp(-3)) # only fit 3Rs
        paras, covariance = curve_fit(exp_profile, xdata[a], ydata[a])
        fit_ydata = exp_profile(xdata, *paras)
        Rs_t2d.append(paras[1])
    if np.size(Rs_t2d)==1 : Rs_t2d=Rs_t2d[0]
    return Rs_t2d


def str_arr(num_arr):
    return [str(x) for x in num_arr]


def linfit_fun(x,p0,p1):
    return p0+x*p1

def linfit(x,y):
    from scipy.optimize import curve_fit
    import numpy as np
    res=curve_fit(linfit_fun,x,y)[0]
    return res

def poly_fun(x,p):
    import numpy as np
    if np.ndim(x)==0:
        x=[x]; p=[p]
    x=np.array(x)
    p=np.array(p)
    y=x*0
    for i,ix in enumerate(x):y[i]=np.sum(p*x[i]**np.arange(0,p.size))
    return y

def cor(x,y):
    from scipy.stats import spearmanr 
    a=spearmanr(x,y)
    return a.correlation

def lumdist(cosmo,redshift):
    y=cosmo.luminosity_distance(redshift).cgs.value 
    return y


def tau_xyz(x,y,z,incl,Hs,Rs,klam,sigma_mdust):
    import numpy as np
    from  scipy import integrate
    if incl<np.deg2rad(89): # 小于90度都成立
        f = lambda zp : klam*sigma_mdust/(4*Hs*np.cos(incl))*np.exp(-np.sqrt(x**2+((zp-z)*np.tan(incl)+y)**2)/Rs-np.abs(zp)/Hs)
        res=integrate1d(f, z, 10*Hs,0.2*Hs)
        #res=integrate.quad(f,z,10*Hs)[0]
    else:
        f = lambda yp :  klam*sigma_mdust/(4*Hs*np.sin(incl))*np.exp(-np.sqrt(x**2+yp**2)/Rs-np.abs(z+(yp-y)/np.tan(incl))/Hs)
        res=integrate1d(f, y, 10*Rs,0.2*Rs)
    return res

def tau_xyz_spin(x,y,z,incl,Hs,Rs,klam,sigma_mdust): #( caution!!: intergrate1d have bugs at lower incl)
    from  scipy import integrate
    import numpy as np
    f = lambda zp : klam*sigma_mdust/(4*Hs)*np.exp(-np.sqrt(x**2+(y*np.cos(incl)-zp*np.sin(incl))**2)/Rs-np.abs(zp*np.cos(incl)+y*np.sin(incl))/Hs)
    if incl<=np.deg2rad(45):  res=integrate1d(f, z, 10*Hs/np.cos(incl),0.2*Hs/np.cos(incl))
    else:                     res=integrate1d(f, z, 10*Rs/np.sin(incl),0.2*Rs/np.sin(incl))
        #res=integrate.quad(f, z, 5*Hs/np.cos(incl))[0]
    return res
    
def alam_2exp_disk_incl(incl,Hs,Rs,klam,sigma_mdust,Hsl,Rsl):
    #仅用于thickness_effect_incl.ipynb 这里就是用到自己编的积分，因为计算量比较大能省点时间（可能由于积分步长存在一点精度问题，影响不大）
    import numpy as np
    sigma_mdust_rs=sigma_mdust*2.82 # convert sigma_mdust_re to sigma_mdust_rs
    # 只积了x>0半部分(yz全部)，两部分对称消光一样
    f=lambda x,y,z: 1/(2*np.pi*Hsl*Rsl**2)*np.exp(-np.sqrt(x**2+y**2)/Rsl-np.abs(z)/Hsl \
          -tau_xyz(x=x,y=y,z=z,incl=incl,Hs=Hs,Rs=Rs,klam=klam,sigma_mdust=sigma_mdust_rs))
    int=integrate3d(f, 0, 10*Rsl, -10*Rsl,10*Rsl,-10*Hsl,10*Hsl,0.2*Rsl,0.2*Rsl,0.2*Hsl)

    return (-np.log(int)*1.086)


##########################################
# Integrate:
# Oct 27th, 2022 
#Auther: Jianbo Qin
# required: numpy 
#scipy.integrate use relative intergrate step(i.e., dx), and can not be ajusted by the user. I write the simple numerical intergral function, and sucessfully applied to the optical depth calculation in exp-disk geometry.   
def integrate1d(f,x1,x2,dx):#,y1,y2,z1,z2,dx,dy,dz):
    import numpy as np
    x_arr=np.arange(x1,x2,dx)
    f_arr=f(x_arr+0.5*dx)
    result=np.sum(f_arr*dx)
    return result

def integrate2d(f,x1,x2,y1,y2,dx,dy):#,y1,y2,z1,z2,dx,dy,dz):
    import numpy as np
    x_arr=np.arange(x1,x2,dx)
    y_arr=np.arange(y1,y2,dy) 
    result=0.0
    for ix,x in enumerate(x_arr):
        for iy,y in enumerate(y_arr):
            result=result+f(x+0.5*dx,y+0.5*dy)*dx*dy
    return result

def integrate3d(f,x1,x2,y1,y2,z1,z2,dx,dy,dz):#,y1,y2,z1,z2,dx,dy,dz):
    import numpy as np
    x_arr=np.arange(x1,x2,dx)#ax.semilogx()
#ax.set(xlabel=r'$\lambda/\mu m$',ylabel=('$L_{\lambda}$/arbitrary units'),xlim=(0.05,1),ylim=(1e21,1e28))
    y_arr=np.arange(y1,y2,dy) 
    z_arr=np.arange(z1,z2,dz)
    result=0.0
    for ix,x in enumerate(x_arr):
        for iy,y in enumerate(y_arr):
            for iz,z in enumerate(z_arr):
                result=result+f(x+0.5*dx,y+0.5*dy,z+0.5*dz)*dx*dy*dz
    return result
#######################################################################

def rho_xyz(x,y,z,incl,Rs,fs):
    import numpy as np
    Hs=fs*Rs
    rho=np.exp(-np.sqrt(x**2+(y*np.cos(incl)-z*np.sin(incl))**2)/Rs-np.sqrt((z*np.cos(incl)+y*np.sin(incl))**2)/Hs)
    #rho=np.exp(-np.sqrt(x**2+(y*np.cos(incl)-z*np.sin(incl))**2+((z*np.cos(incl)+y*np.sin(incl))/fs)**2)/Rs)
    return rho


# ar_proj
# Return projected b/a for a given fs=Hs/Rs and inclination
# for a galaxy density profile of rho~exp(-r/Rs-|h|/Hs)
def ar_proj(fs,incl,map=0): #default map=0: return ar; set map=1: return ar,density_map=
    from photutils import morphology as mor
    import numpy as np
    Rs=1
    n=10
    nbin=100
    Hs=Rs*fs
    x_arr=np.linspace(-n*Rs,n*Rs, nbin) 
    y_arr=np.linspace(-n*Rs,n*Rs, nbin) 
    z_arr=np.linspace(-n*Rs,n*Rs, nbin) 
    density_xy,temp=np.meshgrid(y_arr*0,x_arr*0)
    for i,x in enumerate(x_arr): 
        for j,y in enumerate(y_arr):
            density_xy[j,i]=np.sum(rho_xyz(x=x,y=y,z=z_arr,Rs=Rs,fs=fs,incl=incl))
    #################################################
    
    cat = mor.data_properties(density_xy) 
    columns = ['label', 'xcentroid', 'ycentroid', 'semimajor_sigma',
            'semiminor_sigma', 'orientation']
    tbl = cat.to_table(columns=columns)
    tbl['xcentroid'].info.format = '.10f'  # optional format
    tbl['ycentroid'].info.format = '.10f'
    tbl['semiminor_sigma'].info.format = '.10f'
    tbl['orientation'].info.format = '.10f'
    ar=tbl['semiminor_sigma'].value/tbl['semimajor_sigma'].value
    if map==0: return ar[0]
    else:return ar[0],density_xy
