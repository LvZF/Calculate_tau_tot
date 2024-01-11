from qin import *
import numpy as np

def  logr50_vdw14(logmstar,z):
    from scipy import interpolate
    A=np.array([0.86,0.78,0.70,0.65,0.55,0.51])
    B=np.array([0.25,0.22,0.22,0.23,0.22,0.18])
    hy=np.array([0.25,0.75,1.25,1.75,2.2,2.75])
    if np.ndim(logmstar)==0: 
        logmstar=[logmstar]
        z=[z]
    logmstar=np.array(logmstar)
    z=np.array(z)
    reff=logmstar*0-99.0
    for k,km in enumerate(logmstar): 
        reffz=A+B*(km-10.7)
        fext=interpolate.UnivariateSpline(hy,reffz)
        reff[k]=fext(z[k])
    return reff

def sfr_s14(logmstar,z,cosmo=0):
    if cosmo==0:
        cosmo=flat_cosmology()
    t=cosmo.age(z).value
    logsfr=(0.84-0.026*t )*logmstar-(6.51-0.11*t)
    return logsfr
def sfr_p22(logmstar,z,cosmo=0,poly_fun=0): #default flat_cosmology and Lee+2015 formula
    if cosmo==0:
        cosmo=flat_cosmology()
    t=cosmo.age(z).value
    if poly_fun ==0:
        a0,a1,a2,a3,a4=2.69,-0.18,10.85,-0.073,0.99
        logsfr=a0+a1*t-np.log10(1+(10**(logmstar-(a2+a3*t)))**(-a4))
    else:
        a0,a1,b0,b1,b2=0.20,-0.033,-26.13,4.72,-0.19
        alpha=a0+a1*logmstar
        beta=b0+b1*logmstar+b2*(logmstar)**2
        logsfr=alpha*t+beta
    return logsfr

def zgas_g15(logmstar,z):
    a=8.74
    poly_fun(np.log10(1+z),[10.4,4.46,-1.78])
    zgas=a-0.087*(logmstar-b)**2.0
    sigma_b=np.sqrt( 0.05**2+(0.3*np.log10(1.+z))**2+(0.4*np.log(1.+z)**2)**2)
    zgas_err=np.sqrt(0.06**2+(2*0.087*(logmstar-b)*sigma_b)**2.)
    return zgas

def zgas_s23(logmstar,z,pp04=0):
    if np.size(logmstar)>1: logmstar,z=np.array(logmstar),np.array(z)
    z0,gamma,beta,m0,m1=8.8,0.30,1.08,9.90,2.06
    logm0=m0+m1*np.log10(1+z)
    zgas=z0-gamma/beta*(np.log10(1+10**(-beta*(logmstar-logm0))))
    if pp04 ==0: return zgas 
    else: return Curti20_n2_to_pp04_n2(zgas)

def flat_cosmology():
    from astropy.cosmology import  FlatLambdaCDM 
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3) 
    return cosmo



def Curti2020_n2_calibration(logR):
    from scipy import interpolate
    resR=np.flipud([-0.489,1.513,-2.554,-5.293,-2.867]) 
    x_arr=np.arange(7.6,8.85,0.001)-8.69
    logR_arr=-0.489+1.513*x_arr-2.554*x_arr**2-5.293*x_arr**3-2.867*x_arr**4#np.polyval(resR,x_arr)
    f = interpolate.interp1d(logR_arr, x_arr, bounds_error=False, fill_value=8.9-8.69)
    zgas=f(logR)+8.69
    return zgas


def Curti20_n2_to_pp04_n2(logzoh):
    x_arr=logzoh-8.69
    logR_arr=-0.489+1.513*x_arr-2.554*x_arr**2-5.293*x_arr**3-2.867*x_arr**4#np.polyval(resR,x_arr)
    res_pp04=np.flipud([9.37,2.03,1.26,0.32]) #defined an poly_fun
    logzoh_pp04=np.polyval(res_pp04, logR_arr)
    return logzoh_pp04

def zgas_fmr_sanders21(logmstar,logsfr):
    y=logmstar-0.6*logsfr-10
    zgas=8.80+0.188*y-0.220*y**2-0.0531*y**3
    return zgas