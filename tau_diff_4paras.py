#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 21:38:20 2023

@author: lzf
"""

#! /usr/bin/env python3 ####################################################################################
#import moduels 
import scipy as sp
import matplotlib.pyplot as  plt
from  scipy import io
from astropy.io import fits
from astropy import constants as const
import copy 
from astropy.cosmology import  FlatLambdaCDM 
cosmo = FlatLambdaCDM(H0=70, Om0=0.3) 
import qin,derive
from importlib import reload ; reload(qin); reload(derive)
#%config InlineBackend.figure_format = 'retina'
from numpy import *
from matplotlib.font_manager import FontProperties
from scipy.optimize import curve_fit

#设置不同的FRs,FZs，耗时很大，主要用来画文章中的图1d。不用来做具体拟合



#! /usr/bin/env python3 ####################################################################################
def fun_Rs_t2d(fbc,fRs):
    import numpy as np
    from scipy.optimize import curve_fit
    def exp_profile(r, Rs):
        return Rs*np.exp(-r*Rs)
    ######################
    xdata=np.arange(0.01,5,0.01)
    Rs_t2d=[]
    fbc_arr=np.array([fbc]); fRs_arr=np.array([fRs])
    for i,ifbc in enumerate(fbc_arr):
        ydata=fbc_arr[i]*np.exp(-xdata)+(1-fbc_arr[i])*fRs_arr[i]*np.exp(-xdata*fRs_arr[i])
        a=np.where(ydata >= ydata[0]*np.exp(-3)) # only fit 3Rs
        paras, covariance = curve_fit(exp_profile, xdata[a], ydata[a])
        fit_ydata = exp_profile(xdata, *paras)
        Rs_t2d.append(paras)
    if np.size(Rs_t2d)==1 : Rs_t2d=Rs_t2d[0]
    return Rs_t2d

def alam_2exp_disk(fRs,fZs,sigma_mdust,klam):
    from scipy import integrate
    import numpy as np
    sigma_mdust_rs=sigma_mdust*2.82 # convert sigma_mdust_re to sigma_mdust_rs
    # two part function
    f=lambda Y,X: np.piecewise(X, [Y < 0, Y >= 0], [0.5*X*np.exp(-klam*sigma_mdust_rs/4*np.exp(-fRs*X)*(2-np.exp(fZs*Y))-X+Y),  0.5*X*np.exp(-klam*sigma_mdust_rs/4*np.exp(-fRs*X-fZs*Y)-X-Y)])
    int=integrate.dblquad(f, 0, 20, -20,20)
    return (-np.log(int[0])*1.086)

from numpy import *
kfuv=0.67*2.5/1e5 # Kfuv:Av per solar mass of dust
logsigma_mdust_arr=linspace(0,10,1) 
fbc_arr=linspace(0,1,10)
fRs_arr=linspace(0.1,1.1,10)
fZs_arr=linspace(0.1,1.1,10)

logsigma_mdust_grid,fbc_grid,fRs_grid,fZs_grid=meshgrid(logsigma_mdust_arr,fbc_arr,fRs_arr,fZs_arr,indexing='ij') 
logsigma_mdust_diff_grid=logsigma_mdust_grid*0
for i,fbc in enumerate(fbc_arr):
    for j,fRs in enumerate(fRs_arr):
        fcor=2*log10(fun_Rs_t2d(fbc,fRs))
        for k,fZs in enumerate(fZs_arr):
            logsigma_mdust_diff_grid[:,i,j,k]=log10(1-fbc)+logsigma_mdust_grid[:,i,j,k]+fcor

afuv_mod_grid=fRs_grid*0
klam=kfuv
from multiprocessing import Process,Pool
for i,fbc in enumerate(fbc_arr):
    for j,fRs in enumerate(fRs_arr):
        for k,fZs in enumerate(fZs_arr):
            if __name__ == "__main__":
                pool = Pool(processes=4)
                result = []  # 保存进程
                for l,logsigma_mdust in enumerate(logsigma_mdust_arr):
                    sigma_mdust_diff=10**(logsigma_mdust_diff_grid[l,i,j,k])
                    result.append(pool.apply_async(alam_2exp_disk, args=(fRs,fZs,sigma_mdust_diff,klam)))  # 维持执行的进程总数为10，当一个进程执行完后添加新进程.
                pool.close()
                pool.join()
                # 此时所有子进程已经执行完毕
                for l,res in enumerate(result):
                    afuv_mod_grid[l,i,j,k]=res.get() # 保存进程执行结果到array里
savez('tau_diff_4paras.npz',logsigma_mdust_arr=logsigma_mdust_arr,fRs_arr=fRs_arr,fZs_arr=fZs_arr,fbc_arr=fbc_arr,afuv_mod_grid=afuv_mod_grid)
