#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 22:01:00 2023

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
from importlib import reload
#%config InlineBackend.figure_format = 'retina'
from numpy import *
default_color=['C0','C1','C2','C3']
import matplotlib

#因python是全局变量，所以这里的fRs等参数最好设置为独一无二或者放到函数里面去，避免后面串扰。
###################
logsigma_mdust_arr=arange(0,10,0.1)
fig,axs= plt.subplots(1,4,figsize=(12.5,2.5),constrained_layout=1)


ax=axs.flat[0]
ax.set(xlim=(3,9),ylim=(-2,6),xlabel=r'$\mathrm{\log\Sigma_{dust}^{tot}}$',ylabel=r'$\mathrm{\log IRX}$')#,title=r'$f_{R_s}$=$C_{bc}$=$f_{Z_s}$=0.5')
fbc=0.5; Cbc=0.5; tau_bc=0.5*2.5; fRs=0.5; fZs=0.5
for i,fbc in enumerate([0.5,0.9]):
    ax.plot(logsigma_mdust_arr,qin.irx_model_withfZs(logsigma_mdust=logsigma_mdust_arr,fbc=fbc,fRs=fRs,fZs=fZs,tau_bc=0,Cbc=Cbc),'--',color='C'+str(i))#,label=r'$f_{bc}=%.1f$'%fbc)
    ax.plot(logsigma_mdust_arr,qin.irx_model_withfZs(logsigma_mdust=logsigma_mdust_arr,fbc=fbc,fRs=fRs,fZs=fZs,tau_bc=tau_bc,Cbc=Cbc),'-',color='C'+str(i),label=r'$F_{bc}=%.1f$'%fbc)
ax.text(3.1,5.4,'(a)')
ax.text(3.1,1,'diffuse+BC')
ax.text(4.3,-1.5,'diffuse')
ax.legend(loc=(0.18,0.7))

ax=axs.flat[1]
ax.set(xlim=(3,9),ylim=(-2,6),xlabel=r'$\mathrm{\log\Sigma_{dust}^{tot}}$',ylabel='',yticklabels='')#,title=r'$f_{R_s}$=$f_{bc}$=$f_{Z_s}$=0.5')
fbc=0.5; Cbc=0.5; tau_bc=0.5*2.5; fRs=0.5; fZs=0.5
for i,Cbc in enumerate([0.5,0.9]):
    ax.plot(logsigma_mdust_arr,qin.irx_model_withfZs(logsigma_mdust=logsigma_mdust_arr,fbc=fbc,fRs=fRs,fZs=fZs,tau_bc=0,Cbc=Cbc),'--',color='C'+str(i))#,label=r'$f_{fill}=%.1f$'%ff)
    ax.plot(logsigma_mdust_arr,qin.irx_model_withfZs(logsigma_mdust=logsigma_mdust_arr,fbc=fbc,fRs=fRs,fZs=fZs,tau_bc=tau_bc,Cbc=Cbc),'-',color='C'+str(i),label=r'$C_{bc}=%.1f$'%Cbc)
    #if i==1:ax.plot(logsigma_mdust_arr,irx_model(1,fbc,0.5,fRs,logsigma_mdust_arr),':',color='C'+str(i),label=r'$F=0.5,f_{bc}=%.1f, \tau_{bc}=1$'%fbc)
    #ax.plot(logsigma_mdust_arr,irx_model(0.5,fbc,fRs,logsigma_mdust_arr),':',color='C'+str(i),label=r'$f_{bc}=%.1f, \tau_{bc}=0.5$'%fbc)
#ax.plot(logsigma_mdust_arr,1*logsigma_mdust_arr-5,':',c='gray',label=r'$IRX\propto\Sigma_{dust}$')
ax.legend()
ax.text(3.1,5.4,'(b)')
ax.text(3.1,0.7,'diffuse+BC')
ax.text(4.3,-1.5,'diffuse')
ax.legend(loc=(0.18,0.7))

ax=axs.flat[2]
ax.set(xlim=(3,9),ylim=(-2,6),xlabel=r'$\mathrm{\log\Sigma_{dust}^{tot}}$',ylabel='',yticklabels='')#,title=r'$f_{bc}$=$C_{bc}$=$f_{R_s}$=0.5')
fbc=0.5; Cbc=0.5; tau_bc=0.5*2.5; fRs=0.5; fZs=0.5
for i,fRs in enumerate([0.5,1]):
    ax.plot(logsigma_mdust_arr,qin.irx_model_withfZs(logsigma_mdust=logsigma_mdust_arr,fbc=fbc,fRs=fRs,fZs=fZs,tau_bc=0,Cbc=Cbc),'--',color='C'+str(i))#,label=r'$f_{R_s}=%.1f$'%fRs)
    ax.plot(logsigma_mdust_arr,qin.irx_model_withfZs(logsigma_mdust=logsigma_mdust_arr,fbc=fbc,fRs=fRs,fZs=fZs,tau_bc=tau_bc,Cbc=Cbc),'-',color='C'+str(i),label=r'$\hat{R}=%.1f$'%fRs)
#ax.plot(logsigma_mdust_arr,1*logsigma_mdust_arr-5,':',c='gray',label=r'$IRX\propto\Sigma_{dust}$')
ax.text(3.1,5.4,'(c)')
ax.text(3.1,0.7,'diffuse+BC')
ax.text(4.3,-1.5,'diffuse')
ax.legend(loc=(0.18,0.7))

ax=axs.flat[3]
ax.set(xlim=(3,9),ylim=(-2,6),xlabel=r'$\mathrm{\log\Sigma_{dust}^{tot}}$',ylabel='',yticklabels='')#,title=r'$f_{bc}$=$C_{bc}$=$f_{R_s}$=0.5')
fbc=0.5; Cbc=0.5; tau_bc=0.5*2.5; fRs=0.5; fZs=0.5
for i,fZs in enumerate([0.5,1]):
    ax.plot(logsigma_mdust_arr,qin.irx_model_withfZs(logsigma_mdust=logsigma_mdust_arr,fbc=fbc,fRs=fRs,fZs=fZs,tau_bc=0,Cbc=Cbc),'--',color='C'+str(i))#,label=r'$f_{R_s}=%.1f$'%fRs)
    ax.plot(logsigma_mdust_arr,qin.irx_model_withfZs(logsigma_mdust=logsigma_mdust_arr,fbc=fbc,fRs=fRs,fZs=fZs,tau_bc=tau_bc,Cbc=Cbc),'-',color='C'+str(i),label=r'$\hat{H}=%.1f$'%fZs)
#ax.plot(logsigma_mdust_arr,1*logsigma_mdust_arr-5,':',c='gray',label=r'$IRX\propto\Sigma_{dust}$')
ax.text(3.1,5.4,'(d)')
ax.text(3.1,0.7,'diffuse+BC')
ax.text(4.3,-1.5,'diffuse')
ax.legend(loc=(0.18,0.7))
# 刻度朝内
axes = fig.get_axes()
for ax in axes:
    ax.tick_params(direction='in')
plt.savefig("irx_model_relations.pdf")
