# irx_tau_tot
[![arXiv](https://img.shields.io/badge/arxiv.org/abs/2312.16700-blue)](http://arxiv.org/abs/2312.16700)

## Introduction
irx_tau_tot is a package including tauv_model function to provide a numerical calculation code for equation 13 of Qin+ 2024. Users can obtain the corresponding optical depth by inputting galaxy parameters, including k, SigDust, Fbc, R, H, tau_bc, Cbc.

## Usage
You can download the entire package and run it.
```
tau_v = tauv_model(logsigma_mdust=logsigma_mdust_arr, fbc=fbc, Rhat=Rhat, Hhat=Hhat, tau_bc=tau_bc, Cbc=Cbc, kv=0.62e-5)
tau_fuv = tau_v * 2.5
IRX = (np.exp(tau_fuv) - 1) / 0.46
```
