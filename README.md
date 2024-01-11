# IRX_tau_tot
[![arXiv](https://img.shields.io/badge/arxiv.org/abs/2312.16700-blue)](http://arxiv.org/abs/2312.16700)

## Introduction
Calculate_tau_tot is a package to provide a numerical calculation code for equation 13 of Qin+ 2024. Users can obtain the corresponding optical depth by inputting galaxy parameters, including k, SigDust, Fbc, R, H, tau_bc, Cbc.
This package includes two versions for calculating optical depth: the version with three parameters sets R=H, and the version with four parameters allows R and H to be independent.

## Usage
You can download the entire package and run it.
```
cd path/Calculate_tau_tot
python3 tau_diff_3paras.py
python3 tau_diff_4paras.py
```
