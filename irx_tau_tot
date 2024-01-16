### See in Qin et al. 2024 eq13
### Require:tau_diff_4paras.npz

import numpy as np
from scipy import interpolate

def tauv_model(logsigma_mdust, fbc, Rhat, Hhat, tau_bc, Cbc, kv=0.62e-5):
    """
    Parameters
    ----------
    logsigma_mdust : float or array
        Total dust surface density, in unit of M_sun/kpc^2 in logarithm.
    fbc : float or array
        BC-to-total dust mass fraction.
    Rhat : float or array
        Star-to-diffuse dust disc scale length ratio [Rs_star/Rs_diffdust].
    Hhat : float or array
        Star-to-diffuse dust disc scale Height ratio [Hs_star/Hs_diffdust].
    tau_bc : float or array
        BC optical depth.
    Cbc : float or array
        Fraction of UV light emitted from within BCs.
    kv : float, optional
        Dust extinction coefficient in V-band (default is 0.62e-5 M_sun^-1 kpc^2 ).

    Returns
    -------
    tauv_tot : float or array
        Total depth in V band.

    """

    maxarray = logsigma_mdust * 0 + fbc * 0 + Rhat * 0 + Hhat * 0
    logsigma_mdust, fbc, Rhat, Hhat = maxarray + logsigma_mdust, maxarray+fbc, maxarray + Rhat, maxarray + Hhat
    cat = np.load('tau_diff_ngrid.npz')
    Afuv_diff = interpolate.interpn((cat['logsigma_mdust_arr'], cat['fbc_arr'], cat['fRs_arr'], cat['fZs_arr']), cat['afuv_mod_grid'], np.transpose([logsigma_mdust, fbc, Rhat, Hhat]), bounds_error = False, fill_value=0, method='linear')
    tauv_diff = Afuv_diff / 1.086 / 2.5 * kv / (0.62e-5)
    tauv_bc = -np.log(1 - Cbc + Cbc * np.exp(-tau_bc))
    tauv_tot = tauv_bc + tauv_diff
    return tauv_tot


## Example
#logsigma_mdust_arr, fbc, Rhat, Rhat, tau_bc, Cbc = 0.2, 0.02, 0.12, 0.12, 0.31, 0.84 
#tau_v=tauv_model(logsigma_mdust=logsigma_mdust_arr,fbc=fbc,Rhat=Rhat,Hhat=Hhat,tau_bc=tau_bc,Cbc=Cbc,kv=0.62e-5)
#tau_fuv = tau_v * 2.5 
#IRX = (np.exp(tau_fuv) - 1) / 0.46






