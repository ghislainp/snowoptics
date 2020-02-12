# -*- coding: utf-8 -*-
#
# Snowoptics (C) 2019 Ghislain Picard
#
# MIT License
#

from .refractive_index import refice, refsoot_imag, refhulis_imag, refdust_imag, MAE_dust_caponi
import numpy as np


# B=1.6 and g=0.85 are equivalent to b=4.3 found in Picard et al. 2009
default_B = 1.6
default_g = 0.845


def albedo_KZ04(wavelengths, sza, ssa, r_difftot=0, impurities=None, ni="p2016", B=default_B, g=default_g):
    """compute albedo using Kokhanovsky and Zege 2004 (ART) theory

    :param wavelengths: wavelengths (meter)
    :param sza: solar zenith angle (radian)
    :param ssa: snow specific surface area (m2/kg)
    :param r_difftot: ratio diff/tot of the incoming radiation
    :param impurities: dict with species as key and concentration (kg/kg) as values (or tuple of concentration and bulk density). E.g. {'BC': 10e-9}
    :param ni: refractive index: dataset name (see refractive_index.py) or an array as a function of wavelength)
    :param B: absorption enhancement factor in grains
    :param g: asymmetry factor
"""

    return (1 - r_difftot) * albedo_direct_KZ04(wavelengths, sza, ssa, impurities=impurities, ni=ni, B=B, g=g) + \
        r_difftot * albedo_diffuse_KZ04(wavelengths, ssa, impurities=impurities, ni=ni, B=B, g=g)


def albedo_diffuse_KZ04(wavelengths, ssa, impurities=None, ni="p2016", B=default_B, g=default_g):
    """compute diffuse albedo using Kokhanovsky and Zege 2004 (ART) theory and considering BC impurities from Kokhanovsky et al., 2013.

    :param wavelengths: wavelengths (meter)
    :param ssa: snow specific surface area (m2/kg)
    :param impurities: dict with species as key and concentration (kg/kg) as values (or tuple of concentration and bulk density). E.g. {'BC': 10e-9}
    :param ni: refractive index: dataset name (see refractive_index.py) or an array as a function of wavelength)
    :param B: absorption enhancement factor in grains
    :param g: asymmetry factor
"""
    alpha = compute_alpha(wavelengths, ssa, impurities=impurities, ni=ni, B=B, g=g)
    R = np.exp(-np.sqrt(alpha))
    return R


def albedo_direct_KZ04(wavelengths, sza, ssa, impurities=None, ni="p2016", B=default_B, g=default_g):
    """compute direct albedo using Kokhanovsky and Zege 2004 (ART) theory and considering BC impurities from Kokhanovsky et al., 2013

    :param wavelengths: wavelengths (meter)
    :param sza: solar zenith angle (radian)
    :param ssa: snow specific surface area (m2/kg)
    :param impurities: dict with species as key and concentration (kg/kg) as values (or tuple of concentration and bulk density). E.g. {'BC': 10e-9}
    :param ni: refractive index: dataset name (see refractive_index.py) or an array as a function of wavelength)
    :param B: absorption enhancement factor in grains
    :param g: asymmetry factor
"""

    alpha = compute_alpha(wavelengths, ssa, impurities=impurities, ni=ni, B=B, g=g)
    cos_sza = np.cos(sza)
    assert cos_sza >= 0  # a negative value is probably because sza is not in radian
    R = np.exp(-np.sqrt(alpha) * 3.0 / 7 * (1 + 2 * cos_sza))
    return R


def compute_alpha(wavelengths, ssa, impurities=None, ni="p2016", B=default_B, g=default_g):
    """compute alpha for the effect of BC from Kokhanovsky et al., 2013 (see also Dumont et al., 2017 or Picard et al. 2020)
"""

    cossalb = compute_co_single_scattering_albedo(wavelengths, ssa, impurities=impurities, ni=ni, B=B, g=g)

    alpha = 16. / 3 * cossalb / (1 - g)
    return alpha


def compute_co_single_scattering_albedo(wavelengths, ssa, impurities=None, ni="p2016", B=default_B, g=default_g):
    """compute the co single scattering albedo
"""

    if isinstance(ni, str):
        dataset_name = ni
        n, ni = refice(wavelengths, dataset_name)

    gamma = 4 * np.pi * ni / wavelengths  # ice absorption

    rho_ice = 917.0
    cossalb = 2.0 / (ssa * rho_ice) * B * gamma  # co single scattering albedo

    if impurities is None:
        return cossalb

    for species in impurities:

        try:
            content_impurities, density_impurities = impurities[species]
        except TypeError:
            content_impurities = impurities[species]
            density_impurities = None

        abs_impurities = None
        if species in ["soot", "BC"]:
            if density_impurities is None:
                density_impurities = 1270.
            abs_impurities = refsoot_imag(wavelengths, enhancement_param=1)  # enhancement =1.638 in Tuzet et al. 2019
        elif species == "hulis":
            abs_impurities = refhulis_imag(wavelengths)
        elif species == "dust":
            abs_impurities = refdust_imag(wavelengths)
        elif species == "dust_skiles":
            abs_impurities = refdust_imag(wavelengths, formulation="skiles2014")
        elif species.startswith("dust_"):
            try:
                MAC = MAE_dust_caponi(wavelengths, formulation=species[5:])
            except ValueError:
                raise ValueError("Invalid species")
        else:
            raise ValueError("Invalid species")

        if density_impurities is None:
            raise ValueError("No default density for the impurities '%s'" % species)

        if abs_impurities is not None:
            MAC = 6 * np.pi / wavelengths / density_impurities * np.abs(abs_impurities)

        cossalb += 2.0 / ssa * content_impurities * MAC

    return cossalb


def compute_b(B, g):
    """To translate shape factors from Picard's to Libois' formulations (smallcase b versus B, g)
    :param B: absorption enhancement factor
    :param g: asymmetry factor
"""
    return 4 / 3 * np.sqrt(B / (1 - g))


def extinction_KZ04(wavelengths, rho, ssa, impurities=None, ni="p2016", B=default_B, g=default_g):
    """compute asymptotic extinction (AFEC) according to Kokhanovsky and Zege theory

    :param wavelengths: wavelengths (meter)
    :param rho: snow density (kg/m3)
    :param ssa: snow specific surface area (m2/kg)
    :param impurities: dictionary with species as key and (density, concentration) as values (in kg/kg). E.g. {'BC': 10e-9}
    :param ni: refractive index: dataset name (see refractive_index.py) or an array as a function of wavelength)
    :param B: absorption enhancement factor in grains
    :param g: asymmetry factor
"""

    cossalb = compute_co_single_scattering_albedo(wavelengths, ssa, impurities=impurities, ni=ni, B=B, g=g)

    sigext = rho * ssa / 2.0
    ke = sigext * np.sqrt(3 * cossalb * (1 - g))
    return ke


#
# The following is described in Picard et al. 2020
#


def local_sza(sza, saa, slope, aspect):
    """compute the effective solar zenith angle for a given slope"""
    mu = np.cos(sza) * np.cos(slope) + np.sin(sza) * np.sin(slope) * np.cos(saa - aspect)
    return np.arccos(mu)


def albedo_direct_KZ04_slope(wavelengths, sza, ssa, impurities=None, ni="p2016", B=default_B, g=default_g, slope=0, aspect=0, saa=0):
    """compute direct albedo with AART include change of local sza due to slope"""
    # BC is in g g-1
    # ajout Marie de l'effet de BC from Kokhanovsky et al., 2013 and Dumont et al., 2017

    sza_eff = local_sza(sza, saa, slope, aspect)
    R = albedo_direct_KZ04(wavelengths, sza_eff, ssa, impurities=impurities, ni=ni, B=B, g=g)
    return R


def albedo_P20_slope(wavelengths, sza, saa, ssa, r_difftot, slope, aspect, model,
                     measured_difftot=False, fixed_flat_albedo=None, **kwargs):
    """compute albedo on a tilted terrain based on Picard et al. 2020

    :param wavelengths: wavelengths (meter)
    :param sza: solar zenith angle (radian)
    :param saa: solar azimut angle (radian)
    :param ssa: snow specific surface area (m2/kg)
    :param r_difftot: ratio diff/tot of the incoming radiation
    :param slope: slope inclination (radian)
    :param slope: slope aspect (radian)
    :param model: model to use: "flat", "small_slope", "DM", "DT", "SM", "ST"
    :param impurities: (optional) dict with species as key and concentration (kg/kg) as values (or tuple of concentration and bulk density). E.g. {'BC': 10e-9}
    :param ni: (optional) refractive index (string or array as a function of wavelength)
    :param B: (optional) absorption enhancement factor in grains
    :param g: (optional) asymmetry factor
"""

    lsza = local_sza(sza, saa, slope, aspect)

    if fixed_flat_albedo:
        alb_loc_dir = fixed_flat_albedo
        alb_dir = fixed_flat_albedo
        alb_diff = fixed_flat_albedo
    else:
        alb_loc_dir = albedo_direct_KZ04(wavelengths, lsza, ssa, **kwargs)
        alb_dir = albedo_direct_KZ04(wavelengths, sza, ssa, **kwargs)
        alb_diff = albedo_diffuse_KZ04(wavelengths, ssa, **kwargs)

    K = max(np.cos(lsza) / np.cos(sza), 0)
    V = (1 + np.cos(slope)) / 2
    M = (1 - V) * alb_diff

    if (not measured_difftot) and (model in ['DM', 'SM']):
        model_top = model[:-1] + 'T'  # swith to the equivalent model at the top of the hill
        albedo_top = albedo_P20_slope(wavelengths, sza, saa, ssa, r_difftot, slope, aspect, model_top,
                                      measured_difftot=False, fixed_flat_albedo=fixed_flat_albedo, **kwargs)

        if model == "DM":
            if K > 0:
                return albedo_top / (1 - (1 - V) * r_difftot + (1 - V) / V * albedo_top)
            elif r_difftot > 0:
                return V * alb_diff / (1 + M)
            else:
                return np.nan
        elif model == "SM":
            if K > 0:
                return albedo_top / (1 + (1 - V) *
                                     ((1 - r_difftot) * (K * alb_loc_dir + M * alb_dir) / (1 - M**2) +
                                      r_difftot * (V / (1 - M) * alb_diff - 1)))
            elif r_difftot > 0:
                return alb_diff
            else:
                return np.nan
        else:
            raise RuntimeError("should not be here")

    if model == "small_slope":
        Adir = K * alb_loc_dir
        Adiff = alb_diff
    elif model == "DT":
        Adir = V * K * alb_loc_dir
        Adiff = V**2 * alb_diff
    elif model == "DM":
        Adir = V / (1 + M) * K * alb_loc_dir
        Adiff = V / (1 + M) * alb_diff
        if K == 0:
            raise NotImplementedError("Incorrect in the self-shadow")
    elif model == "ST":
        Adir = ((V + M * (1 - V)) * K * alb_loc_dir + (M * V + (1 - V)) * alb_dir) / (1 - M**2)
        Adiff = V / (1 - M) * alb_diff
    elif model == "SM":
        Adir = V / (1 + M) * K * alb_loc_dir + (1 - V) / (1 + M) * alb_dir
        Adiff = alb_diff
        if K == 0:
            raise NotImplementedError("Incorrect in the self-shadow")
    elif model == "flat":  # flat
        Adir = alb_dir
        Adiff = alb_diff
    else:
        raise ValueError("Invalid slope model")

    alpha = (1 - r_difftot) * Adir + r_difftot * Adiff

    try:
        len(alpha)
        alpha[alpha < 0] = np.nan
    except TypeError:
        if alpha < 0:
            alpha = np.nan
    return alpha


def albedo_correction_without_slope(wavelengths, albedo, difftot, sza, albedo_0=0.98):
    """method to estimate slope parameters from one albedo spectrum, and to subsequently correct the albedo using the
     estimated slope parameters"""

    n_approx = 3. / 7 * (1 + 2 * np.cos(sza))
    mask = (wavelengths > 400) & (wavelengths < 550)

    K = np.sum((1 - difftot[mask] * albedo_0) * (albedo[mask] - difftot[mask])) / np.sum((1 - difftot[mask])**2 * albedo_0**n_approx)
    return K, albedo_correction_with_slope(albedo, difftot, sza, K)


def albedo_correction_with_slope(albedo, difftot, sza, K, niteration=5):
    """iterative method to correct the albedo from known slope parameters."""
    # implemented in the paper
    n = 3. / 7 * (1 + 2 * K * np.cos(sza))

    albedo_diff = np.minimum(albedo, 1)  # first guess
    for i in range(niteration):
        albedo_diff = ((albedo - (1 - difftot) * K * (albedo_diff**n - albedo_diff)) / (difftot + (1 - difftot) * K))
        albedo_diff = np.maximum(albedo_diff, 0)

    return albedo_diff, albedo_diff**n


def albedo_correction_with_slope2(albedo, difftot, sza, K, niteration=5):
    """alternative iterative method to correct the albedo from known slope parameters."""

    n = 3. / 7 * (1 + 2 * K * np.cos(sza))

    albedo_diff = np.minimum(albedo, 1)  # first guess
    for i in range(niteration):
        albedo_diff = ((albedo - difftot * (albedo_diff - albedo_diff**n)) / ((1 - difftot) * K + difftot))**(1. / n)
        albedo_diff = np.maximum(albedo_diff, 0)

    return albedo_diff, albedo_diff**n
