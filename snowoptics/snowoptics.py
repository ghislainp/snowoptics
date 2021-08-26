# -*- coding: utf-8 -*-
#
# Snowoptics (C) 2019-2020 Ghislain Picard
#
# MIT License
#
import datetime

from .refractive_index import refice, refsoot_imag, refhulis_imag, refdust_imag, MAE_dust_caponi
import numpy as np
import scipy.optimize

try:
    import ephem
except ImportError:
    ephem = None


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

        if abs_impurities is not None:
            if density_impurities is None:
                raise ValueError("No default density for the impurities '%s'" % species)
            MAC = 6 * np.pi / wavelengths / density_impurities * np.abs(abs_impurities)

        cossalb += 2.0 / ssa * content_impurities * MAC

    return cossalb


def compute_b(B, g):
    """To translate shape factors from Picard's to Libois' formulations (smallcase b versus B, g)
    :param B: absorption enhancement factor
    :param g: asymmetry factor
"""
    return 4 / 3 * np.sqrt(B / (1 - g))


def compute_g(gG):
    """Compute the physical asymmetry factor g for the geometric assymetry factor gG
"""
    return 0.5 * (1 + gG)


def compute_gG(g):
    """Compute the geometrical asymmetry factor gG for the physical assymetry factor g
"""
    return 2 * g - 1


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


def albedo_diffuse_bubbly_ice(wavelengths, a, f, ni="p2016"):
    """compute diffuse albedo of pure ice without taking into account for the surface reflectance and
     using the assymptotic radiative transfer theory

    :param a: bubble radius (m)
    :param f: is bubble fractional volume w/r to total volume (m3/m3). f = n * 4/3 * pi * a**3 if n is the number density of bubbles (m^-3)
    :param ni: refractive index: dataset name (see refractive_index.py) or an array as a function of wavelength)
"""
    if isinstance(ni, str):
        dataset_name = ni
        n, ni = refice(wavelengths, dataset_name)

    gG = 0.79  # for spherical bubbles

    absorption = (1 - f) * 4 * np.pi * ni / wavelengths  # ice_absorption coefficient

    y = 4 * np.sqrt(absorption * (4 / 3 * a) / (3 * f * (1 - gG)))

    albedo = np.exp(-y)

    return albedo

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

    if lsza >= np.pi / 2 or lsza < 0:
        return np.nan

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
        alpha = np.where(alpha >= 0, alpha, np.nan)
    except TypeError:
        if alpha < 0:
            alpha = np.nan
    return alpha


def albedo_correction_without_slope(wavelengths, albedo, difftot, sza, albedo_0=0.98):
    """method to estimate slope parameters from one albedo spectrum, and to subsequently correct the albedo using the
     estimated slope parameters

    :param wavelengths: wavelengths in meter
    :param albedo: albedo spectrum. Must have the same size as wavelengths
    :param difftot: difftot spectrum. Must have the same size as wavelengths
    :param sza: tsolar zenith angle (radian).
    :param albedo_0: target albedo in the visible range (400-550nm)
"""

    n_approx = 3. / 7 * (1 + 2 * np.cos(sza))
    mask = (wavelengths > 400e-9) & (wavelengths < 550e-9)

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


def albedo_timeseries_correction(wavelengths, albedo, difftot, sza, saa, constrained, wavelength_range_0=None, albedo_0=0.98):
    """method to correct a timeseries of albedo, with given timeseries of difftot, timeseries of sza and saa

    :param wavelengths: an array of wavelengths (meter)
    :param albedo: timeseries of albedos. Must be a list of array or 2D array. The second dimension is the same as that of the wavelengths.
    :param difftot: timeseries of diffuse over total ratios. Must be a list of array or 2D array. The second dimension is the same as that of the wavelengths.
    :param sza: timeseries of solar zenith angle (radian). The length is equal to the first dimension of the albedo array.
    :param saa: timeseries of solar aimuth angle (radian). The length is equal to the first dimension of the albedo array.
    :param constrained: whether to use the constrained or unconstrained method (see Picard et al. 2020)
    :param wavelength_range_0: len-2 tuple with minimum and maximum wavelengths to use for the constraint, when constrained=True
    :param albedo_0: albedo value to constrain, when constrained=True

    :returns: diffuse albedo, slope, aspect
"""
    if wavelength_range_0 is None:
        wavelength_range_0 = 400e-9, 500e-9

    def difference_function(params, *extras, return_model=False, constrained=False):
        if constrained:
            slope, aspect, multiplier, *adiff = params
        else:
            slope, aspect, *adiff = params

        albedo, difftot, sza, saa, wls = extras

        # time is the first dimension, wavelength is the second dimension
        adiff = np.array(adiff)[None, :]

        # local sza
        coslsza = np.cos(sza) * np.cos(slope) + np.sin(sza) * np.sin(slope) * np.cos(saa - aspect)
        K = coslsza / np.cos(sza)
        n = 3. / 7 * (1 + 2 * coslsza)

        model = (1 - difftot) * K[:, None] * adiff**n[:, None] + difftot * adiff
        if return_model:
            return model
        elif constrained:
            return np.concatenate((
                np.ravel(model - albedo),
                multiplier * (np.ravel(adiff)[(wls >= wavelength_range_0[0]) & (wls <= wavelength_range_0[1])] - albedo_0)
            ))
        else:
            return np.ravel(model - albedo)

    def cost_function(*args, **kwargs):
        return np.sum(difference_function(*args, **kwargs)**2)

    albedo = np.array(albedo)
    difftot = np.array(difftot)

    # if albedo.shape != difftot.shape:
    #    raise Exception("albedo and difftot array must have the same shape")

    params0 = [0.001, np.pi, *np.mean(albedo, axis=0)]

    if constrained:
        constraints = ({'type': 'eq', 
                        'fun': lambda params: params[2:][(wavelengths >= wavelength_range_0[0]) & (wavelengths <= wavelength_range_0[1])] - albedo_0}, )
        result = scipy.optimize.minimize(cost_function, params0,
                                         args=(albedo, difftot, sza, saa, wavelengths), constraints=constraints)
        result = result.x
    else:
        result, cov, info, msg, ierr = scipy.optimize.leastsq(difference_function, params0,
                                                              args=(albedo, difftot, sza, saa, wavelengths), full_output=True)

    albedo_diff = result[2:]

    slope = result[0]
    aspect = result[1]

    if slope < 0:
        slope = -slope
        aspect += np.pi

    return albedo_diff, slope, np.mod(aspect, 2 * np.pi)


def compute_sun_position(lon, lat, dts):
    """compute the position of the sun."""

    if ephem is None:
        raise Exception("The pyephem module must be installed to use this function")

    o = ephem.Observer()
    o.lat, o.long = str(lat), str(lon)

    sza = []
    saa = []

    for dt in np.atleast_1d(dts):
        if isinstance(dt, np.datetime64):
            dt = datetime.datetime.utcfromtimestamp(dt.astype('O') / 1e9)
        o.date = dt

        sza.append(np.pi / 2 - float(ephem.Sun(o).alt))
        saa.append(float(ephem.Sun(o).az))

    return np.array(sza), np.array(saa)
