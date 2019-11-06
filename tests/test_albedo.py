
import numpy as np
import snowoptics


def test_KZ04():

    albedo = snowoptics.albedo_KZ04(wavelengths=800e-9, sza=np.deg2rad(45),
                                    ssa=20, r_difftot=0.1)
    assert np.allclose(albedo, 0.8905460508683)


def test_P20():

    albedo = snowoptics.albedo_P20_slope(wavelengths=800e-9, sza=np.deg2rad(45), saa=np.deg2rad(180),
                                         ssa=20, r_difftot=0.1, slope=np.deg2rad(10), aspect=np.deg2rad(180),
                                         model="small_slope")

    assert np.allclose(albedo, 1.00753124608813)
