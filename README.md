

Snowoptics library
===================

Snowoptics provides functions to compute spectral albedo and extinction of snow using the Asymptotic Radiative Transfer theory on flat and tilted terrains. It also provides functions to correct for slope distortion of spectral albedo measurements [Picard et al. submitted]().

Snowoptics only implements simple analytical equations, and considers homogeneous and semi-infinite snowpack. For more complex layered snowpacks, see [tartes model](https://pypi.org/project/tartes/).

Examples
----------

To compute the albedo of a flat snow surface with SSA 20 kg/m2 at 800 nm and with 10% radiation from the sky and 90% from the sun at 45° zenith angle:

```python
import numpy as np
import snowoptics

print(snowoptics.albedo_KZ04(wavelengths=800e-9, sza=np.deg2rad(45), ssa=20, r_difftot=0.1))
```

And with some 100 ng/g of black carbon (calculation at 400 nm):

```python
import numpy as np
import snowoptics

print(snowoptics.albedo_KZ04(wavelengths=400e-9, sza=np.deg2rad(45), ssa=20, r_difftot=0.1, {'BC': 100e-9}))
```

To compute albedo over a tilted surface (10° south facing) as described in [Picard et al. submitted]()

```python
print(snowoptics.albedo_P20_slope(wavelengths=800e-9, sza=np.deg2rad(45), saa=np.deg2rad(180), ssa=20, r_difftot=0.1, slope=np.deg2rad(10), aspect=np.deg2rad(180), model="small_slope"))
```

Intercative albedo calculations
--------------------------------

Lazy to code today? see the interactive webapp that provides most of snowoptics [snowslope](https://snowslope.pythonanywhere.com/)

License information
---------------------

Snowoptics is really open-source (MIT License)

Copyright (c) 2019 Ghislain Picard



