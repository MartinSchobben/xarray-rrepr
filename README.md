# Minimised, Random String Representation for Xarray


As opposed to `xarray` standard string representation `rrepr` produces a
representation that can be evaluated. However the object is a minimised
random subsample of the original xarray object. It can therefore be used
as faithful excerpts for creating easily readible unit tests and example
use cases in documentation of otherwise unpractically large datasets.

## Installation

``` {bash}
pip install git+https://github.com/MartinSchobben/xarray-rrepr.git
```

## Basic Usage

This example showcases the two prime characteristics of the `rrepr`
method that distinguishes it from the orignal xarray `repr` method.

``` python
import numpy as np
import datetime
import pyperclip
import xarray as xr
from xarray.testing import assert_allclose
from xarray_rrepr import rrepr

xr.set_options(display_style="text")
```

    <xarray.core.options.set_options at 0x7fb058364ad0>

Let’s use xarray example dataset “air_temperature” loaded as an xarray
`Dataset`.

``` python
ds = xr.tutorial.load_dataset("air_temperature")
```

Let’s inspect how the string representation differs when printed.

``` python
print(repr(ds))
```

    <xarray.Dataset> Size: 31MB
    Dimensions:  (time: 2920, lat: 25, lon: 53)
    Coordinates:
      * time     (time) datetime64[ns] 23kB 2013-01-01 ... 2014-12-31T18:00:00
      * lat      (lat) float32 100B 75.0 72.5 70.0 67.5 65.0 ... 22.5 20.0 17.5 15.0
      * lon      (lon) float32 212B 200.0 202.5 205.0 207.5 ... 325.0 327.5 330.0
    Data variables:
        air      (time, lat, lon) float64 31MB 241.2 242.5 243.5 ... 296.2 295.7
    Attributes:
        Conventions:  COARDS
        title:        4x daily NMC reanalysis (1948)
        description:  Data is from NMC initialized reanalysis\n(4x/day).  These a...
        platform:     Model
        references:   http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanaly...

This is the usual output of xarray’s `repr`.

``` python
print(rrepr(ds))
```

    xr.Dataset(
        {
            "air": (
                ("time", "lat", "lon"),
                np.array(
                    [[[271.5, 275.6], [297.1, 295.8]], [[264.7, 273.9], [297.1, 294.7]]]
                ),
            )
        },
        coords={
            "lat": (("lat",), np.array([55.0, 15.0])),
            "lon": (("lon",), np.array([235.0, 325.0])),
            "time": (
                ("time",),
                np.array(
                    [
                        datetime.datetime(2014, 1, 17, 18, 0),
                        datetime.datetime(2014, 3, 22, 0, 0),
                    ],
                    dtype=object,
                ),
            ),
        },
    )

We can see that in the case of `rrepr` we created an output that looks
like how we would create an xarray `Dataset` from scratch.

### Evaluate the String Representation

The first characteristics of `rrepr` is that it can be evaluated in the
context of globals and locals.

``` python
eval(rrepr(ds))
```

<pre>&lt;xarray.Dataset&gt; Size: 112B
Dimensions:  (time: 2, lat: 2, lon: 2)
Coordinates:
  * time     (time) datetime64[ns] 16B 2013-09-10T12:00:00 2014-11-23
  * lat      (lat) float64 16B 32.5 30.0
  * lon      (lon) float64 16B 225.0 272.5
Data variables:
    air      (time, lat, lon) float64 64B 293.9 297.5 294.9 ... 294.1 290.7</pre>

Which can’t be done with standard `repr`.

``` python
eval(repr(ds))
```

    SyntaxError: invalid decimal literal (<string>, line 1)
    Traceback [36m(most recent call last)[39m:

      File [92m~/Documents/work/projects/xarray-rrepr/.venv/lib/python3.13/site-packages/IPython/core/interactiveshell.py:3701[39m in [95mrun_code[39m
        exec(code_obj, self.user_global_ns, self.user_ns)

    [36m  [39m[36mCell[39m[36m [39m[32mIn[6][39m[32m, line 1[39m
    [31m    [39m[31meval(repr(ds))[39m

      [36mFile [39m[32m<string>:1[39m
    [31m    [39m[31m<xarray.Dataset> Size: 31MB[39m
                                ^
    [31mSyntaxError[39m[31m:[39m invalid decimal literal

### Minimised representation

However, `rrepr` is not a true representative string of the xarray
object, but, rather a subsample of the original object. This makes it
ideal for debugging and testing large datasets as well as create
documentation with easily readible and comprehendible code examples.

``` python
assert_allclose(ds, eval(rrepr(ds)))
```

    AssertionError: Left and right Dataset objects are not close
    Differing dimensions:
        (time: 2920, lat: 25, lon: 53) != (time: 2, lat: 2, lon: 2)
    Differing coordinates:
    L * lat      (lat) float32 100B 75.0 72.5 70.0 67.5 65.0 ... 22.5 20.0 17.5 15.0
    R * lat      (lat) float64 16B 57.5 60.0
    L * lon      (lon) float32 212B 200.0 202.5 205.0 207.5 ... 325.0 327.5 330.0
    R * lon      (lon) float64 16B 287.5 205.0
    L * time     (time) datetime64[ns] 23kB 2013-01-01 ... 2014-12-31T18:00:00
    R * time     (time) datetime64[ns] 16B 2013-04-21T18:00:00 2014-04-09
    Differing data variables:
    L   air      (time, lat, lon) float64 31MB 241.2 242.5 243.5 ... 296.2 295.7
    R   air      (time, lat, lon) float64 64B 260.5 274.9 260.0 ... 256.1 264.2
    [31m---------------------------------------------------------------------------[39m
    [31mAssertionError[39m                            Traceback (most recent call last)
    [36mCell[39m[36m [39m[32mIn[7][39m[32m, line 1[39m
    [32m----> [39m[32m1[39m [43massert_allclose[49m[43m([49m[43mds[49m[43m,[49m[43m [49m[38;5;28;43meval[39;49m[43m([49m[43mrrepr[49m[43m([49m[43mds[49m[43m)[49m[43m)[49m[43m)[49m

        [31m[... skipping hidden 1 frame][39m

    [36mFile [39m[32m~/Documents/work/projects/xarray-rrepr/.venv/lib/python3.13/site-packages/xarray/testing/assertions.py:260[39m, in [36massert_allclose[39m[34m(a, b, rtol, atol, decode_bytes, check_dim_order)[39m
    [32m    256[39m [38;5;28;01melif[39;00m [38;5;28misinstance[39m(a, Dataset):
    [32m    257[39m     allclose = a._coord_names == b._coord_names [38;5;129;01mand[39;00m utils.dict_equiv(
    [32m    258[39m         a.variables, b.variables, compat=compat_variable
    [32m    259[39m     )
    [32m--> [39m[32m260[39m     [38;5;28;01massert[39;00m allclose, formatting.diff_dataset_repr(a, b, compat=equiv)
    [32m    261[39m [38;5;28;01melif[39;00m [38;5;28misinstance[39m(a, Coordinates):
    [32m    262[39m     allclose = utils.dict_equiv(a.variables, b.variables, compat=compat_variable)

    [31mAssertionError[39m: Left and right Dataset objects are not close
    Differing dimensions:
        (time: 2920, lat: 25, lon: 53) != (time: 2, lat: 2, lon: 2)
    Differing coordinates:
    L * lat      (lat) float32 100B 75.0 72.5 70.0 67.5 65.0 ... 22.5 20.0 17.5 15.0
    R * lat      (lat) float64 16B 57.5 60.0
    L * lon      (lon) float32 212B 200.0 202.5 205.0 207.5 ... 325.0 327.5 330.0
    R * lon      (lon) float64 16B 287.5 205.0
    L * time     (time) datetime64[ns] 23kB 2013-01-01 ... 2014-12-31T18:00:00
    R * time     (time) datetime64[ns] 16B 2013-04-21T18:00:00 2014-04-09
    Differing data variables:
    L   air      (time, lat, lon) float64 31MB 241.2 242.5 243.5 ... 296.2 295.7
    R   air      (time, lat, lon) float64 64B 260.5 274.9 260.0 ... 256.1 264.2
