import string

import numpy as np
import pandas as pd
import xarray as xr
from xarray.testing import assert_allclose, assert_equal

from xarray_rrepr.wrap import (
    deparse_xarray_variable,
    deparse_xarray_variables,
    drop_all_attrs_array,
    drop_all_attrs_dataset,
    random_sample_xarray_obj,
    rrepr,
    xarray_rrepr_template,
)

alphabet_string = string.ascii_lowercase
alphabet_array = np.array(list(alphabet_string))


def make_coords(values) -> dict:
    array = np.array(values)
    coords = {
        alphabet_array[dim].__str__(): np.arange(array.shape[dim])
        for dim in np.arange(array.ndim)
    }
    return coords


def make_array(values: list, coords: dict | None = None) -> xr.DataArray:
    if coords is None:
        coords = make_coords(values)
    return xr.DataArray(values, coords=coords)


def make_dataset(values: dict, coords: dict | None = None) -> xr.Dataset:
    if coords is None:
        coords = make_coords(values)
    return xr.Dataset(values, coords=coords)


def test_simplify_xarray():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds["air"]
    assert drop_all_attrs_array(da).attrs == {}
    assert (
        drop_all_attrs_dataset(ds).attrs == {}
        and drop_all_attrs_dataset(ds)["air"].attrs == {}
    )


def test_randomised_sample_array():
    da = make_array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]
    )
    result = random_sample_xarray_obj(da, size=3, seed=42)
    assert_equal(
        result,
        make_array(
            [[25, 23, 22], [5, 3, 2], [20, 18, 17]],
            coords={"a": [4, 0, 3], "b": [4, 2, 1]},
        ),
    )


def test_randomised_sample_dataset():
    ds = xr.tutorial.load_dataset("air_temperature")
    result = random_sample_xarray_obj(ds, size=3, seed=42)
    assert_allclose(
        result,
        make_dataset(
            {
                "air": (
                    ["time", "lat", "lon"],
                    [
                        [
                            [272.9, 279.79, 273.2],
                            [292.79, 286.7, 286.4],
                            [273.29, 280.29, 273.6],
                        ],
                        [
                            [271.6, 275.29, 270.6],
                            [287.7, 284.79, 277.6],
                            [272.2, 276.1, 271.1],
                        ],
                        [
                            [284.5, 285.6, 293.9],
                            [297.7, 291.6, 297.79],
                            [288.79, 286.79, 294.0],
                        ],
                    ],
                )
            },
            coords={
                "time": pd.to_datetime(
                    [
                        "2014-04-23T18:00:00.000000000",
                        "2013-03-07T00:00:00.000000000",
                        "2014-07-19T18:00:00.000000000",
                    ]
                ),
                "lat": [50.0, 37.5, 47.5],
                "lon": [307.5, 215.0, 285.0],
            },
        ),
    )


def test_convert_xarray2string():
    ds = make_dataset(
        {
            "air": (
                ["time", "lat", "lon"],
                [
                    [
                        [272.9, 279.79, 273.2],
                        [292.79, 286.7, 286.4],
                        [273.29, 280.29, 273.6],
                    ],
                    [
                        [271.6, 275.29, 270.6],
                        [287.7, 284.79, 277.6],
                        [272.2, 276.1, 271.1],
                    ],
                    [
                        [284.5, 285.6, 293.9],
                        [297.7, 291.6, 297.79],
                        [288.79, 286.79, 294.0],
                    ],
                ],
            )
        },
        coords={
            "time": pd.to_datetime(
                [
                    "2014-04-23T18:00:00.000000000",
                    "2013-03-07T00:00:00.000000000",
                    "2014-07-19T18:00:00.000000000",
                ]
            ),
            "lat": [50.0, 37.5, 47.5],
            "lon": [307.5, 215.0, 285.0],
        },
    )
    variable_expr = deparse_xarray_variable(ds.to_dict()["data_vars"]["air"])
    expected = "(('time', 'lat', 'lon'), [[[272.9, 279.79, 273.2], [292.79, 286.7, 286.4], [273.29, 280.29, 273.6]], [[271.6, 275.29, 270.6], [287.7, 284.79, 277.6], [272.2, 276.1, 271.1]], [[284.5, 285.6, 293.9], [297.7, 291.6, 297.79], [288.79, 286.79, 294.0]]]"
    coords_expr = deparse_xarray_variables(ds.to_dict()["coords"])
    expected = "{'time': ((...), [...]), 'lat': ((...), [...]), 'lon': ((...), [...])}"
    data_vars_expr = deparse_xarray_variables(ds.to_dict()["data_vars"])
    expected = "{'air': ((...), [...])}"
    xarray_rrepr_template("xr.Dataset", data_vars_expr, coords_expr)
    xarray_rrepr_template("xr.DataArray", ds["air"].to_dict()["data"], coords_expr)


def test_randomised_repr_array():
    da = make_array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]
    )
    result = rrepr(da, size=3, seed=42)
    expected = 'xr.DataArray(\n    np.array([[25, 23, 22], [5, 3, 2], [20, 18, 17]]),\n    coords=[np.array([4, 0, 3]), np.array([4, 2, 1])],\n    dims=["a", "b"],\n)\n'
    assert result == expected


def test_randomised_repr_dataset():
    ds = xr.tutorial.load_dataset("air_temperature")
    result = rrepr(ds, size=5, seed=42)
    expected = 'xr.DataArray(\n    np.array([[25, 23, 22], [5, 3, 2], [20, 18, 17]]),\n    coords=[np.array([4, 0, 3]), np.array([4, 2, 1])],\n    dims=["a", "b"],\n)\n'
    assert result == expected
