import string

import numpy as np
import numpy.testing as np_test
import pandas as pd
import xarray as xr
import xarray.testing as xr_test
from numpy.typing import ArrayLike

from xarray_rrepr.wrap import (
    deparse_xarray_variable,
    deparse_xarray_variables,
    random_sample_dims,
    random_sample_xarray_obj,
    rrepr,
    xarray_rrepr_template,
)

alphabet_string = string.ascii_lowercase
alphabet_array = np.array(list(alphabet_string))


def make_coords(values: ArrayLike) -> dict:
    array = np.array(values)
    return {
        alphabet_array[dim].__str__(): np.arange(array.shape[dim])
        for dim in np.arange(array.ndim)
    }


def make_array(values: list, coords: dict | None = None) -> xr.DataArray:
    if coords is None:
        coords = make_coords(values)
    return xr.DataArray(values, coords=coords)


def make_dataset(values: dict, coords: dict | None = None) -> xr.Dataset:
    if coords is None:
        coords = make_coords(values)
    return xr.Dataset(values, coords=coords)


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
    result = random_sample_dims(da, size=3, seed=42)
    np_test.assert_equal(result, np.array([[2, 0, 4], [2, 0, 3]], dtype=np.int64))

    result = random_sample_xarray_obj(da, size=3, seed=42)
    xr_test.assert_equal(
        result,
        make_array(
            [[13, 11, 14], [3, 1, 4], [23, 21, 24]],
            coords={"a": [2, 0, 4], "b": [2, 0, 3]},
        ),
    )


def test_randomised_sample_dataset():
    ds = xr.tutorial.load_dataset("air_temperature")
    result = random_sample_xarray_obj(ds, size=3, seed=42)
    xr_test.assert_allclose(
        result,
        make_dataset(
            {
                "air": (
                    ["time", "lat", "lon"],
                    [
                        [
                            [294.9, 299.7, 295.1],
                            [269.29, 259.79, 271.4],
                            [269.29, 259.79, 271.4],
                        ],
                        [
                            [296.4, 300.5, 296.79],
                            [274.5, 273.29, 275.6],
                            [274.5, 273.29, 275.6],
                        ],
                        [
                            [295.2, 298.9, 294.79],
                            [260.2, 248.89, 263.1],
                            [260.2, 248.89, 263.1],
                        ],
                    ],
                )
            },
            coords={
                "time": pd.to_datetime(
                    [
                        "2014-04-03T12:00:00.000000000",
                        "2013-06-20T00:00:00.000000000",
                        "2013-01-28T00:00:00.000000000",
                    ]
                ),
                "lat": [15.0, 67.5, 67.5],
                "lon": [327.5, 287.5, 330.0],
            },
        ),
    )


def test_deparse_xarray():
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
    expected = "(('time', 'lat', 'lon'), array([[[272.9, 279.8, 273.2],\n"
    expected += "        [292.8, 286.7, 286.4],\n"
    expected += "        [273.3, 280.3, 273.6]],\n"
    expected += "\n"
    expected += "       [[271.6, 275.3, 270.6],\n"
    expected += "        [287.7, 284.8, 277.6],\n"
    expected += "        [272.2, 276.1, 271.1]],\n"
    expected += "\n"
    expected += "       [[284.5, 285.6, 293.9],\n"
    expected += "        [297.7, 291.6, 297.8],\n"
    expected += "        [288.8, 286.8, 294. ]]]))"
    assert str(variable_expr) == expected

    coords_expr = deparse_xarray_variables(ds.to_dict()["coords"])
    expected = "{'time': (('time',), array([datetime.datetime(2014, 4, 23, 18, 0),\n"
    expected += "       datetime.datetime(2013, 3, 7, 0, 0),\n"
    expected += "       datetime.datetime(2014, 7, 19, 18, 0)], dtype=object)), "
    expected += "'lat': (('lat',), array([50. , 37.5, 47.5])), 'lon': (('lon',),"
    expected += " array([307.5, 215. , 285. ]))}"
    assert str(coords_expr) == expected

    data_vars_expr = deparse_xarray_variables(ds.to_dict()["data_vars"])
    expected = "{'air': (('time', 'lat', 'lon'), array([[[272.9, 279.8, 273.2],\n"
    expected += "        [292.8, 286.7, 286.4],\n"
    expected += "        [273.3, 280.3, 273.6]],\n"
    expected += "\n"
    expected += "       [[271.6, 275.3, 270.6],\n"
    expected += "        [287.7, 284.8, 277.6],\n"
    expected += "        [272.2, 276.1, 271.1]],\n"
    expected += "\n"
    expected += "       [[284.5, 285.6, 293.9],\n"
    expected += "        [297.7, 291.6, 297.8],\n"
    expected += "        [288.8, 286.8, 294. ]]]))}"
    assert str(data_vars_expr) == expected

    xarray_dataset_expr = xarray_rrepr_template(
        "xr.Dataset", data_vars_expr, coords_expr
    )
    expected = "xr.Dataset({'air': (('time', 'lat', 'lon'), array([[[272.9, 279.8,"
    expected += " 273.2],\n        [292.8, 286.7, 286.4],\n        [273.3, 280.3,"
    expected += " 273.6]],\n\n       [[271.6, 275.3, 270.6],\n        [287.7, 284.8,"
    expected += " 277.6],\n        [272.2, 276.1, 271.1]],\n\n       [[284.5, 285.6,"
    expected += " 293.9],\n        [297.7, 291.6, 297.8],\n        [288.8, 286.8,"
    expected += " 294. ]]]))}, coords={'time': (('time',), array([datetime."
    expected += "datetime(2014, 4, 23, 18, 0),\n       datetime.datetime(2013,"
    expected += " 3, 7, 0, 0),\n       datetime.datetime(2014, 7, 19, 18, 0)],"
    expected += " dtype=object)), 'lat': (('lat',), array([50. , 37.5, 47.5])),"
    expected += " 'lon': (('lon',), array([307.5, 215. , 285. ]))})"
    assert xarray_dataset_expr == expected

    xarray_data_array_expr = xarray_rrepr_template(
        "xr.DataArray", ds["air"].to_dict()["data"], coords_expr
    )
    expected = "xr.DataArray([[[272.9, 279.79, 273.2], [292.79, 286.7, 286.4],"
    expected += " [273.29, 280.29, 273.6]], [[271.6, 275.29, 270.6], [287.7,"
    expected += " 284.79, 277.6], [272.2, 276.1, 271.1]], [[284.5, 285.6,"
    expected += " 293.9], [297.7, 291.6, 297.79], [288.79, 286.79, 294.0]]],"
    expected += " coords={'time': (('time',), array([datetime.datetime(2014,"
    expected += " 4, 23, 18, 0),\n       datetime.datetime(2013, 3, 7, 0, 0),\n"
    expected += "       datetime.datetime(2014, 7, 19, 18, 0)], dtype=object)),"
    expected += " 'lat': (('lat',), array([50. , 37.5, 47.5])), 'lon': (('lon',),"
    expected += " array([307.5, 215. , 285. ]))})"
    assert xarray_data_array_expr == expected


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
    expected = "xr.DataArray(\n    np.array([[13, 11, 14], [3, 1, 4], [23, 21, 24]]),\n"
    expected += '    coords={"a": (("a",), np.array([2, 0, 4])), "b": (("b",),'
    expected += " np.array([2, 0, 3]))},\n)\n"
    assert result == expected


def test_randomised_repr_dataset():
    ds = xr.tutorial.load_dataset("air_temperature")
    result = rrepr(ds, size=5, seed=42)
    expected = (
        'xr.Dataset(\n    {\n        "air": (\n            ("time", "lat", "lon"),\n'
    )
    expected += "            np.array(\n                [\n                    [\n"
    expected += "                        [257.6, 255.5, 239.6, 267.0, 255.5],\n"
    expected += "                        [290.2, 291.4, 295.4, 292.7, 294.5],\n"
    expected += "                        [257.6, 255.5, 239.6, 267.0, 255.5],\n"
    expected += "                        [291.5, 292.7, 296.7, 293.4, 295.4],\n"
    expected += "                        [286.2, 284.7, 291.7, 289.8, 289.3],\n"
    expected += "                    ],\n                    [\n"
    expected += "                        [291.4, 292.7, 276.0, 273.4, 271.9],\n"
    expected += "                        [290.8, 293.4, 301.9, 296.9, 297.0],\n"
    expected += "                        [291.4, 292.7, 276.0, 273.4, 271.9],\n"
    expected += "                        [291.1, 293.6, 300.7, 297.0, 297.3],\n"
    expected += "                        [291.1, 292.9, 301.3, 293.9, 294.1],\n"
    expected += "                    ],\n                    [\n"
    expected += "                        [241.3, 241.5, 242.3, 257.1, 250.2],\n"
    expected += "                        [288.9, 290.9, 294.1, 292.8, 295.2],\n"
    expected += "                        [241.3, 241.5, 242.3, 257.1, 250.2],\n"
    expected += "                        [290.9, 292.2, 296.0, 294.0, 295.9],\n"
    expected += "                        [283.8, 283.8, 285.4, 291.9, 288.5],\n"
    expected += "                    ],\n                    [\n"
    expected += "                        [278.1, 281.3, 272.9, 273.5, 272.3],\n"
    expected += "                        [294.3, 295.9, 302.0, 297.9, 299.6],\n"
    expected += "                        [278.1, 281.3, 272.9, 273.5, 272.3],\n"
    expected += "                        [293.1, 295.6, 301.3, 297.6, 299.7],\n"
    expected += "                        [293.7, 295.6, 293.1, 297.6, 297.0],\n"
    expected += "                    ],\n                    [\n"
    expected += "                        [252.2, 248.0, 246.2, 265.4, 250.6],\n"
    expected += "                        [293.5, 293.4, 291.6, 295.4, 297.8],\n"
    expected += "                        [252.2, 248.0, 246.2, 265.4, 250.6],\n"
    expected += "                        [292.8, 295.1, 292.9, 295.1, 298.5],\n"
    expected += "                        [289.5, 287.7, 281.6, 291.4, 294.0],\n"
    expected += "                    ],\n                ]\n            ),\n        )\n"
    expected += '    },\n    coords={\n        "lat": (("lat",), np.array([67.5, 27.5,'
    expected += ' 67.5, 25.0, 37.5])),\n        "lon": (("lon",), np.array([225.0,'
    expected += ' 217.5, 267.5, 325.0, 312.5])),\n        "time": (\n'
    expected += '            ("time",),\n            np.array(\n'
    expected += "                [\n                    datetime.datetime(2014, 4,"
    expected += " 3, 12, 0),\n                    datetime.datetime(2013, 6,"
    expected += " 20, 0, 0),\n                    datetime.datetime(2013, 1,"
    expected += " 28, 0, 0),\n                    datetime.datetime(2014, 8,"
    expected += " 14, 12, 0),\n                    datetime.datetime(2014, 11,"
    expected += " 28, 18, 0),\n                ],\n                dtype=object,\n"
    expected += "            ),\n        ),\n    },\n)\n"
    assert result == expected
