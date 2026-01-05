import string

import numpy as np
import pandas as pd
import xarray as xr
from xarray.testing import assert_allclose, assert_equal

from xarray_rrepr.wrap import (
    deparse_xarray_variable,
    deparse_xarray_variables,
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
    expected = "xr.DataArray(\n    np.array([[25, 23, 22], [5, 3, 2], [20, 18, 17]]),\n"
    expected += '    coords={"a": (("a",), np.array([4, 0, 3])), "b": (("b",),'
    expected += " np.array([4, 2, 1]))},\n)\n"
    assert result == expected


def test_randomised_repr_dataset():
    ds = xr.tutorial.load_dataset("air_temperature")
    result = rrepr(ds, size=5, seed=42)
    expected = (
        'xr.Dataset(\n    {\n        "air": (\n            ("time", "lat", "lon"),\n'
    )
    expected += "            np.array(\n                [\n                    [\n"
    expected += "                        [279.0, 276.9, 273.9, 275.0, 277.6],\n"
    expected += "                        [299.1, 299.6, 301.2, 301.9, 298.9],\n"
    expected += "                        [285.0, 285.5, 286.8, 292.5, 281.6],\n"
    expected += "                        [289.7, 291.5, 293.6, 290.6, 297.7],\n"
    expected += "                        [286.6, 286.7, 287.6, 289.5, 286.3],\n"
    expected += "                    ],\n                    [\n"
    expected += "                        [241.4, 258.5, 250.2, 247.9, 263.8],\n"
    expected += "                        [297.2, 297.4, 300.3, 301.1, 300.0],\n"
    expected += "                        [271.5, 278.0, 273.2, 274.6, 271.2],\n"
    expected += "                        [286.6, 289.6, 285.6, 282.1, 291.2],\n"
    expected += "                        [278.4, 279.9, 272.0, 275.0, 274.2],\n"
    expected += "                    ],\n                    [\n"
    expected += "                        [258.2, 261.2, 251.0, 249.4, 262.8],\n"
    expected += "                        [297.0, 296.1, 300.5, 300.4, 298.7],\n"
    expected += "                        [274.6, 279.2, 269.8, 263.5, 270.9],\n"
    expected += "                        [286.1, 285.3, 283.7, 276.2, 291.5],\n"
    expected += "                        [278.4, 280.0, 271.7, 265.2, 273.5],\n"
    expected += "                    ],\n                    [\n"
    expected += "                        [240.7, 243.4, 248.9, 248.9, 257.3],\n"
    expected += "                        [297.2, 295.4, 297.8, 299.8, 298.9],\n"
    expected += "                        [274.6, 275.1, 270.1, 267.4, 267.5],\n"
    expected += "                        [283.4, 282.6, 278.6, 274.1, 285.2],\n"
    expected += "                        [277.7, 275.3, 268.9, 270.4, 269.7],\n"
    expected += "                    ],\n                    [\n"
    expected += "                        [256.7, 258.6, 239.9, 237.7, 263.8],\n"
    expected += "                        [299.4, 298.8, 302.7, 302.4, 301.2],\n"
    expected += "                        [279.8, 280.0, 264.4, 265.7, 272.4],\n"
    expected += "                        [289.2, 291.7, 277.5, 272.0, 290.4],\n"
    expected += "                        [283.3, 281.0, 265.8, 262.1, 276.9],\n"
    expected += "                    ],\n                ]\n            ),\n        )\n"
    expected += (
        '    },\n    coords={\n        "lat": (("lat",), np.array([72.5, 15.0, 52.5,'
    )
    expected += ' 40.0, 50.0])),\n        "lon": (("lon",), np.array([232.5, 215.0,'
    expected += ' 287.5, 280.0, 302.5])),\n        "time": (\n'
    expected += '            ("time",),\n            np.array(\n'
    expected += "                [\n                    datetime.datetime(2014, 7,"
    expected += " 19, 6, 0),\n                    datetime.datetime(2013, 11,"
    expected += " 17, 6, 0),\n                    datetime.datetime(2014, 4,"
    expected += " 23, 12, 0),\n                    datetime.datetime(2013, 3,"
    expected += " 7, 0, 0),\n                    datetime.datetime(2013, 11,"
    expected += " 13, 0, 0),\n                ],\n                dtype=object,\n"
    expected += "            ),\n        ),\n    },\n)\n"
    assert result == expected
