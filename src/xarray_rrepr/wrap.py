"""xarray-rrepr: A utility for concise, minimal, reproducible xarray representations.

This module provides a monkey-patching utility to replace the default ``__repr__``
method of xarray ``Dataset`` and ``DataArray`` objects. The new representation
generates a minimised, randomly-sampled version of the object, which can be used for
debugging, logging, or displaying examples of large datasets
in a clean and readable format without overwhelming the console output.

The core functionality is built from several utility functions that handle
sampling, deparsing, and formatting, which can also be used independently.

Core Representation and Sampling Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These functions form the core engine for creating and manipulating the
minimised representations.

rrepr(obj, size=2, seed=None)
    The main public function to generate the minimised, random string
    representation for a given xarray Dataset or DataArray.

random_sample_xarray_obj(obj, size, seed=None)
    Generates a new, smaller xarray object by randomly sampling from each dimension.

random_sample_dims(obj, size, seed=None)
    Calculates random sampling indices from dimensions of given sizes.

Deparsing and Formatting Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These utilities handle the conversion of xarray objects into valid Python
code strings and the final formatting of the output.

deparse_xarray_variable(var)
    Converts a dictionary representing an xarray Variable into a ``repr`` string.
    Formatted as a valid Python code snippet (e.g., ``"var": (dims, data)``)

deparse_xarray_variables(vars)
    Converts a dictionary representing an xarray Dataset (``xarray.Dataset.to_dict()``)
    into a ``repr`` string representations.

xarray_rrepr_template(xarray_type, data, coords)
    Formats the final, complete string for the minimised xarray object,
    wrapping it in a template that includes coordinates and data variables.

Code Formatting Utility
~~~~~~~~~~~~~~~~~~~~~~

Utility for ensuring consistent code style in the generated output.

ruff_format(code_string)
    Applies code formatting (using the `ruff` formatter) to a generated
    code string to ensure consistent style.

The typical workflow is to patch the representations at the beginning of a
session or script to see simplified outputs.

Examples
--------
>>> import numpy as np
>>> import xarray as xr
>>> rom xarray_rrepr.wrap import rrepr

Now, create a large dataset. Inspecting it will automatically use the
new, minimised representation.

>>> large_ds = xr.Dataset(
...     {
...         "temperature": (["time", "lat", "lon"], np.random.rand(100, 180, 360)),
...         "pressure": (["time", "lat", "lon"], np.random.rand(100, 180, 360)),
...     },
...     coords={
...         "time": np.arange("2020-01-01", "2020-04-10", dtype="datetime64[D]"),
...         "lat": np.linspace(-90, 90, 180),
...         "lon": np.linspace(-180, 180, 360),
...     },
... )

>>> # The output will be a much smaller, random sample of the dataset
>>> print(rrepr(large_ds, seed=42))
xr.Dataset(
    {
        "temperature": (
            ("time", "lat", "lon"),
            np.array([[[0.9, 1.0], [1.0, 0.3]], [[0.3, 0.8], [0.6, 0.9]]]),
        ),
        "pressure": (
            ("time", "lat", "lon"),
            np.array([[[0.4, 0.4], [0.8, 1.0]], [[0.1, 0.9], [0.2, 0.4]]]),
        ),
    },
    coords={
        "time": (
            ("time",),
            np.array(
                [
                    datetime.datetime(2020, 3, 9, 0, 0),
                    datetime.datetime(2020, 1, 11, 0, 0),
                ],
                dtype=object,
            ),
        ),
        "lat": (("lat",), np.array([-25.6, -54.8])),
        "lon": (("lon",), np.array([-55.7, 54.7])),
    },
)

"""

import subprocess

import numpy as np
import pyperclip
import xarray as xr
from numpy.random import MT19937, RandomState, SeedSequence
from numpy.typing import ArrayLike
from xarray.core.common import AbstractArray, DataWithCoords


def rrepr(
    obj: xr.DataArray | xr.Dataset, size: int = 2, seed: dict | None = None
) -> None:
    """Generate a minimised and randomised string representation of the object.

    This method is designed to be monkey-patched onto `xarray.Dataset` and
    `xarray.DataArray` objects. It provides a concise representation that
    captures the structure of a larger object by subsampling its dimensions
    and data. This is particularly useful for unit tests, debugging, logging,
    or displaying examples in documentation about large datasets in a more
    readable format.

    The representation works by:
    1.  Randomly selecting a small, contiguous slice from each dimension.
    2.  Building a new, smaller Dataset/DataArray from this slice.
    3.  Returning the string representation of this new object.
    4.  The original object is **not** modified.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The xarray object instance. This parameter is implicit when the
        method is used as a monkey patch.
    size : int
        The sample size of the subsample.
    seed : int, optional
        Seed for the random number generator to ensure the same minimised
        sample is produced on each call. If ``None`` (default), the
        sampling is non-deterministic.

    Returns
    -------
    str
        A string representing a minimised, randomly sampled version of the
        xarray object.

    Notes
    -----
    This method is intended to be patched as ``__repr__`` or called
    explicitly. When patched as ``__repr__``, it will be used automatically
    when an object is inspected in an interactive console or printed.

    For reproducibility of the random sampling, it's recommended to set
    a random seed before calling the method, e.g., ``np.random.seed(0)``.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> rom xarray_rrepr.wrap import rrepr

    Now, create a large dataset. Inspecting it will automatically use the
    new, minimised representation.

    >>> large_ds = xr.Dataset(
    ...     {
    ...         "temperature": (["time", "lat", "lon"], np.random.rand(100, 180, 360)),
    ...         "pressure": (["time", "lat", "lon"], np.random.rand(100, 180, 360)),
    ...     },
    ...     coords={
    ...         "time": np.arange("2020-01-01", "2020-04-10", dtype="datetime64[D]"),
    ...         "lat": np.linspace(-90, 90, 180),
    ...         "lon": np.linspace(-180, 180, 360),
    ...     },
    ... )

    >>> # The output will be a much smaller, random sample of the dataset
    >>> print(rrepr(large_ds, seed=42))
    xr.Dataset(
        {
            "temperature": (
                ("time", "lat", "lon"),
                np.array([[[0.9, 1.0], [1.0, 0.3]], [[0.3, 0.8], [0.6, 0.9]]]),
            ),
            "pressure": (
                ("time", "lat", "lon"),
                np.array([[[0.4, 0.4], [0.8, 1.0]], [[0.1, 0.9], [0.2, 0.4]]]),
            ),
        },
        coords={
            "time": (
                ("time",),
                np.array(
                    [
                        datetime.datetime(2020, 3, 9, 0, 0),
                        datetime.datetime(2020, 1, 11, 0, 0),
                    ],
                    dtype=object,
                ),
            ),
            "lat": (("lat",), np.array([-25.6, -54.8])),
            "lon": (("lon",), np.array([-55.7, 54.7])),
        },
    )

    """
    resampled_obj = random_sample_xarray_obj(obj, size=size, seed=seed).to_dict()
    coords_expr = deparse_xarray_variables(resampled_obj["coords"])
    if isinstance(obj, xr.DataArray):
        xarray_type = "xr.DataArray"
        data_expr = round_float_array(resampled_obj["data"])
    elif isinstance(obj, xr.Dataset):
        xarray_type = "xr.Dataset"
        data_expr = deparse_xarray_variables(resampled_obj["data_vars"])
    else:
        msg = "Unknown data type"
        raise TypeError(msg)
    code_string = xarray_rrepr_template(xarray_type, data_expr, coords_expr)
    code_string_with_numpy_import = code_string.replace("array", "np.array")
    ruff_formatted_code_string = ruff_format(code_string_with_numpy_import)
    pyperclip.copy(ruff_formatted_code_string)
    return ruff_formatted_code_string


def random_sample_dims(
    obj: xr.DataArray | xr.Dataset, size: int, seed: int | None
) -> ArrayLike:
    """Calculate random sampling indices from dimensions of given sizes."""
    rng = RandomState(MT19937(SeedSequence(seed)))
    max_id = np.array(list(obj.sizes.values()))[..., np.newaxis]
    return rng.randint(max_id, size=(len(obj.sizes), size))


def random_sample_xarray_obj(
    obj: xr.DataArray | xr.Dataset, size: int, seed: int | None = None
) -> xr.DataArray | xr.Dataset:
    """Generate a new, smaller xarray object by randomly sampling from each
    dimension.
    """  # noqa: D205
    indices = random_sample_dims(obj=obj, size=size, seed=seed)
    indices_dict = dict(zip(obj.sizes.keys(), indices, strict=True))
    return obj.isel(indices_dict)


def deparse_xarray_variable(variable: dict) -> DataWithCoords:
    """Convert a dictionary representing an xarray Variable into a ``repr``
    string. Formatted as a valid Python code snippet (e.g., ``"var":
    (dims, data)``).
    """  # noqa: D205
    var_data = round_float_array(variable["data"])
    return (variable["dims"], var_data)


def round_float_array(arr: ArrayLike, ndigits: int = 1) -> ArrayLike:
    """Round numpy array with float values."""
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        return np.round(arr, ndigits)
    return arr


def deparse_xarray_variables(variables: dict) -> DataWithCoords:
    """Convert a dictionary representing an xarray Dataset
    (``xarray.Dataset.to_dict()``) into a ``repr`` string representations.
    """  # noqa: D205
    return {name: deparse_xarray_variable(var) for name, var in variables.items()}


def xarray_rrepr_template(
    xarray_type: str, data: AbstractArray | DataWithCoords, coords: DataWithCoords
) -> str:
    """Format the final, complete string for the minimised xarray object,
    wrapping it in a template that includes coordinates and data variables.
    """  # noqa: D205
    return f"{xarray_type}({data!r}, coords={coords!r})"


def ruff_format(code_string: str) -> str:
    """Apply code formatting (using the `ruff` formatter) to a generated
    code string to ensure consistent style.
    """  # noqa: D205
    result = subprocess.run(
        [  # noqa: S607
            "ruff",
            "format",
            "-",
        ],
        check=False,
        input=code_string,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout
