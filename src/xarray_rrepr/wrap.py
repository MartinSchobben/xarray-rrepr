import subprocess

import numpy as np
import pyperclip
import xarray as xr


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
    First, let's monkey patch the ``__repr__`` method of an `xarray.Dataset`.

    >>> import numpy as np
    >>> import xarray as xr
    >>> from your_module import _repr_minimised
    >>> xr.Dataset.__repr__ = _repr_minimised

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
        raise TypeError("Unknown data type")
    code_string = xarray_rrepr_template(xarray_type, data_expr, coords_expr)
    code_string_with_numpy_import = code_string.replace("array", "np.array")
    ruff_formatted_code_string = ruff_format(code_string_with_numpy_import)
    pyperclip.copy(ruff_formatted_code_string)
    return ruff_formatted_code_string


def random_sample_dims(obj, dim, size, seed):
    rng = np.random.default_rng(seed=seed)
    return rng.choice(np.arange(obj.sizes[dim]), size=size, replace=False)


def random_sample_xarray_obj(
    obj: xr.DataArray | xr.Dataset, size: int = 2, seed=None
) -> xr.DataArray | xr.Dataset:
    indices = {
        dim: random_sample_dims(
            obj=obj, dim=dim, size=size, seed=n + seed if seed else None
        )
        for n, dim in enumerate(obj.sizes.keys())
    }
    return obj.isel(indices)


def deparse_xarray_variable(var: dict):
    var_data = round_float_array(var["data"])
    return (var["dims"], var_data)


def round_float_array(arr, ndigits=1):
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        return np.round(arr, ndigits)
    return arr


def deparse_xarray_variables(vars: dict):
    return {name: deparse_xarray_variable(var) for name, var in vars.items()}


def xarray_rrepr_template(xarray_type, data, coords):
    return f"{xarray_type}({data!r}, coords={coords!r})"


def ruff_format(code_string):
    result = subprocess.run(
        [
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
