import subprocess

import numpy as np
import pyperclip
import xarray as xr


def rrepr(obj: xr.DataArray, size: int = 2, seed=None) -> None:
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


def drop_all_attrs_array(da: xr.DataArray) -> xr.DataArray:
    return da.drop_attrs()


def drop_all_attrs_dataset(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.drop_attrs()
    for v in ds.variables:
        ds[v] = drop_all_attrs_array(ds[v])
    return ds


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
