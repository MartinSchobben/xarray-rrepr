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
```

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
                    [[[299.1, 295.3], [296.3, 294.1]], [[299.3, 296.4], [295.3, 294.7]]]
                ),
            )
        },
        coords={
            "lat": (("lat",), np.array([20.0, 27.5])),
            "lon": (("lon",), np.array([292.5, 317.5])),
            "time": (
                ("time",),
                np.array(
                    [
                        datetime.datetime(2014, 3, 6, 0, 0),
                        datetime.datetime(2014, 2, 18, 18, 0),
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

<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in notebooks */
&#10;:root {
  --xr-font-color0: var(
    --jp-content-font-color0,
    var(--pst-color-text-base rgba(0, 0, 0, 1))
  );
  --xr-font-color2: var(
    --jp-content-font-color2,
    var(--pst-color-text-base, rgba(0, 0, 0, 0.54))
  );
  --xr-font-color3: var(
    --jp-content-font-color3,
    var(--pst-color-text-base, rgba(0, 0, 0, 0.38))
  );
  --xr-border-color: var(
    --jp-border-color2,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 10))
  );
  --xr-disabled-color: var(
    --jp-layout-color3,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 40))
  );
  --xr-background-color: var(
    --jp-layout-color0,
    var(--pst-color-on-background, white)
  );
  --xr-background-color-row-even: var(
    --jp-layout-color1,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 5))
  );
  --xr-background-color-row-odd: var(
    --jp-layout-color2,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 15))
  );
}
&#10;html[theme="dark"],
html[data-theme="dark"],
body[data-theme="dark"],
body.vscode-dark {
  --xr-font-color0: var(
    --jp-content-font-color0,
    var(--pst-color-text-base, rgba(255, 255, 255, 1))
  );
  --xr-font-color2: var(
    --jp-content-font-color2,
    var(--pst-color-text-base, rgba(255, 255, 255, 0.54))
  );
  --xr-font-color3: var(
    --jp-content-font-color3,
    var(--pst-color-text-base, rgba(255, 255, 255, 0.38))
  );
  --xr-border-color: var(
    --jp-border-color2,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 10))
  );
  --xr-disabled-color: var(
    --jp-layout-color3,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 40))
  );
  --xr-background-color: var(
    --jp-layout-color0,
    var(--pst-color-on-background, #111111)
  );
  --xr-background-color-row-even: var(
    --jp-layout-color1,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 5))
  );
  --xr-background-color-row-odd: var(
    --jp-layout-color2,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 15))
  );
}
&#10;.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
  line-height: 1.6;
}
&#10;.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}
&#10;.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}
&#10;.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}
&#10;.xr-obj-type,
.xr-obj-name,
.xr-group-name {
  margin-left: 2px;
  margin-right: 10px;
}
&#10;.xr-group-name::before {
  content: "📁";
  padding-right: 0.3em;
}
&#10;.xr-group-name,
.xr-obj-type {
  color: var(--xr-font-color2);
}
&#10;.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
  margin-block-start: 0;
  margin-block-end: 0;
}
&#10;.xr-section-item {
  display: contents;
}
&#10;.xr-section-item input {
  display: inline-block;
  opacity: 0;
  height: 0;
  margin: 0;
}
&#10;.xr-section-item input + label {
  color: var(--xr-disabled-color);
  border: 2px solid transparent !important;
}
&#10;.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}
&#10;.xr-section-item input:focus + label {
  border: 2px solid var(--xr-font-color0) !important;
}
&#10;.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}
&#10;.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}
&#10;.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}
&#10;.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}
&#10;.xr-section-summary-in + label:before {
  display: inline-block;
  content: "►";
  font-size: 11px;
  width: 15px;
  text-align: center;
}
&#10;.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}
&#10;.xr-section-summary-in:checked + label:before {
  content: "▼";
}
&#10;.xr-section-summary-in:checked + label > span {
  display: none;
}
&#10;.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
}
&#10;.xr-section-inline-details {
  grid-column: 2 / -1;
}
&#10;.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-top: 4px;
  margin-bottom: 5px;
}
&#10;.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}
&#10;.xr-group-box {
  display: inline-grid;
  grid-template-columns: 0px 20px auto;
  width: 100%;
}
&#10;.xr-group-box-vline {
  grid-column-start: 1;
  border-right: 0.2em solid;
  border-color: var(--xr-border-color);
  width: 0px;
}
&#10;.xr-group-box-hline {
  grid-column-start: 2;
  grid-row-start: 1;
  height: 1em;
  width: 20px;
  border-bottom: 0.2em solid;
  border-color: var(--xr-border-color);
}
&#10;.xr-group-box-contents {
  grid-column-start: 3;
}
&#10;.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}
&#10;.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}
&#10;.xr-preview {
  color: var(--xr-font-color3);
}
&#10;.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}
&#10;.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}
&#10;.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}
&#10;.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}
&#10;.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}
&#10;.xr-dim-list:before {
  content: "(";
}
&#10;.xr-dim-list:after {
  content: ")";
}
&#10;.xr-dim-list li:not(:last-child):after {
  content: ",";
  padding-right: 5px;
}
&#10;.xr-has-index {
  font-weight: bold;
}
&#10;.xr-var-list,
.xr-var-item {
  display: contents;
}
&#10;.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  border-color: var(--xr-background-color-row-odd);
  margin-bottom: 0;
  padding-top: 2px;
}
&#10;.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}
&#10;.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
  border-color: var(--xr-background-color-row-even);
}
&#10;.xr-var-name {
  grid-column: 1;
}
&#10;.xr-var-dims {
  grid-column: 2;
}
&#10;.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}
&#10;.xr-var-preview {
  grid-column: 4;
}
&#10;.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}
&#10;.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}
&#10;.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}
&#10;.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  border-top: 2px dotted var(--xr-background-color);
  padding-bottom: 20px !important;
  padding-top: 10px !important;
}
&#10;.xr-var-attrs-in + label,
.xr-var-data-in + label,
.xr-index-data-in + label {
  padding: 0 1px;
}
&#10;.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}
&#10;.xr-var-data > table {
  float: right;
}
&#10;.xr-var-data > pre,
.xr-index-data > pre,
.xr-var-data > table > tbody > tr {
  background-color: transparent !important;
}
&#10;.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}
&#10;.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}
&#10;dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}
&#10;.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}
&#10;.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}
&#10;.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}
&#10;.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}
&#10;.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
&#10;.xr-var-attrs-in:checked + label > .xr-icon-file-text2,
.xr-var-data-in:checked + label > .xr-icon-database,
.xr-index-data-in:checked + label > .xr-icon-database {
  color: var(--xr-font-color0);
  filter: drop-shadow(1px 1px 5px var(--xr-font-color2));
  stroke-width: 0.8px;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt; Size: 112B
Dimensions:  (time: 2, lat: 2, lon: 2)
Coordinates:
  * time     (time) datetime64[ns] 16B 2014-07-31T06:00:00 2013-10-08T06:00:00
  * lat      (lat) float64 16B 70.0 32.5
  * lon      (lon) float64 16B 285.0 260.0
Data variables:
    air      (time, lat, lon) float64 64B 283.2 285.2 299.8 ... 298.2 289.5</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-cf70650e-8e7b-4383-a2be-45db2e8c9af2' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-cf70650e-8e7b-4383-a2be-45db2e8c9af2' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>time</span>: 2</li><li><span class='xr-has-index'>lat</span>: 2</li><li><span class='xr-has-index'>lon</span>: 2</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-12571934-6dbc-4125-a42d-5853bf789ef2' class='xr-section-summary-in' type='checkbox'  checked><label for='section-12571934-6dbc-4125-a42d-5853bf789ef2' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2014-07-31T06:00:00 2013-10-08T0...</div><input id='attrs-d5eca2ee-664a-4731-89c5-e8bf2f72dd3f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d5eca2ee-664a-4731-89c5-e8bf2f72dd3f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f39271d7-c6ac-4034-be81-7ae7bbd2bba9' class='xr-var-data-in' type='checkbox'><label for='data-f39271d7-c6ac-4034-be81-7ae7bbd2bba9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2014-07-31T06:00:00.000000000&#x27;, &#x27;2013-10-08T06:00:00.000000000&#x27;],
      dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>lat</span></div><div class='xr-var-dims'>(lat)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>70.0 32.5</div><input id='attrs-99abfdb4-f63d-485c-baec-e3e998127f6c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-99abfdb4-f63d-485c-baec-e3e998127f6c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2e16432a-0059-4d7e-97c9-fcdfe7d09028' class='xr-var-data-in' type='checkbox'><label for='data-2e16432a-0059-4d7e-97c9-fcdfe7d09028' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([70. , 32.5])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>lon</span></div><div class='xr-var-dims'>(lon)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>285.0 260.0</div><input id='attrs-a93edff8-c27c-4d64-99fb-fbcca269782a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a93edff8-c27c-4d64-99fb-fbcca269782a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-61f6f9ca-debd-4b86-a876-b9b6ddec7de5' class='xr-var-data-in' type='checkbox'><label for='data-61f6f9ca-debd-4b86-a876-b9b6ddec7de5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([285., 260.])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-700643c7-fa43-4863-8a72-eec2cb7b11ac' class='xr-section-summary-in' type='checkbox'  checked><label for='section-700643c7-fa43-4863-8a72-eec2cb7b11ac' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>air</span></div><div class='xr-var-dims'>(time, lat, lon)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>283.2 285.2 299.8 ... 298.2 289.5</div><input id='attrs-c51171a8-542b-43a9-9c75-0353ea60c68c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c51171a8-542b-43a9-9c75-0353ea60c68c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3ad560b1-8ba8-40a9-8b6b-91a9f339a790' class='xr-var-data-in' type='checkbox'><label for='data-3ad560b1-8ba8-40a9-8b6b-91a9f339a790' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[283.2, 285.2],
        [299.8, 299.6]],
&#10;       [[264.5, 270.2],
        [298.2, 289.5]]])</pre></div></li></ul></div></li></ul></div></div>

Which can’t be done with standard `repr`.

``` python
eval(repr(ds))
```

    SyntaxError: invalid decimal literal (<string>, line 1)
    Traceback [36m(most recent call last)[39m:

      File [92m~/Documents/work/projects/xarray-rrepr/.venv/lib/python3.13/site-packages/IPython/core/interactiveshell.py:3701[39m in [95mrun_code[39m
        exec(code_obj, self.user_global_ns, self.user_ns)

    [36m  [39m[36mCell[39m[36m [39m[32mIn[13][39m[32m, line 1[39m
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
    R * lat      (lat) float64 16B 72.5 70.0
    L * time     (time) datetime64[ns] 23kB 2013-01-01 ... 2014-12-31T18:00:00
    R * time     (time) datetime64[ns] 16B 2014-05-13T12:00:00 2014-08-30
    L * lon      (lon) float32 212B 200.0 202.5 205.0 207.5 ... 325.0 327.5 330.0
    R * lon      (lon) float64 16B 272.5 207.5
    Differing data variables:
    L   air      (time, lat, lon) float64 31MB 241.2 242.5 243.5 ... 296.2 295.7
    R   air      (time, lat, lon) float64 64B 264.3 275.1 265.1 ... 280.5 274.9
    [31m---------------------------------------------------------------------------[39m
    [31mAssertionError[39m                            Traceback (most recent call last)
    [36mCell[39m[36m [39m[32mIn[14][39m[32m, line 1[39m
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
    R * lat      (lat) float64 16B 72.5 70.0
    L * time     (time) datetime64[ns] 23kB 2013-01-01 ... 2014-12-31T18:00:00
    R * time     (time) datetime64[ns] 16B 2014-05-13T12:00:00 2014-08-30
    L * lon      (lon) float32 212B 200.0 202.5 205.0 207.5 ... 325.0 327.5 330.0
    R * lon      (lon) float64 16B 272.5 207.5
    Differing data variables:
    L   air      (time, lat, lon) float64 31MB 241.2 242.5 243.5 ... 296.2 295.7
    R   air      (time, lat, lon) float64 64B 264.3 275.1 265.1 ... 280.5 274.9
