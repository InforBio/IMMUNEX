#!/usr/bin/env python
import resource
import os
import re
import cv2
import time
import logging
import tifffile
import importlib
import itertools
import numpy as np
import scanpy as sc
from tqdm import tqdm
import bin2cell as b2c
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from csbdeep.utils import normalize
from skimage.filters import threshold_otsu
import pandas as pd
import scanpy as sc
import scipy.spatial
import scipy.sparse
import scipy.stats
import anndata as ad
import skimage
import scanpy as sc
import numpy as np
import os
from scanpy import read_10x_h5
from scanpy import logging as logg
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from copy import deepcopy

def sample_regions_from_adata(
    adata,
    n_regions: int = 10,
    area_min: int = 100,
    area_max: int = 1000,
    max_tries: int = 10000,
    seed: int = 42,
    min_side: int = 5,
    min_bins_present: int = 100,
):
    rng = np.random.default_rng(seed)
    rows = adata.obs['array_row'].astype(int).to_numpy()
    cols = adata.obs['array_col'].astype(int).to_numpy()

    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()
    grid_h = (rmax - rmin + 1)
    grid_w = (cmax - cmin + 1)

    # feasible side lengths s so s^2 in [area_min, area_max] and fits grid
    s_low_area  = int(np.ceil(np.sqrt(max(area_min, 1))))
    s_high_area = int(np.floor(np.sqrt(max(area_max, 1))))
    side_low  = max(min_side, s_low_area, 1)
    side_high = min(s_high_area, grid_w, grid_h)
    if side_low > side_high:
        raise ValueError(
            f"No feasible square size: side_low={side_low} > side_high={side_high} "
            f"(grid {grid_w}x{grid_h}, area_min={area_min}, area_max={area_max})."
        )

    regions, details, seen = [], [], set()
    tries = 0
    while len(regions) < n_regions and tries < max_tries:
        tries += 1
        s = int(rng.integers(side_low, side_high + 1))
        if grid_w < s or grid_h < s:
            continue

        x = int(rng.integers(cmin, cmax - s + 2))  # [x, x+s)
        y = int(rng.integers(rmin, rmax - s + 2))  # [y, y+s)
        key = (x, y, s)
        if key in seen:
            continue

        sel = (
            (adata.obs['array_row'] >= y) & (adata.obs['array_row'] < y + s) &
            (adata.obs['array_col'] >= x) & (adata.obs['array_col'] < x + s)
        )
        n_in = int(sel.sum())
        if n_in < min_bins_present:
            continue

        seen.add(key)
        regions.append({"x": x, "y": y, "w": s, "h": s})
        details.append({
            "idx": len(regions), "x": x, "y": y, "w": s, "h": s,
            "area_bins": s * s, "n_bins_present": n_in
        })

    if len(regions) < n_regions:
        print(f"Warning: only found {len(regions)} regions after {tries} attempts "
              f"(feasible sides: {side_low}..{side_high}).")

    return pd.DataFrame(details)



def load_or_make_regions_table(adata, table_path: Path,
                               n_regions=10, area_min=100, area_max=1000,
                               min_bins_present=100, seed=42):
    table_path = Path(table_path)
    if table_path.exists():
        # reuse existing coordinates (DO NOT regenerate)
        df = pd.read_csv(table_path)
        # ensure required columns and dtypes
        need = {"x","y","w","h"}
        if not need.issubset(df.columns):
            raise ValueError(f"Existing table {table_path} is missing columns {need - set(df.columns)}.")
        for col in ["x","y","w","h","idx","area_bins","n_bins_present"]:
            if col in df.columns:
                df[col] = df[col].astype(int, errors="ignore")
        print(f"[coords] loaded {len(df)} regions from {table_path}")
        return df
    else:
        # generate new, then save
        df = sample_regions_from_adata(
            adata,
            n_regions=n_regions,
            area_min=area_min,
            area_max=area_max,
            seed=seed,
            min_bins_present=min_bins_present
        )
        # ensure index and columns order
        cols = ["idx","x","y","w","h","area_bins","n_bins_present"]
        df[cols].to_csv(table_path, index=False)
        print(f"[coords] generated & saved {len(df)} regions -> {table_path}")
        return df





def region_mask(adata, reg, inclusive_right: bool = False):

    if not {'array_row', 'array_col'}.issubset(adata.obs.columns):
        raise KeyError("Expected obs columns 'array_row' and 'array_col'.")

    # Make sure coords are numeric arrays
    col = pd.to_numeric(adata.obs['array_col'], errors='coerce').to_numpy()
    row = pd.to_numeric(adata.obs['array_row'], errors='coerce').to_numpy()

    x = float(reg["x"]); y = float(reg["y"])
    w = float(reg["w"]); h = float(reg["h"])

    if inclusive_right:
        sel = (row >= y) & (row <= y + h) & (col >= x) & (col <= x + w)
    else:
        # half-open on right/bottom avoids double counting adjacent tiles
        sel = (row >= y) & (row <  y + h) & (col >= x) & (col <  x + w)

    return sel.astype(bool)




def save_hist(x, title, xlabel, vline=None, output_base_path = '.', fname="plot.png", bins=100, logx=False):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    fig, ax = plt.subplots(figsize=(7,4))
    # for log x, build log-spaced bins to avoid distortion
    if logx:
        x_pos = x[x > 0]
        if x_pos.size:
            bmin, bmax = np.percentile(x_pos, [0.5, 99.5])
            bins_edges = np.logspace(np.log10(max(bmin, 1e-6)), np.log10(bmax), bins)
            ax.hist(x_pos, bins=bins_edges)
            ax.set_xscale('log')
        else:
            ax.hist(x, bins=bins)
    else:
        ax.hist(x, bins=bins)
    if vline is not None:
        ax.axvline(vline, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Bins")
    fig.tight_layout()
    fig.savefig(os.path.join(output_base_path, fname), dpi=150)
    plt.close(fig)

    print(f"\t \t >> >> >> Hist. exported to {output_base_path} ")

def bin_to_cell(adata, labels_key="labels_expanded", spatial_keys=["spatial"], diameter_scale_factor=None):
    '''
    Collapse all bins for a given nonzero ``labels_key`` into a single cell. 
    Gene expression added up, array coordinates and ``spatial_keys`` averaged out. 
    ``"spot_diameter_fullres"`` in the scale factors multiplied by 
    ``diameter_scale_factor`` to reflect increased unit size. Returns cell level AnnData, 
    including ``.obs["bin_count"]`` reporting how many bins went into creating the cell.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw or destriped counts. Needs ``labels_key`` in ``.obs`` 
        and ``spatial_keys`` in ``.obsm``.
    labels_key : ``str``, optional (default: ``"labels_expanded"``)
        Which ``.obs`` key to use for grouping 2um bins into cells. Integers, with 0 being 
        unassigned to an object. If an extra ``"_source"`` column is detected as a result 
        of ``b2c.salvage_secondary_labels()`` calling, its info will be propagated per 
        label.
    spatial_keys : list of ``str``, optional (default: ``["spatial"]``)
        Which ``.obsm`` keys to average out across all bins falling into a cell to get a 
        cell's respective spatial coordinates.
    diameter_scale_factor : ``float`` or ``None``, optional (default: ``None``)
        The object's ``"spot_diameter_fullres"`` will be multiplied by this much to reflect 
        the change in unit per observation. If ``None``, will default to the square root of 
        the mean of the per-cell bin counts.
    '''

    
    print('adata shape ')
    print(adata.shape)
    
    #a label of 0 means there's nothing there, ditch those bins from this operation
    adata = adata[adata.obs[labels_key]!=0]
    adata.obs[labels_key] = adata.obs[labels_key].astype('str') 
    adata.obs[labels_key] = 'HE' + adata.obs[labels_key] 
    labels = adata.obs[labels_key]
    print('adata shape filtred')
    print(adata.shape)
    
    #use the newly inserted labels to make pandas dummies, as sparse because the data is huge
    cell_to_bin = pd.get_dummies(adata.obs[labels_key], sparse=True)
    print('dummies')
    print(cell_to_bin)
    
    #take a quick detour to save the cell labels as they appear in the dummies
    #they're likely to be integers, make them strings to avoid complications in the downstream AnnData
    cell_names = [str(i) for i in cell_to_bin.columns]
    print('cell_names to str')
    print(cell_names[:10])
    
    # Get the actual unique label values from the filtered adata
    label_values = adata.obs[labels_key].unique()
    # Use these as obs_names (as strings)
    cell_names = [str(i) for i in label_values]
    print('fixed cell names')
    print(cell_names[:10])


    #then pull out the actual internal sparse matrix (.sparse) as a scipy COO one, turn to CSR
    #this has bins as rows, transpose so cells are as rows (and CSR becomes CSC for .dot())
    cell_to_bin = cell_to_bin.sparse.to_coo().tocsr().T
    #can now generate the cell expression matrix by adding up the bins (via matrix multiplication)
    #cell-bin * bin-gene = cell-gene
    #(turn it to CSR at the end as somehow it comes out CSC)
    X = cell_to_bin.dot(adata.X).tocsr()
    #create object, stash stuff
    cell_adata = ad.AnnData(X, var = adata.var)
    cell_adata.obs_names = cell_names
    #turn the cell names back to int and stash that as metadata too
    cell_adata.obs['object_id'] = [int(i.replace('HE','')) for i in cell_names]
    cell_adata.obs['labels_he'] = cell_names

    print(cell_adata.obs.head())
    #need to bust out deepcopy here as otherwise altering the spot diameter gets back-propagated
    cell_adata.uns['spatial'] = deepcopy(adata.uns['spatial'])
    #getting the centroids (means of bin coords) involves computing a mean of each cell_to_bin row
    #premultiplying by a diagonal matrix multiplies each row by a value: https://solitaryroad.com/c108.html
    #use that to divide each row by it sum (.sum(axis=1)), then matrix multiply the result by bin coords
    #stash the sum into a separate variable for subsequent object storage
    #cell-cell * cell-bin * bin-coord = cell-coord
    bin_count = np.asarray(cell_to_bin.sum(axis=1)).flatten()
    row_means = scipy.sparse.diags(1/bin_count)
    cell_adata.obs['bin_count'] = bin_count
    #take the thing out for a spin with array coordinates
    cell_adata.obs["array_row"] = row_means.dot(cell_to_bin).dot(adata.obs["array_row"].values)
    cell_adata.obs["array_col"] = row_means.dot(cell_to_bin).dot(adata.obs["array_col"].values)
    #generate the various spatial coordinate systems
    #just in case a single is passed as a string
    if type(spatial_keys) is not list:
        spatial_keys = [spatial_keys]
    for spatial_key in spatial_keys:
        cell_adata.obsm[spatial_key] = row_means.dot(cell_to_bin).dot(adata.obsm[spatial_key])
    #of note, the default scale factor bin diameter at 2um resolution stops rendering sensibly in plots
    #by default estimate it as the sqrt of the bin count mean
    if diameter_scale_factor is None:
        diameter_scale_factor = np.sqrt(np.mean(bin_count))
    #bump it up to something a bit more sensible
    library = list(adata.uns['spatial'].keys())[0]
    cell_adata.uns['spatial'][library]['scalefactors']['spot_diameter_fullres'] *= diameter_scale_factor
    
    return cell_adata

    






def fit_array_to_pixel_transform(arr_col, arr_row, px_x, px_y, valid=None):
    """
    Fit 1D affine maps:
        x_px ≈ ax * array_col + bx
        y_px ≈ ay * array_row + by
    Returns (ax, bx, ay, by)
    """
    arr_col = np.asarray(arr_col, float)
    arr_row = np.asarray(arr_row, float)
    px_x    = np.asarray(px_x, float)
    px_y    = np.asarray(px_y, float)

    if valid is None:
        valid = np.ones_like(arr_col, dtype=bool)
    else:
        valid = np.asarray(valid, bool)

    # Robust 1st-order polyfit on valid points
    ax, bx = np.polyfit(arr_col[valid], px_x[valid], 1)
    ay, by = np.polyfit(arr_row[valid], px_y[valid], 1)
    return ax, bx, ay, by


def crop_image_by_region(img, reg, ax, bx, ay, by, pad_px=0):
    """
    Crop a rectangle defined in array_col/array_row units from image pixels.
    reg: dict with x (col), y (row), w, h  (half-open on right/bottom)
    Mapping: x_px = ax*col + bx ; y_px = ay*row + by
    """
    x, y, w, h = int(reg["x"]), int(reg["y"]), int(reg["w"]), int(reg["h"])

    # half-open → pixel box [x0:x1, y0:y1)
    x0 = int(np.floor(ax * x       + bx)) - pad_px
    y0 = int(np.floor(ay * y       + by)) - pad_px
    x1 = int(np.ceil (ax * (x + w) + bx)) + pad_px
    y1 = int(np.ceil (ay * (y + h) + by)) + pad_px

    H, W = img.shape[:2]
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(x0 + 1, min(W, x1))
    y1 = max(y0 + 1, min(H, y1))

    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def draw_rect(ax, box, **kw):
    """Quick debug rectangle on an axis given (x0, y0, x1, y1)."""
    import matplotlib.patches as patches
    x0, y0, x1, y1 = box
    rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, **kw)
    ax.add_patch(rect)
    return rect

    import imageio.v2 as iio

import imageio.v2 as iio

def to_uint8(img):
    if img.dtype == np.uint8:
        return img
    arr = img
    if np.issubdtype(arr.dtype, np.floating):
        # if in [0,1], scale up; else clip to 0..255
        if np.nanmax(arr) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255)
    else:
        arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)

def save_png(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(path), to_uint8(img))






def bin_to_cell(adata, labels_key="labels_expanded", spatial_keys=["spatial"], diameter_scale_factor=None):
    """
    Collapse all bins sharing the same nonzero `labels_key` into a single cell:
    - sums gene counts,
    - averages coordinates for keys in `spatial_keys`,
    - scales `spot_diameter_fullres`,
    - adds `.obs['bin_count']` (bins per cell),
    - **adds `.obs['bins_2um']`**: the list of original bin obs_names per cell.
    """
    # --- keep only labeled bins (>0) ---
    adata = adata[adata.obs[labels_key] > 0].copy()
    adata.obs_names = adata.obs_names.astype(str)

    # --- normalize labels and prefix with 'HE' to match your convention ---
    adata.obs[labels_key] = adata.obs[labels_key].astype(str)
    adata.obs[labels_key] = 'HE' + adata.obs[labels_key]

    # --- bin→cell one-hot (sparse) and keep column order ---
    cell_to_bin_df = pd.get_dummies(adata.obs[labels_key], sparse=True)
    cell_names = cell_to_bin_df.columns.astype(str).tolist()

    # --- to CSR with cells as rows ---
    cell_to_bin = cell_to_bin_df.sparse.to_coo().tocsr().T  # (cells × bins)

    # --- aggregate expression (sum over bins) ---
    X = cell_to_bin.dot(adata.X).tocsr() if scipy.sparse.issparse(adata.X) else cell_to_bin.dot(adata.X)

    # --- create cell-level AnnData ---
    cell_adata = ad.AnnData(X, var=adata.var)
    cell_adata.obs_names = cell_names
    cell_adata.obs['object_id'] = [int(n.replace('HE', '')) for n in cell_names]
    cell_adata.obs['labels_he'] = cell_names

    # --- copy spatial meta ---
    cell_adata.uns['spatial'] = deepcopy(adata.uns['spatial'])

    # --- per-cell bin count and mean coords ---
    bin_count = np.asarray(cell_to_bin.sum(axis=1)).ravel()
    cell_adata.obs['bin_count'] = bin_count
    row_means = scipy.sparse.diags(1.0 / bin_count)

    cell_adata.obs['array_row'] = row_means.dot(cell_to_bin).dot(adata.obs['array_row'].values)
    cell_adata.obs['array_col'] = row_means.dot(cell_to_bin).dot(adata.obs['array_col'].values)

    # --- average additional spatial keys ---
    if not isinstance(spatial_keys, list):
        spatial_keys = [spatial_keys]
    for sk in spatial_keys:
        cell_adata.obsm[sk] = row_means.dot(cell_to_bin).dot(adata.obsm[sk])

    # === NEW: attach list of original bin ids per cell in `.obs['bins_2um']` ===
    # group bins by the (already 'HE'-prefixed) label, then align to cell_adata.obs_names
    bins_by_label = (
        adata.obs
        .groupby(labels_key, sort=False)
        .apply(lambda df: df.index.tolist())
        .to_dict()
    )
    cell_adata.obs['bins_2um'] = pd.Series(
        [bins_by_label.get(name, []) for name in cell_adata.obs_names],
        index=cell_adata.obs_names,
        dtype=object
    )

    # --- scale spot diameter for rendering ---
    if diameter_scale_factor is None:
        diameter_scale_factor = float(np.sqrt(np.mean(bin_count))) if len(bin_count) else 1.0
    library = list(adata.uns['spatial'].keys())[0]
    cell_adata.uns['spatial'][library]['scalefactors']['spot_diameter_fullres'] *= diameter_scale_factor

    return cell_adata
