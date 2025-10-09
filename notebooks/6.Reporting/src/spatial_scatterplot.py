# ---------- Basic sanity ----------
needed = {"array_row","array_col"}
missing = needed - set(df.columns)
if missing:
    raise KeyError(f"Missing columns: {missing}")

# ---------- Columns ----------
res_cols = [c for c in df.columns if c.startswith("resolution_leiden_")]
if not res_cols:
    raise ValueError("No 'resolution_leiden_*' columns found.")
qc_cols = [c for c in [
    "n_genes_by_counts","total_counts","log1p_total_counts",
    "pct_counts_mt","counts_per_bin","genes_per_bin"
] if c in df.columns]

# add pct_in_cluster if present (so you can color by it)
extra_cols = []
if "pct_in_cluster" in df.columns:
    extra_cols.append("pct_in_cluster")

color_options = res_cols + qc_cols + extra_cols

# choose default resolution = resolution_leiden_0.5 if available
default_res = next((c for c in res_cols if c == "resolution_leiden_0.5"), res_cols[0])

# ---------- Palettes ----------
import matplotlib as mpl, numpy as np, matplotlib.pyplot as plt, ipywidgets as W, pandas as pd
from ipywidgets import Layout
cmaps_cat = ['tab10','tab20','Set1','Set2','Set3','Pastel1','Pastel2','Dark2','Accent','Paired','tab20b','tab20c']
cmaps_cont = ['viridis','plasma','inferno','magma','cividis','Spectral','coolwarm','bwr','seismic','RdBu_r','PiYG','PRGn','BrBG','rainbow','nipy_spectral','Greys','Blues','Greens','Oranges','Reds']
cmap_all = cmaps_cat + cmaps_cont

# ---------- Widgets ----------
# default color-by: prefer pct_in_cluster if present, else default_res
default_color = "pct_in_cluster" if "pct_in_cluster" in color_options else default_res

color_dd     = W.Dropdown(options=color_options, value=default_color, description="Color by")
cmap_dd      = W.Dropdown(options=cmap_all, value='tab20', description="Colormap")
cluster_dd   = W.Dropdown(options=res_cols, value=default_res, description="Clusters for summaries")
hide_smallcb = W.Checkbox(value=True, description="Hide small (<0.5%)")
size_sl      = W.IntSlider(value=2, min=1, max=6, step=1, description="Size", continuous_update=False)

# Max pts: wide slider + synced numeric box
max_allowed  = int(len(df))
limit_sl     = W.IntSlider(value=min(len(df), 200_000),
                           min=10_000, max=max_allowed, step=5_000,
                           description="Max pts", continuous_update=False,
                           layout=Layout(width='450px'))
limit_box    = W.BoundedIntText(value=limit_sl.value, min=10_000, max=max_allowed, step=1000, description='')
W.jslink((limit_sl, 'value'), (limit_box, 'value'))

alpha_sl     = W.FloatSlider(value=0.7, min=0.1, max=1.0, step=0.05, description="Alpha", readout_format=".2f", continuous_update=False)
flip_y_cb    = W.Checkbox(value=True,  description="Invert Y")
flip_x_cb    = W.Checkbox(value=True,  description="Invert X")

# --- Cluster selector + clear button ---
cluster_label = W.HTML("<b>Show clusters:</b>")
cluster_filter_box = W.Box(layout=Layout(display='flex', flex_flow='row wrap',
                                         overflow_x='auto', overflow_y='auto',
                                         border='1px solid #ddd', padding='4px',
                                         max_height='80px', gap='6px'))

btn_clear = W.Button(description="Unselect all", tooltip="Clear all cluster checkboxes",
                     layout=Layout(width='120px', margin='0 0 0 8px'))

# Header row for the cluster box
cluster_header = W.HBox([cluster_label, btn_clear], layout=Layout(align_items='center'))

controls_row  = W.HBox([color_dd, cmap_dd, cluster_dd, hide_smallcb])
controls_row2 = W.HBox([size_sl, alpha_sl, W.HBox([limit_sl, limit_box], layout=Layout(align_items='center')), flip_y_cb, flip_x_cb])
controls_row3 = W.VBox([cluster_header, cluster_filter_box])

# ---------- Outputs ----------
out_scatter = W.Output()
out_counts  = W.Output()
out_box     = W.Output()

# ---- Global extents ----
XMIN, XMAX = float(df["array_col"].min()), float(df["array_col"].max())
YMIN, YMAX = float(df["array_row"].min()), float(df["array_row"].max())
_padx = 0.01 * (XMAX - XMIN) or 1.0
_pady = 0.01 * (YMAX - YMIN) or 1.0

# ---- Helpers ----
_RENDER_LOCK = False  # prevents re-render storms when toggling many checkboxes at once

def _cat_colors(categories, cmap_name):
    cmap = mpl.colormaps.get_cmap(cmap_name)
    N = max(len(categories), 1)
    xs = np.linspace(0, 1, N, endpoint=False)
    return [cmap(x) for x in xs]

def _get_selected_from_box():
    return {cb.description for cb in cluster_filter_box.children if isinstance(cb, W.Checkbox) and cb.value}

def _set_box_from_list(names):
    # build fresh checkboxes (default = selected)
    cluster_filter_box.children = [W.Checkbox(value=True, description=str(n), indent=False) for n in names]
    for cb in cluster_filter_box.children:
        cb.observe(_render, names='value')

def _unselect_all(_=None):
    global _RENDER_LOCK
    _RENDER_LOCK = True
    try:
        for cb in cluster_filter_box.children:
            if isinstance(cb, W.Checkbox) and cb.value:
                cb.value = False
    finally:
        _RENDER_LOCK = False
        _render()

btn_clear.on_click(_unselect_all)

def _render(*_):
    if _RENDER_LOCK:
        return

    cby = color_dd.value
    is_categorical = (cby in res_cols) or (not pd.api.types.is_numeric_dtype(df[cby]))
    cluster_dd.disabled = is_categorical
    cluster_col = cby if is_categorical else cluster_dd.value

    # keep set
    full_clusters_str = df[cluster_col].astype(str)
    full_counts = full_clusters_str.value_counts()
    if hide_smallcb.value:
        keep_names = set(full_counts[(full_counts / len(df)) >= 0.005].index.astype(str))
    else:
        keep_names = set(full_counts.index.astype(str))

    # order & selector
    cluster_order = full_counts.loc[list(keep_names)].sort_values(ascending=False).index.tolist()
    current_names = [cb.description for cb in cluster_filter_box.children]
    if current_names != list(map(str, cluster_order)):
        _set_box_from_list(cluster_order)

    selected_names = _get_selected_from_box()
    active_names = [c for c in cluster_order if c in selected_names] if selected_names else []

    # sample
    if len(active_names) > 0:
        pool_mask = df[cluster_col].astype(str).isin(active_names)
        pool_idx = np.flatnonzero(pool_mask.to_numpy())
    else:
        pool_idx = np.array([], dtype=int)

    n = min(limit_sl.value, len(pool_idx))
    data = df.iloc[np.random.default_rng().choice(pool_idx, size=n, replace=False)] if n > 0 else df.iloc[[]].copy()

    # ---- SCATTER ----
    with out_scatter:
        out_scatter.clear_output(wait=True)
        x = data["array_col"].to_numpy()
        y = data["array_row"].to_numpy()
        fig, ax = plt.subplots(figsize=(9, 9))

        if is_categorical:
            labels = data[cby].astype(str).to_numpy()
            lut = _cat_colors(cluster_order, cmap_dd.value)
            color_map = {nm: lut[i] for i, nm in enumerate(cluster_order)}
            colors = [color_map.get(s, (0.6,0.6,0.6,1.0)) for s in labels]
            ax.scatter(x, y, c=colors, s=size_sl.value, alpha=alpha_sl.value)
            shown = (active_names or cluster_order)[:20]
            handles = [plt.Line2D([0],[0], marker='o', linestyle='', markersize=6,
                                  markerfacecolor=color_map[nm], markeredgecolor="none", label=str(nm))
                       for nm in shown]
            if handles:
                ax.legend(handles=handles, title=cby, loc="upper right",
                          fontsize=8, title_fontsize=9, frameon=True)
        else:
            vals = data[cby].to_numpy()
            sc = ax.scatter(x, y, c=vals, cmap=cmap_dd.value, s=size_sl.value, alpha=alpha_sl.value)
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=8)

        ax.set_xlabel("array_col"); ax.set_ylabel("array_row")
        if flip_x_cb.value: ax.set_xlim(XMAX + _padx, XMIN - _padx)
        else:               ax.set_xlim(XMIN - _padx, XMAX + _padx)
        if flip_y_cb.value: ax.set_ylim(YMAX + _pady, YMIN - _pady)
        else:               ax.set_ylim(YMIN - _pady, YMAX + _pady)

        ax.set_title(f"{sample_id} — {len(data):,} cells — {cby}", fontsize=11)
        plt.tight_layout(); from IPython.display import display as _d; _d(fig); plt.close(fig)

    # ---- BOTTOM PLOTS ----
    palette = _cat_colors(cluster_order, cmap_dd.value)
    color_map = {cluster_order[i]: palette[i] for i in range(len(cluster_order))}
    clusters_str_full = df[cluster_col].astype(str)
    active_order = [c for c in cluster_order if c in active_names] if active_names else []

    with out_counts:
        out_counts.clear_output(wait=True)
        if active_order:
            counts = clusters_str_full.value_counts().reindex(active_order, fill_value=0)
            fig, ax = plt.subplots(figsize=(10, max(2.5, 0.28*len(active_order))))
            yidx = np.arange(len(active_order))
            ax.barh(yidx, counts.values, edgecolor='none', color=[color_map[c] for c in active_order])
            ax.set_yticks(yidx); ax.set_yticklabels(active_order, fontsize=8)
            ax.set_xlabel("Cells"); ax.set_title(f"Counts per cluster ({cluster_col})", fontsize=10)
            ax.invert_yaxis()
            plt.tight_layout(); from IPython.display import display as _d; _d(fig); plt.close(fig)

    with out_box:
        out_box.clear_output(wait=True)
        if "total_counts" in df.columns and active_order:
            data_box = [df.loc[clusters_str_full == c, "total_counts"].values for c in active_order]
            fig, ax = plt.subplots(figsize=(10, max(2.5, 0.28*len(active_order))))
            bp = ax.boxplot(data_box, vert=False, tick_labels=active_order, patch_artist=True, showfliers=False)
            for patch, c in zip(bp['boxes'], [color_map[c] for c in active_order]):
                patch.set_facecolor(c); patch.set_edgecolor('none')
            for median in bp['medians']: median.set_color('black'); median.set_linewidth(1.2)
            for whisker in bp['whiskers']: whisker.set_color('#444')
            for cap in bp['caps']: cap.set_color('#444')
            ax.set_xlabel("total_counts"); ax.set_title(f"Total counts per cluster ({cluster_col})", fontsize=10)
            plt.tight_layout(); from IPython.display import display as _d; _d(fig); plt.close(fig)

# Reactivity
for w in [color_dd, cmap_dd, cluster_dd, hide_smallcb, size_sl, alpha_sl, limit_sl, flip_y_cb, flip_x_cb, limit_box]:
    w.observe(_render, names='value')

# ---------- Layout ----------
display(W.VBox([
    controls_row,
    controls_row2,
    controls_row3,
    out_scatter,
    W.HBox([out_counts, out_box], layout=Layout(align_items='flex-start'))
]))

# Initial draw
_render()
