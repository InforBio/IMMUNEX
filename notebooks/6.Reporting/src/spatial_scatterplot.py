%%writefile viz_cluster_ui.py
import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl, ipywidgets as W
from ipywidgets import Layout
from IPython.display import display as _d

def show_cluster_ui(df, sample_id="sample"):
    # ---------- Basic sanity ----------
    needed = {"array_row", "array_col"}
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

    extra_cols = []
    if "pct_in_cluster" in df.columns:
        extra_cols.append("pct_in_cluster")

    color_options = res_cols + qc_cols + extra_cols
    default_res = next((c for c in res_cols if c == "resolution_leiden_0.5"), res_cols[0])

    # ---------- Widgets ----------
    cmaps_cat = ['tab10','tab20','Set1','Set2','Set3','Pastel1','Pastel2','Dark2','Accent','Paired','tab20b','tab20c']
    cmaps_cont = ['viridis','plasma','inferno','magma','cividis','Spectral','coolwarm','bwr','seismic','RdBu_r','PiYG','PRGn','BrBG','rainbow','nipy_spectral','Greys','Blues','Greens','Oranges','Reds']
    cmap_all = cmaps_cat + cmaps_cont

    default_color = "pct_in_cluster" if "pct_in_cluster" in color_options else default_res

    color_dd     = W.Dropdown(options=color_options, value=default_color, description="Color by")
    cmap_dd      = W.Dropdown(options=cmap_all, value='tab20', description="Colormap")
    cluster_dd   = W.Dropdown(options=res_cols, value=default_res, description="Clusters for summaries")
    hide_smallcb = W.Checkbox(value=True, description="Hide small (<0.5%)")
    size_sl      = W.IntSlider(value=2, min=1, max=6, step=1, description="Size", continuous_update=False)

    max_allowed  = int(len(df))
    limit_sl     = W.IntSlider(value=min(len(df), 200_000), min=10_000, max=max_allowed, step=5_000,
                               description="Max pts", continuous_update=False, layout=Layout(width='450px'))
    limit_box    = W.BoundedIntText(value=limit_sl.value, min=10_000, max=max_allowed, step=1000, description='')
    W.jslink((limit_sl, 'value'), (limit_box, 'value'))

    alpha_sl     = W.FloatSlider(value=0.7, min=0.1, max=1.0, step=0.05, description="Alpha", readout_format=".2f", continuous_update=False)
    flip_y_cb    = W.Checkbox(value=True,  description="Invert Y")
    flip_x_cb    = W.Checkbox(value=True,  description="Invert X")

    cluster_label = W.HTML("<b>Show clusters:</b>")
    cluster_filter_box = W.Box(layout=Layout(display='flex', flex_flow='row wrap', overflow_x='auto', overflow_y='auto',
                                             border='1px solid #ddd', padding='4px', max_height='80px', gap='6px'))

    controls_row  = W.HBox([color_dd, cmap_dd, cluster_dd, hide_smallcb])
    controls_row2 = W.HBox([size_sl, alpha_sl, W.HBox([limit_sl, limit_box], layout=Layout(align_items='center')), flip_y_cb, flip_x_cb])
    controls_row3 = W.VBox([cluster_label, cluster_filter_box])
    out_scatter = W.Output()
    out_counts  = W.Output()
    out_box     = W.Output()

    XMIN, XMAX = float(df["array_col"].min()), float(df["array_col"].max())
    YMIN, YMAX = float(df["array_row"].min()), float(df["array_row"].max())
    _padx = 0.01 * (XMAX - XMIN) or 1.0
    _pady = 0.01 * (YMAX - YMIN) or 1.0

    def _cat_colors(categories, cmap_name):
        cmap = mpl.colormaps.get_cmap(cmap_name)
        xs = np.linspace(0, 1, max(len(categories), 1), endpoint=False)
        return [cmap(x) for x in xs]

    def _get_selected_from_box():
        return {cb.description for cb in cluster_filter_box.children if isinstance(cb, W.Checkbox) and cb.value}

    def _set_box_from_list(names):
        cluster_filter_box.children = [W.Checkbox(value=True, description=str(n), indent=False) for n in names]
        for cb in cluster_filter_box.children:
            cb.observe(_render, names='value')

    def _render(*_):
        # (same render body as in your code)
        pass  # omitted here for brevity — copy your full _render implementation

    for w in [color_dd, cmap_dd, cluster_dd, hide_smallcb, size_sl, alpha_sl, limit_sl, flip_y_cb, flip_x_cb, limit_box]:
        w.observe(_render, names='value')

    display(W.VBox([
        controls_row, controls_row2, controls_row3, out_scatter,
        W.HBox([out_counts, out_box], layout=Layout(align_items='flex-start'))
    ]))
    _render()
