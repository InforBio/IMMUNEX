# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd, numpy as np, ipywidgets as W
from IPython.display import display, Image, HTML

# Try to read global BASE_DIR; fallback to CWD
BASE_DIR = globals().get("BASE_DIR", Path.cwd())

def _find_samples_de(base_dir: Path):
    """Return {sample_name: {'dir': sample_dir, 'de_files': [paths...]}}."""
    out = {}
    if not base_dir.exists():
        return out
    for d in sorted([p for p in base_dir.iterdir() if p.is_dir()]):
        de_dir = d / "DE_exports"
        files = sorted(de_dir.glob("DE_resolution_leiden_*.csv")) if de_dir.exists() else []
        if files:
            out[d.name] = {"dir": d, "de_dir": de_dir, "de_files": files}
    return out

_SAMPLES = _find_samples_de(BASE_DIR)

sample_dd = W.Dropdown(
    options=sorted(_SAMPLES.keys()),
    value=(sorted(_SAMPLES.keys())[0] if _SAMPLES else None),
    description="Sample",
    layout=W.Layout(width="350px")
)
refresh_btn = W.Button(description="↻ Refresh", tooltip="Rescan", layout=W.Layout(width="110px"))
info_html = W.HTML()
header = W.HBox([sample_dd, refresh_btn])

def _set_info():
    if not _SAMPLES:
        info_html.value = f"<div style='color:#b00020'>No DE folders found under <code>{BASE_DIR}</code> (looked for <code>*/DE_exports/DE_resolution_leiden_*.csv</code>)</div>"
    else:
        info_html.value = f"<small>Base: <code>{BASE_DIR}</code> — {len(_SAMPLES)} sample(s) with DE</small>"

def _rescan(_=None):
    global _SAMPLES
    _SAMPLES = _find_samples_de(BASE_DIR)
    sample_dd.options = sorted(_SAMPLES.keys())
    if sample_dd.options:
        if sample_dd.value not in sample_dd.options:
            sample_dd.value = sample_dd.options[0]
    else:
        sample_dd.value = None
    _set_info()
    _refresh_resolutions()

refresh_btn.on_click(_rescan)
_set_info()

# ---------- Dynamic resolution mapping ----------
res_dd      = W.Dropdown(options=[], description="Resolution")
top_mode_dd = W.Dropdown(options=["Per-cluster top N", "Global top N"], value="Per-cluster top N", description="Top mode")
topN_sl     = W.IntSlider(value=20, min=5, max=200, step=5, description="Top N", continuous_update=False)

# Filters
pval_max    = W.FloatText(value=1.0,  description="pval ≤",    step=1e-4)
padj_max    = W.FloatText(value=0.05, description="p_adj ≤",   step=1e-4)
lfc_min     = W.FloatText(value=0.0,  description="logFC ≥",   step=0.1)
pdiff_min   = W.FloatText(value=0.0,  description="pct_diff ≥", step=1.0)

# Sort
sort_by_dd  = W.Dropdown(
    options=["scores","logfoldchanges","pvals_adj","pvals","pct_diff","pct_in_cluster","pct_in_rest"],
    value="scores",
    description="Sort by"
)
asc_cb      = W.Checkbox(value=False, description="Ascending")

# Cluster selection + pagination
hide_small_cb = W.Checkbox(value=True, description="Hide clusters <0.5%")
clusters_box  = W.VBox([])  # filled dynamically with Checkboxes
select_all_btn  = W.Button(description="All", layout=W.Layout(width="60px"))
select_none_btn = W.Button(description="None", layout=W.Layout(width="60px"))

gene_q      = W.Text(value="", description="Gene contains")
only_up_cb  = W.Checkbox(value=False, description="Only upregulated (logFC>0)")
sig_cb      = W.Checkbox(value=True,  description="Only significant")
alpha_ft    = W.FloatText(value=0.05, description="α (p_adj)", step=1e-4)

page_size = W.IntSlider(value=50, min=10, max=500, step=10, description="Rows/page")
page      = W.IntSlider(value=1, min=1, max=1, step=1, description="Page")

export_btn  = W.Button(description="Export filtered CSV", button_style="")
status_out  = W.Output()
table_out   = W.Output()
png_out     = W.Output()

_updating = False
_current_df = None
_res2file = {}

def _res_from_path(p: Path) -> str:
    # DE_resolution_leiden_0.5.csv -> resolution_leiden_0.5
    return p.stem.replace("DE_", "")

def _refresh_resolutions(*_):
    global _res2file
    if not sample_dd.value:
        res_dd.options = []
        return
    files = _SAMPLES[sample_dd.value]["de_files"]
    res_keys = [_res_from_path(p) for p in files]
    _res2file = { _res_from_path(p): p for p in files }
    res_dd.options = res_keys
    if res_keys:
        res_dd.value = res_keys[0]
    _update()

def _load_deg(res_key: str) -> pd.DataFrame:
    df = pd.read_csv(_res2file[res_key])
    needed = ["names","scores","logfoldchanges","pvals","pvals_adj",
              "pct_diff","pct_in_cluster","pct_in_rest","cluster","resolution","frac_in_dataset"]
    for c in needed:
        if c not in df.columns:
            if c == "pct_diff" and ("pct_nz_group" in df.columns and "pct_nz_reference" in df.columns):
                df["pct_diff"] = df["pct_nz_group"]*100.0 - df["pct_nz_reference"]*100.0
            else:
                df[c] = np.nan
    df["cluster"] = df["cluster"].astype(str)
    return df

def _get_selected_clusters():
    return [cb.description for cb in clusters_box.children if isinstance(cb, W.Checkbox) and cb.value]

def _refresh_clusters_options(df: pd.DataFrame):
    global _updating
    if _updating: return
    _updating = True
    try:
        if "frac_in_dataset" in df.columns and df["frac_in_dataset"].notna().any():
            by_cl = (df[["cluster","frac_in_dataset"]]
                     .dropna()
                     .groupby("cluster", as_index=False)["frac_in_dataset"]
                     .max())
            if hide_small_cb.value:
                by_cl = by_cl[by_cl["frac_in_dataset"] >= 0.005]
            clusters = by_cl["cluster"].astype(str).tolist()
        else:
            clusters = sorted(df["cluster"].astype(str).unique().tolist())
        clusters = sorted(clusters, key=lambda x: (len(x), x))
        clusters_box.children = [W.Checkbox(value=True, description=c, indent=False) for c in clusters]
    finally:
        _updating = False

def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    q = df.copy()
    if sig_cb.value and "pvals_adj" in q.columns:
        q = q[q["pvals_adj"] <= alpha_ft.value]
    if "pvals" in q.columns:          q = q[q["pvals"]      <= pval_max.value]
    if "pvals_adj" in q.columns:      q = q[q["pvals_adj"]  <= padj_max.value]
    if "logfoldchanges" in q.columns: q = q[q["logfoldchanges"] >= lfc_min.value]
    if "pct_diff" in q.columns:       q = q[q["pct_diff"]   >= pdiff_min.value]
    if only_up_cb.value and "logfoldchanges" in q.columns:
        q = q[q["logfoldchanges"] > 0]
    sel_clusters = _get_selected_clusters()
    if sel_clusters:
        q = q[q["cluster"].isin(sel_clusters)]
    if gene_q.value.strip():
        pat = gene_q.value.strip().lower()
        q = q[q["names"].astype(str).str.lower().str.contains(pat, na=False)]
    return q

def _topN(df: pd.DataFrame) -> pd.DataFrame:
    key = sort_by_dd.value; ascending = asc_cb.value
    if top_mode_dd.value.startswith("Per-cluster"):
        if key not in df.columns: return df.head(0)
        return (df.groupby("cluster", group_keys=False)
                  .apply(lambda x: x.sort_values(key, ascending=ascending).head(topN_sl.value)))
    else:
        if key not in df.columns: return df.head(0)
        return df.sort_values(key, ascending=ascending).head(topN_sl.value)

def _update(*_):
    global _updating, _current_df
    if not sample_dd.value or not res_dd.options:
        table_out.clear_output(); png_out.clear_output()
        return
    if _updating: return
    _updating = True
    try:
        _current_df = _load_deg(res_dd.value)
        _refresh_clusters_options(_current_df)

        q = _apply_filters(_current_df)
        out = _topN(q)

        cols = ["cluster", "names", "scores", "logfoldchanges",
                "pvals", "pvals_adj", "frac_in_dataset",
                "pct_in_cluster", "pct_in_rest", "pct_diff"]
        out = out[[c for c in cols if c in out.columns]].copy()

        if "cluster" in out.columns and sort_by_dd.value in out.columns:
            out = out.sort_values(["cluster", sort_by_dd.value],
                                  ascending=[True, asc_cb.value])

        total = len(out)
        page.max = max(1, (total-1)//page_size.value + 1)
        i = (page.value-1) * page_size.value
        out_page = out.iloc[i:i + page_size.value]

        with table_out:
            table_out.clear_output(wait=True)
            pd.set_option("display.max_rows", 50)
            pd.set_option("display.max_colwidth", 32)
            display(HTML(f"<b>{len(out_page):,}</b> rows shown "
                         f"(page {page.value}/{page.max}; filtered {len(q):,} of base {len(_current_df):,})."))
            display(out_page.reset_index(drop=True))

        with png_out:
            png_out.clear_output(wait=True)
            base = res_dd.value
            de_dir = _SAMPLES[sample_dd.value]["de_dir"]
            png_path = de_dir / f"matrixplot_{base}.png"
            if png_path.exists():
                display(Image(filename=str(png_path), embed=True))
            else:
                display(HTML(f"<i>No matrix plot found at {png_path.name}</i>"))
    finally:
        _updating = False

def _on_res_change(_): _update()
def _export(_):
    if _current_df is None:
        return
    df = _current_df
    q  = _apply_filters(df)
    out = _topN(q)
    cols = ["cluster", "names", "scores", "logfoldchanges",
            "pvals", "pvals_adj", "frac_in_dataset",
            "pct_in_cluster", "pct_in_rest", "pct_diff"]
    out = out[[c for c in cols if c in out.columns]]
    de_dir = _SAMPLES[sample_dd.value]["de_dir"]
    dest = de_dir / f"DEUI_{res_dd.value}_filtered.csv"
    out.to_csv(dest, index=False)
    with status_out:
        status_out.clear_output(wait=True)
        print(f"Saved: {dest}")

def _select_all(_):
    for cb in clusters_box.children:
        if isinstance(cb, W.Checkbox): cb.value = True
    _update()

def _select_none(_):
    for cb in clusters_box.children:
        if isinstance(cb, W.Checkbox): cb.value = False
    _update()

# ---------- Wire events & layout ----------
sample_dd.observe(_refresh_resolutions, names="value")
res_dd.observe(_on_res_change, names="value")
for w in [top_mode_dd, topN_sl, pval_max, padj_max, lfc_min, pdiff_min,
          sort_by_dd, asc_cb, gene_q, only_up_cb, sig_cb, alpha_ft,
          hide_small_cb, page, page_size]:
    w.observe(lambda _: _update(), names="value")

select_all_btn.on_click(_select_all)
select_none_btn.on_click(_select_none)
export_btn.on_click(_export)

controls1 = W.HBox([res_dd, top_mode_dd, topN_sl, sort_by_dd, asc_cb])
controls2 = W.HBox([pval_max, padj_max, lfc_min, pdiff_min, only_up_cb, sig_cb, alpha_ft])
controls3 = W.HBox([gene_q, export_btn, status_out])
controls_pager = W.HBox([page_size, page])

cluster_panel = W.VBox([
    W.HBox([W.Label("Clusters"), hide_small_cb, select_all_btn, select_none_btn]),
    W.Box([clusters_box], layout=W.Layout(max_height='220px', overflow='auto', border='1px solid #ddd', padding='4px'))
])

display(W.VBox([header, info_html, controls1, controls2, controls3, cluster_panel, controls_pager, table_out, png_out]))

# initial
_refresh_resolutions()
