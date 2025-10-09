# ---------- Imports ----------
import io, requests
import numpy as np
import pandas as pd
import ipywidgets as W
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# =========================
#   REMOTE DATA CONFIG
# =========================
SAMPLE_ID = globals().get("SAMPLE_ID", "IMMUNEX001")  # IMMUNEX004..IMMUNEX015
HE_SUFFIX = globals().get("HE_SUFFIX", globals().get("SEG_SUFFIX", "he0001"))

# Candidate resolutions to expose (extend if you publish more)
RES_CANDIDATES = ["resolution_leiden_0.5", "resolution_leiden_0.3"]

def _remote_csv_url(sample_id: str, res_key: str) -> str:
    return (
        "https://media.githubusercontent.com/media/InforBio/IMMUNEX/refs/heads/main/"
        f"notebooks/6.Reporting/files_{sample_id}__{HE_SUFFIX}/DE_exports/DE_{res_key}.csv"
    )

def _remote_png_url(sample_id: str, res_key: str) -> str:
    return (
        "https://raw.githubusercontent.com/InforBio/IMMUNEX/refs/heads/main/"
        f"notebooks/6.Reporting/files_{sample_id}__{HE_SUFFIX}/DE_exports/matrixplot_{res_key}.png"
    )

REQUIRED_COLS = {
    "names","cluster","scores","logfoldchanges","pvals","pvals_adj",
    "pct_in_cluster","pct_in_rest","resolution"
}
OPTIONAL_COLS = {"pct_diff","frac_in_dataset","pct_nz_group","pct_nz_reference"}

def _read_csv_robust(url: str) -> pd.DataFrame:
    """HTTP GET + robust CSV parse. Try comma first, then tab."""
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    text = r.text
    # Quick 404/HTML guard
    if "<html" in text.lower() or "not found" in text.lower():
        raise RuntimeError(f"URL doesn’t return CSV (HTML/404): {url}")

    # Try comma
    df = pd.read_csv(io.StringIO(text))
    if REQUIRED_COLS.issubset(set(df.columns)):
        return df

    # Try tab
    df_tab = pd.read_csv(io.StringIO(text), sep="\t")
    if REQUIRED_COLS.issubset(set(df_tab.columns)):
        return df_tab

    # If here, show a helpful diagnostic
    raise ValueError(
        f"CSV at {url} missing required cols.\n"
        f"Found columns: {list(df.columns) if len(df.columns)>0 else '<<none>>'}"
    )

# Probe which resolutions actually exist and are valid
resolutions = []
res2file    = {}
for r in RES_CANDIDATES:
    url = _remote_csv_url(SAMPLE_ID, r)
    try:
        df_probe = _read_csv_robust(url)
        resolutions.append(r)
        res2file[r] = url
    except Exception:
        pass




if not resolutions:
    # Last resort: expose 0.5 and let the UI show the error text cleanly
    resolutions = ["resolution_leiden_0.5"]
    res2file[resolutions[0]] = _remote_csv_url(SAMPLE_ID, resolutions[0])

# ---------- Widgets ----------
res_dd      = W.Dropdown(options=resolutions, value=resolutions[0], description="Resolution")
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
    value="scores", description="Sort by"
)
asc_cb      = W.Checkbox(value=False, description="Ascending")

# Cluster selection panel
hide_small_cb = W.Checkbox(value=True, description="Hide clusters <0.5%")
clusters_box  = W.VBox([])  # filled dynamically with Checkboxes
select_all_btn  = W.Button(description="All", layout=W.Layout(width="60px"))
select_none_btn = W.Button(description="None", layout=W.Layout(width="60px"))

# Gene filter, quick toggles
gene_q      = W.Text(value="", description="Gene contains")
only_up_cb  = W.Checkbox(value=False, description="Only upregulated (logFC>0)")
sig_cb      = W.Checkbox(value=True,  description="Only significant")
alpha_ft    = W.FloatText(value=0.05, description="α (p_adj)", step=1e-4)

# Pagination
page_size = W.IntSlider(value=50, min=10, max=500, step=10, description="Rows/page")
page      = W.IntSlider(value=1, min=1, max=1, step=1, description="Page")

# I/O
export_btn  = W.Button(description="Export filtered CSV", button_style="")
status_out  = W.Output()
table_out   = W.Output()
png_out     = W.Output()

# internal guard/cache
_updating = False
_current_df = None

# ---------- Helpers ----------
def _load_deg(res_key: str) -> pd.DataFrame:
    url = res2file[res_key]
    df = _read_csv_robust(url)

    # Fill optional columns when absent
    if "pct_diff" not in df.columns and {"pct_nz_group","pct_nz_reference"}.issubset(df.columns):
        df["pct_diff"] = df["pct_nz_group"]*100.0 - df["pct_nz_reference"]*100.0
    for c in OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Types & cleanliness
    df["cluster"] = df["cluster"].astype(str)
    df["names"]   = df["names"].astype(str)

    # Ensure only columns (avoid index leaks)
    if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:
        df = df.reset_index(drop=True)
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
    # significance filter first
    if sig_cb.value and "pvals_adj" in q.columns:
        q = q[pd.to_numeric(q["pvals_adj"], errors="coerce") <= float(alpha_ft.value)]
    if "pvals" in q.columns:
        q = q[pd.to_numeric(q["pvals"], errors="coerce") <= float(pval_max.value)]
    if "pvals_adj" in q.columns:
        q = q[pd.to_numeric(q["pvals_adj"], errors="coerce") <= float(padj_max.value)]
    if "logfoldchanges" in q.columns:
        q = q[pd.to_numeric(q["logfoldchanges"], errors="coerce") >= float(lfc_min.value)]
    if "pct_diff" in q.columns:
        q = q[pd.to_numeric(q["pct_diff"], errors="coerce") >= float(pdiff_min.value)]
    if only_up_cb.value and "logfoldchanges" in q.columns:
        q = q[pd.to_numeric(q["logfoldchanges"], errors="coerce") > 0]

    sel_clusters = _get_selected_clusters()
    if sel_clusters:
        q = q[q["cluster"].isin(sel_clusters)]

    if gene_q.value.strip():
        pat = gene_q.value.strip().lower()
        q = q[q["names"].str.lower().str.contains(pat, na=False)]
    return q

def _topN(df: pd.DataFrame) -> pd.DataFrame:
    key = sort_by_dd.value
    ascending = bool(asc_cb.value)
    if key not in df.columns:
        return df.head(0)
    if top_mode_dd.value.startswith("Per-cluster"):
        # Sort first, then groupby-head to avoid MultiIndex and ambiguity
        return (
            df.sort_values(["cluster", key], ascending=[True, ascending])
              .groupby("cluster", as_index=False, sort=False)
              .head(int(topN_sl.value))
        )
    return df.sort_values(key, ascending=ascending).head(int(topN_sl.value))

# ---------- Main update ----------
def _update(*_):
    global _updating, _current_df
    if _updating: return
    _updating = True
    try:
        try:
            _current_df = _load_deg(res_dd.value)
        except Exception as e:
            with table_out:
                table_out.clear_output(wait=True)
                display(HTML(f"<b style='color:#b00'>Failed to load CSV:</b> {e}"))
            return

        _refresh_clusters_options(_current_df)

        q = _apply_filters(_current_df)
        out = _topN(q)

        # ensure 'cluster' only a column
        if isinstance(out.index, pd.MultiIndex) and "cluster" in out.index.names:
            out = out.reset_index(level="cluster")
        elif out.index.name == "cluster":
            out = out.reset_index()

        cols = ["cluster","names","scores","logfoldchanges","pvals","pvals_adj",
                "frac_in_dataset","pct_in_cluster","pct_in_rest","pct_diff"]
        out = out[[c for c in cols if c in out.columns]].copy()

        # stable sort: cluster then current sort key
        if "cluster" in out.columns and sort_by_dd.value in out.columns:
            out = out.sort_values(["cluster", sort_by_dd.value],
                                  ascending=[True, bool(asc_cb.value)])

        # ---- Pagination ----
        total = len(out)
        page.max = max(1, (total-1)//int(page_size.value) + 1)
        i = (int(page.value)-1) * int(page_size.value)
        out_page = out.iloc[i:i + int(page_size.value)]

        with table_out:
            table_out.clear_output(wait=True)
            pd.set_option("display.max_rows", 50)
            pd.set_option("display.max_colwidth", 32)

            base_n = len(_current_df)
            filt_n = len(q)
            display(HTML(
                f"<small>Source: GitHub raw — {SAMPLE_ID}/{res_dd.value}</small><br>"
                f"<b>{len(out_page):,}</b> rows shown (page {page.value}/{page.max}; "
                f"filtered <b>{filt_n:,}</b> of base <b>{base_n:,}</b>)."
            ))

            if filt_n == 0:
                display(HTML(
                    "<span style='color:#b00'><b>No rows after filters.</b></span> "
                    "Tip: toggle off <i>Only significant</i> or increase <i>p_adj ≤</i>."
                ))
                # Show a small unfiltered preview to prove the CSV is fine
                prev = _current_df[[c for c in cols if c in _current_df.columns]].head(10).copy()
                display(HTML("<i>Preview of raw rows (first 10):</i>"))
                display(prev.reset_index(drop=True))
            else:
                display(out_page.reset_index(drop=True))

        # ---- Remote PNG preview (let browser load; shows alt on 404) ----
        with png_out:
            png_out.clear_output(wait=True)
            img_url = _remote_png_url(SAMPLE_ID, res_dd.value)
            
            print(img_url)
            display(HTML(
                f'<div style="margin-top:6px">'
                f'<img src="{img_url}" alt="No matrix plot found at {img_url}" style="max-width:100%"/></div>'
            ))
    finally:
        _updating = False

def _on_res_change(_): _update()

def _export(_):
    # Save locally even if source is remote
    dest_dir = Path.cwd() / "DE_exports_local"
    dest_dir.mkdir(exist_ok=True, parents=True)
    df = _current_df if _current_df is not None else _load_deg(res_dd.value)
    q  = _apply_filters(df)
    out = _topN(q)
    cols = ["cluster", "names", "scores", "logfoldchanges",
            "pvals", "pvals_adj", "frac_in_dataset",
            "pct_in_cluster", "pct_in_rest", "pct_diff"]
    out = out[[c for c in cols if c in out.columns]]
    dest = dest_dir / f"DEUI_{SAMPLE_ID}_{res_dd.value}_filtered.csv"
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

display(W.VBox([controls1, controls2, controls3, cluster_panel, controls_pager, table_out, png_out]))

# initial draw
_update()
