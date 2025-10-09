# ---------- Imports ----------
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from IPython.display import display, HTML
import ipywidgets as W
import matplotlib.pyplot as plt

# --- keep your existing mapping from files to resolution keys ---
def _res_from_path(p: Path) -> str:
    return p.stem.replace("DE_", "")   # e.g., resolution_leiden_0.5

resolutions = [_res_from_path(p) for p in de_files]
res2file = { _res_from_path(p): p for p in de_files }

# ---- Panels (unchanged) ----
PANEL_NSCLC = {
    "T cells (CD3/CD4/CD8)": ["CD3D","CD3E","CD4","CD8A","CD8B"],
    "Cytotoxic T/NK effector": ["NKG7","GZMB","PRF1","GNLY","KLRD1"],
    "Exhausted/Tfh T": ["PDCD1","LAG3","HAVCR2","TIGIT","CXCR5","ICOS","BCL6"],
    "Treg": ["FOXP3","IL2RA","CTLA4","IKZF2"],
    "B cells": ["MS4A1","CD79A","CD79B","BANK1","TCL1A"],
    "Plasma cells": ["SDC1","MZB1","XBP1","JCHAIN","PRDM1"],
    "TLS chemokines": ["CXCL13","CCL19","CCL21"],
    "cDC1": ["XCR1","CLEC9A","BATF3"],
    "cDC2": ["CD1C","FCER1A","ITGAX"],
    "Migratory DC (LAMP3)": ["LAMP3","CCR7","CCL19","CCL21"],
    "Macrophage (M2/TAM-like)": ["MRC1","CD163","CSF1R","MSR1","MARCO","SPP1"],
    "Neutrophils": ["S100A8","S100A9","CSF3R","CXCR2"],
    "Mast cells": ["KIT","TPSAB1","CPA3"],
    "Endothelial": ["PECAM1","VWF","KDR","FLT1","DLL4"],
    "Lymphatic EC": ["PDPN","PROX1","LYVE1"],
    "Fibroblasts": ["COL1A1","COL1A2","PDGFRB","DCN"],
    "Myofibroblasts/CAF": ["ACTA2","TAGLN","FAP","PDGFRB","ITGB1"],
    "Tumor/Epithelial (pan)": ["EPCAM","KRT8","KRT18","KRT19","KRT17","MUC1","KRT7"],
    "LUAD-like": ["NKX2-1","SLC34A2","MUC1","KRT7","CLDN18","KRT8"],
    "LUSC-like": ["KRT5","KRT6A","TP63","SOX2","DSG3"],
    "Neuroendocrine": ["CHGA","CHGB","SYP","ENO2","NCAM1"],
    "EMT/stress": ["VIM","FN1","ZEB1","SNAI1","ITGA6"],
    "Proliferation": ["MKI67","TOP2A","PCNA","TYMS"],
    "Hypoxia/Angio": ["CA9","VEGFA","ANGPT2"],
    "Alveolar type II": ["SFTPC","SFTPA1","SFTPB","SLC34A2"],
    "Alveolar type I":  ["AGER","CAV1","PDPN"],
    "Club": ["SCGB1A1","SCGB3A2"],
    "Ciliated": ["FOXJ1","TPPP3","PIFO","DNAH5"],
    "Basal": ["KRT5","KRT14","TP63"],
    "Goblet/mucous": ["MUC5AC","MUC5B","SPDEF"],

    "Immune cells": ["PTPRC","CD3D","MS4A1","NKG7","GNLY","LYZ","LST1","FCGR3A","CCR7","CXCR5","PDCD1","MRC1","CD163","XCR1","CD1C"],
    "Tumor": ["EPCAM","KRT8","KRT18","KRT19","KRT7","MUC1","KRT17","TACSTD2","ERBB2","CEACAM6"],
    "Stromal": ["COL1A1","COL1A2","PDGFRB","DCN","VIM","ACTA2","TAGLN","PDGFRA","FAP","ITGB1"],
}

PANEL_PA = {
    "All" : ["KRT18","IRX2","NAPSA","SPINK13","KRT7","CAPN8","GPRC5A","TP63","CALML3","KRT5","PKP1","TESC","SPINK1","C9orf152","CD3D","CD3E","CD8A","CD8B","BCL6","CD81","POU2AF1","RGS13","IGHA1","IGHG1","JCHAIN","AKNA","ARHGAP25","CCL21","CD180","CD2","CD27","CD37","CD38","CLEC17A","CLEC9A","CLECL1","FAIM3","FAM65B","GIMAP4","MAP4K1","PAX5","TNFRSF17","TRAF3IP3","CD209","S100A1","S100A10","S100A11","S100A12","S100A13","S100A14","S100A16","S100A2","S100A3","S100A4","S100A5","S100A6","S100A7","S100A7L1","S100A8","S100A9","S100B","S100G","S100P","S100PBP","S100Z","ANLN","BRIP1","BUB1B","CASC5","CCNB1","CCNB2","CCNE2","CEP55","CKAP2L","DLGAP5","DTL","E2F8","ECT2","ESCO2","EXO1","EXOC6","FBXO5","FIGNL1","HELLS","HMMR","IARS","KIF11","KIF18A","KIF20A","KNTC1","MAD2L1","MASTL","MTHFD2","NCAPG2","NCAPH","NEIL3","NUF2","PRC1","PSAT1","RTKN2","ADRM1","AHSA1","C1GALT1C1","CCT5","CCT6B","CETN3","CSE1L","EIF2S1","GAL","GEMIN6","GPT2","KIAA0101","MND1","MPZL1","MRPS16","PCNA","PTRH2","RFC5","SPC25","TIMM13","TIMM8B","TK1","TUBB","TXNDC17","FCGR3A","FCN1","CD68","MSR1","APOE","C1QA","XCR1","CLEC9A","CD1C","CLEC10A","FCER1A","LILRA4","SPIB","IRF7","KIT","CPA3","CDH5","PECAM1","VWF","CCL21","FLT4","LYVE1","PROX1","ACTA2","COL13A1","COL14A1","MYH11","PDGFRA","PDGFRB","VIM","CNN1","CRYAB","DES","SYNPO2","ABTB1","AMPD2","CAMP","EMR4P","FPR1","FPR2","GPR77","MAEA","PROK2","SEC14L1","SEPX1","SLC25A37","TNFSF14","TREML4","VNN2","XPO6","BATF","CCR10","CCR8","CTLA4","FOXP3","IKZF2","IKZF4","IL10RA","IL2RA","IL32","TIGIT","TNFRSF18"],
    "Tumor cells 1": ["KRT18","IRX2","NAPSA","SPINK13","KRT7","CAPN8","GPRC5A"],
    "Tumor cells 2": ["TP63","CALML3","KRT5","PKP1","TESC","SPINK1","C9orf152","KRT7"],
    "T cells": ["CD3D","CD3E","CD8A","CD8B"],
    "Germinal_center_Bcells": ["BCL6","CD81","POU2AF1","RGS13"],
    "Plasma cells": ["IGHA1","IGHG1","JCHAIN"],
    "B_cells": ["AKNA","ARHGAP25","CCL21","CD180","CD2","CD27","CD37","CD38","CLEC17A","CLEC9A","CLECL1","FAIM3","FAM65B","GIMAP4","MAP4K1","PAX5","TNFRSF17","TRAF3IP3"],
    "Dendritic_cells": ["CD209","S100A1","S100A10","S100A11","S100A12","S100A13","S100A14","S100A16","S100A2","S100A3","S100A4","S100A5","S100A6","S100A7","S100A7L1","S100A8","S100A9","S100B","S100G","S100P","S100PBP","S100Z"],
    "Activated_CD4_T_cell": ["ANLN","BRIP1","BUB1B","CASC5","CCNB1","CCNB2","CCNE2","CEP55","CKAP2L","DLGAP5","DTL","E2F8","ECT2","ESCO2","EXO1","EXOC6","FBXO5","FIGNL1","HELLS","HMMR","IARS","KIF11","KIF18A","KIF20A","KNTC1","MAD2L1","MASTL","MTHFD2","NCAPG2","NCAPH","NEIL3","NUF2","PRC1","PSAT1","RTKN2"],
    "Activated_CD8_T_cell": ["ADRM1","AHSA1","C1GALT1C1","CCT5","CCT6B","CETN3","CSE1L","EIF2S1","GAL","GEMIN6","GPT2","KIAA0101","MND1","MPZL1","MRPS16","PCNA","PTRH2","RFC5","SPC25","TIMM13","TIMM8B","TK1","TUBB","TXNDC17"],
    "Myeloid cells": ["FCGR3A","FCN1","CD68","MSR1","APOE","C1QA","XCR1","CLEC9A","CD1C","CLEC10A","FCER1A","LILRA4","SPIB","IRF7","KIT","CPA3"],
    "Macrophage": ["CD68","MSR1","APOE","C1QA"],
    "cDC1": ["XCR1","CLEC9A"],
    "cDC2": ["CD1C","CLEC10A","FCER1A"],
    "pDC": ["LILRA4","SPIB","IRF7"],
    "Mast cells": ["KIT","CPA3"],
    "Endothelial cells": ["CDH5","PECAM1","VWF"],
    "Lymphatic endothelial cells": ["CCL21","FLT4","LYVE1","PROX1"],
    "Fibroblasts": ["ACTA2","COL13A1","COL14A1","MYH11","PDGFRA","PDGFRB","VIM"],
    "Smooth muscle cells": ["CNN1","CRYAB","DES","SYNPO2"],
    "Neutrophils": ["ABTB1","AMPD2","CAMP","EMR4P","FPR1","FPR2","GPR77","MAEA","PROK2","SEC14L1","SEPX1","SLC25A37","TNFSF14","TREML4","VNN2","XPO6"],
    "Treg": ["BATF","CCR10","CCR8","CTLA4","FOXP3","IKZF2","IKZF4","IL10RA","IL2RA","IL32","TIGIT","TNFRSF18"],
}

# ---------- Widgets ----------
res_dd   = W.Dropdown(options=resolutions, value=resolutions[0], description="Resolution")
panel_dd = W.Dropdown(options=["NSCLC panel","P.A. panel"], value="NSCLC panel", description="Markers")

def _current_panel():
    return PANEL_NSCLC if panel_dd.value == "NSCLC panel" else PANEL_PA

ctype_dd = W.Dropdown(options=list(_current_panel().keys()),
                      value=("Tumor" if "Tumor" in _current_panel() else "Tumor/Epithelial (pan)"),
                      description="Cell type")

# ---- Value options now include expression stats & deltas ----
VAL_OPTIONS = [
    # classic
    "scores","logfoldchanges","pct_diff","pct_in_cluster","pct_in_rest","-log10(pvals_adj)",
    # means & FC
    "mean_in","mean_rest","diff_mean","log2fc_mean",
    # non-zero means
    "nz_mean_in","nz_mean_rest","nz_mean_delta",
    # medians (sampled)
    "median_in_s","median_rest_s","median_delta_s",
    # log1p means (sampled)
    "mean_log1p_in_s","mean_log1p_rest_s","mean_log1p_delta_s",
    # zero fraction (0..1)
    "zero_frac_in","zero_frac_rest","zero_frac_delta",
]
value_dd = W.Dropdown(options=VAL_OPTIONS, value="scores", description="Value")
agg_dd   = W.Dropdown(options=["max","mean","median"], value="max", description="Aggregate")

scale_dd = W.Dropdown(
    options=["raw","zscore_by_gene","zscore_by_cluster","zscore_global",
             "minmax_by_gene","minmax_by_cluster","minmax_global"],
    value="zscore_by_gene", description="Scale"
)
cmap_dd  = W.Dropdown(
    options=["RdBu_r","Spectral","viridis","plasma","magma","cividis","coolwarm",
             "bwr","seismic","turbo","cubehelix","rocket","mako","YlOrRd","YlGnBu"],
    value="RdBu_r", description="Colormap"
)

hide_small = W.Checkbox(value=True, description="Hide clusters <0.5%")
row_cl_cb  = W.Checkbox(value=True,  description="Cluster rows")
col_cl_cb  = W.Checkbox(value=True,  description="Cluster cols")
vmin_ft    = W.FloatText(value=np.nan, description="vmin")
vmax_ft    = W.FloatText(value=np.nan, description="vmax")

# ---- NEW: Filters (leave NaN to disable a filter) ----
p_adj_max     = W.FloatText(value=np.nan, description="p_adj ≤")
log2fc_min    = W.FloatText(value=np.nan, description="log2FC ≥")
pct_in_min    = W.FloatText(value=np.nan, description="%in ≥")        # expects 0..100
median_in_min = W.FloatText(value=np.nan, description="median_in ≥")
log1p_in_min  = W.FloatText(value=np.nan, description="log1p_in ≥")
nzmean_in_min = W.FloatText(value=np.nan, description="nz_mean_in ≥")
zero_in_max   = W.FloatText(value=np.nan, description="zero_in ≤")    # expects 0..1

ui = W.VBox([
    W.HBox([res_dd, panel_dd, ctype_dd, value_dd, agg_dd]),
    W.HBox([scale_dd, cmap_dd, hide_small, row_cl_cb, col_cl_cb, vmin_ft, vmax_ft]),
    W.HBox([p_adj_max, log2fc_min, pct_in_min, median_in_min, log1p_in_min, nzmean_in_min, zero_in_max]),
])
out = W.Output()

def _refresh_celltypes(*_):
    panel = _current_panel()
    opts = list(panel.keys())
    default = "Tumor" if "Tumor" in panel else ("Tumor/Epithelial (pan)" if "Tumor/Epithelial (pan)" in panel else opts[0])
    ctype_dd.options = opts
    ctype_dd.value = default

panel_dd.observe(_refresh_celltypes, names="value")

# ---------- Helpers ----------
NEEDED_BASE_COLS = [
    "names","scores","logfoldchanges","pvals_adj","pct_diff","pct_in_cluster","pct_in_rest",
    "cluster","frac_in_dataset",
    # new stats
    "mean_in","mean_rest","nz_mean_in","nz_mean_rest","median_in_s","median_rest_s",
    "q10_in_s","q90_in_s","q10_rest_s","q90_rest_s","mean_log1p_in_s","mean_log1p_rest_s",
    "zero_frac_in","zero_frac_rest","diff_mean","log2fc_mean"
]

def _load_deg(res_key: str) -> pd.DataFrame:
    df = pd.read_csv(res2file[res_key])
    for c in NEEDED_BASE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df["cluster"] = df["cluster"].astype(str)
    df["names"]   = df["names"].astype(str)
    return df

def _prepare_values(df: pd.DataFrame, val_col: str) -> pd.DataFrame:
    """Return a copy with a numeric column V ready for pivot/agg."""
    X = df.copy()
    if val_col == "-log10(pvals_adj)":
        V = -np.log10(X["pvals_adj"].astype(float))
        V.replace([np.inf, -np.inf], np.nan, inplace=True)
        return X.assign(V=V)

    # deltas composed on the fly
    if val_col == "nz_mean_delta":
        return X.assign(V = X["nz_mean_in"] - X["nz_mean_rest"])
    if val_col == "median_delta_s":
        return X.assign(V = X["median_in_s"] - X["median_rest_s"])
    if val_col == "mean_log1p_delta_s":
        return X.assign(V = X["mean_log1p_in_s"] - X["mean_log1p_rest_s"])
    if val_col == "zero_frac_delta":
        return X.assign(V = X["zero_frac_in"] - X["zero_frac_rest"])

    # direct pass-through columns
    if val_col in X.columns:
        return X.rename(columns={val_col: "V"})

    raise ValueError(f"Unknown value column: {val_col}")

def _build_matrix(df: pd.DataFrame, genes, val_col, agg):
    dfv = _prepare_values(df, val_col)
    aggfn = {"max":"max","mean":"mean","median":"median"}[agg]
    aggdf = dfv.groupby(["cluster","names"], as_index=False)["V"].agg(aggfn)
    clusters = sorted(aggdf["cluster"].unique().tolist())
    mat = aggdf.pivot(index="cluster", columns="names", values="V").reindex(index=clusters, columns=genes)
    return mat.astype(float)

def _apply_scale(mat: pd.DataFrame, mode: str):
    M = mat.copy()
    if mode == "raw": return M
    if mode == "zscore_by_gene":    return (M - M.mean(0)) / (M.std(0).replace(0, np.nan))
    if mode == "zscore_by_cluster": return (M.sub(M.mean(1), axis=0)).div(M.std(1).replace(0, np.nan), axis=0)
    if mode == "zscore_global":
        mu, sd = np.nanmean(M.values), np.nanstd(M.values)
        return (M - mu) / (sd if sd else np.nan)
    if mode == "minmax_by_gene":
        mn, mx = M.min(0), M.max(0)
        return (M - mn) / (mx - mn).replace(0, np.nan)
    if mode == "minmax_by_cluster":
        mn, mx = M.min(1), M.max(1)
        return (M.sub(mn, axis=0)).div((mx - mn).replace(0, np.nan), axis=0)
    if mode == "minmax_global":
        mn, mx = np.nanmin(M.values), np.nanmax(M.values)
        return (M - mn) / (mx - mn) if mx != mn else M * np.nan
    return M

def _apply_row_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level filters on cluster×gene rows. NaN disables a given filter."""
    Q = df.copy()
    def _isnum(x): 
        try: return not np.isnan(float(x))
        except: return False

    if _isnum(p_adj_max.value):
        Q = Q[Q["pvals_adj"] <= float(p_adj_max.value)]
    if _isnum(log2fc_min.value):
        Q = Q[Q["log2fc_mean"] >= float(log2fc_min.value)]
    if _isnum(pct_in_min.value):
        Q = Q[Q["pct_in_cluster"] >= float(pct_in_min.value)]  # 0..100
    if _isnum(median_in_min.value):
        Q = Q[Q["median_in_s"] >= float(median_in_min.value)]
    if _isnum(log1p_in_min.value):
        Q = Q[Q["mean_log1p_in_s"] >= float(log1p_in_min.value)]
    if _isnum(nzmean_in_min.value):
        Q = Q[Q["nz_mean_in"] >= float(nzmean_in_min.value)]
    if _isnum(zero_in_max.value):
        Q = Q[Q["zero_frac_in"] <= float(zero_in_max.value)]    # 0..1
    return Q

# ---------- Main drawing ----------
def _draw(*_):
    with out:
        out.clear_output(wait=True)
        res_key = res_dd.value
        genes   = _current_panel()[ctype_dd.value]
        df      = _load_deg(res_key)

        # keep clusters ≥0.5% if available
        if hide_small.value and df["frac_in_dataset"].notna().any():
            big = (df[["cluster","frac_in_dataset"]].dropna()
                   .groupby("cluster", as_index=False)["frac_in_dataset"].max())
            keep = set(big.loc[big["frac_in_dataset"] >= 0.005, "cluster"])
            df = df[df["cluster"].isin(keep)]

        sub = df[df["names"].isin(genes)].copy()
        if sub.empty:
            display(HTML(f"<i>No matching genes in {res_key} for {ctype_dd.value}.</i>"))
            return

        # ---- NEW: apply filters ----
        sub = _apply_row_filters(sub)
        if sub.empty:
            display(HTML("<i>All rows filtered out by thresholds.</i>"))
            return

        # ---- build matrix and scale ----
        mat = _build_matrix(sub, genes, value_dd.value, agg_dd.value)
        mat = _apply_scale(mat, scale_dd.value)

        # --- Clean numeric matrix ---
        mat = mat.replace([np.inf, -np.inf], np.nan)
        mat = mat.dropna(axis=0, how='all').dropna(axis=1, how='all')
        if mat.empty:
            display(HTML("<i>Matrix became empty after filtering or NaN removal.</i>"))
            return
        if np.all(np.isnan(mat.values)):
            display(HTML("<i>All values are NaN — check filters or scaling.</i>"))
            return
        mat = mat.fillna(0)

        # Keep sensible order if clustering off
        if not row_cl_cb.value:
            order = np.argsort(-np.nanmean(mat.values, axis=1))
            mat = mat.iloc[order, :]
        if not col_cl_cb.value:
            # keep panel order for columns (but only those present)
            keep_cols = [g for g in genes if g in mat.columns]
            mat = mat.loc[:, keep_cols]

        vmin = None if np.isnan(vmin_ft.value) else float(vmin_ft.value)
        vmax = None if np.isnan(vmax_ft.value) else float(vmax_ft.value)

        try:
            g = sns.clustermap(
                mat,
                cmap=cmap_dd.value,
                vmin=vmin, vmax=vmax,
                row_cluster=row_cl_cb.value,
                col_cluster=col_cl_cb.value,
                linewidths=0,
                cbar_kws={"label": f"{value_dd.value} ({scale_dd.value})"},
                figsize=(max(7, 0.65*mat.shape[1]), max(4, 0.4*mat.shape[0])),
                dendrogram_ratio=(.12, .12),
            )
            g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)
            g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=8)
            g.fig.suptitle(f"{res_key} — {ctype_dd.value} — {panel_dd.value}", y=1.02)
            display(g.fig)
            plt.close(g.fig)
        except ValueError as e:
            display(HTML(f"<b style='color:red'>Error:</b> {e}<br><i>Try disabling clustering or change scaling.</i>"))

# ---------- Reactivity ----------
for w in [panel_dd, res_dd, ctype_dd, value_dd, agg_dd, scale_dd, cmap_dd,
          hide_small, row_cl_cb, col_cl_cb, vmin_ft, vmax_ft,
          p_adj_max, log2fc_min, pct_in_min, median_in_min, log1p_in_min, nzmean_in_min, zero_in_max]:
    w.observe(_draw, names="value")

display(ui, out)
_refresh_celltypes()
_draw()
