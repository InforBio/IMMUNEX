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

# ---- Panels (same as your post; truncated if you like) ----
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

# Value options (unchanged; filters are independent)
VAL_OPTIONS = [
    "scores","logfoldchanges","pct_diff","pct_in_cluster","pct_in_rest","-log10(pvals_adj)",
    "mean_in","mean_rest","diff_mean","log2fc_mean",
    "nz_mean_in","nz_mean_rest","nz_mean_delta",
    "median_in_s","median_rest_s","median_delta_s",
    "mean_log1p_in_s","mean_log1p_rest_s","mean_log1p_delta_s",
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

# ---- NEW: gene-level MEAN filters (use mean_in) ----
mean_any_min  = W.FloatText(
    description="≥ mean in any cluster",
    description_tooltip="Keep genes whose max(mean_in across clusters) ≥ this",
    style={'description_width': '300px'},
    layout=W.Layout(width='400px'),
    value=0.05,
)


mean_mean_min = W.FloatText(
    value=0,
    description="≥ mean in all clusters",
    description_tooltip="Keep genes whose mean(mean_in across clusters) ≥ this",
    style={'description_width': '300px'},
    layout=W.Layout(width='400px'),

)


ui = W.VBox([
    W.HBox([res_dd, panel_dd, ctype_dd, value_dd]),
    W.HBox([scale_dd, cmap_dd, hide_small, row_cl_cb, col_cl_cb]),
    W.HBox([mean_any_min, mean_mean_min]),
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
NEEDED = [
    "names","cluster","frac_in_dataset","pvals_adj","pct_in_cluster","pct_in_rest","pct_diff",
    "scores","logfoldchanges",
    "mean_in","mean_rest","nz_mean_in","nz_mean_rest",
    "median_in_s","median_rest_s","mean_log1p_in_s","mean_log1p_rest_s",
    "zero_frac_in","zero_frac_rest","diff_mean","log2fc_mean"
]

def _load_deg(res_key: str) -> pd.DataFrame:
    df = pd.read_csv(res2file[res_key])
    for c in NEEDED:
        if c not in df.columns: df[c] = np.nan
    df["cluster"] = df["cluster"].astype(str)
    df["names"]   = df["names"].astype(str)
    return df

def _prepare_values(df: pd.DataFrame, val_col: str) -> pd.DataFrame:
    X = df.copy()
    if val_col == "-log10(pvals_adj)":
        V = -np.log10(X["pvals_adj"].astype(float))
        V.replace([np.inf, -np.inf], np.nan, inplace=True)
        return X.assign(V=V)
    if val_col == "nz_mean_delta":
        return X.assign(V = X["nz_mean_in"] - X["nz_mean_rest"])
    if val_col == "median_delta_s":
        return X.assign(V = X["median_in_s"] - X["median_rest_s"])
    if val_col == "mean_log1p_delta_s":
        return X.assign(V = X["mean_log1p_in_s"] - X["mean_log1p_rest_s"])
    if val_col == "zero_frac_delta":
        return X.assign(V = X["zero_frac_in"] - X["zero_frac_rest"])
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

def _apply_gene_mean_filters(sub: pd.DataFrame, selected_genes: list):
    """
    Compute per-gene stats across clusters using mean_in (full cluster mean).
    Keeps a gene if:
      - max mean across clusters ≥ mean_any_min (if set)
      - mean of means across clusters ≥ mean_mean_min (if set)
    Missing (gene, cluster) means are treated as 0.
    """
    if sub.empty:
        return sub, pd.DataFrame(columns=["min_mean","mean_mean","max_mean","n_clusters"])

    G = sub[sub["names"].isin(selected_genes)].copy()
    G["mean_in"] = pd.to_numeric(G["mean_in"], errors="coerce").fillna(0.0)
    G["cluster"] = G["cluster"].astype(str)
    all_clusters = sorted(sub["cluster"].astype(str).unique().tolist())

    idx = pd.MultiIndex.from_product([selected_genes, all_clusters], names=["names","cluster"])
    means = (G.set_index(["names","cluster"])["mean_in"]
               .reindex(idx, fill_value=0.0)
               .groupby(level="names")
               .agg(min_mean="min", mean_mean="mean", max_mean="max", n_clusters="count"))

    keep = pd.Series(True, index=means.index)
    if not np.isnan(mean_any_min.value):
        keep &= (means["max_mean"]  >= float(mean_any_min.value))
    if not np.isnan(mean_mean_min.value):
        keep &= (means["mean_mean"] >= float(mean_mean_min.value))

    kept_genes = means.index[keep].tolist()
    Gf = G[G["names"].isin(kept_genes)].copy()
    return Gf, means.loc[kept_genes]

# ---------- Main drawing ----------
def _draw(*_):
    with out:
        out.clear_output(wait=True)
        res_key = res_dd.value
        panel_genes = _current_panel()[ctype_dd.value]
        df = _load_deg(res_key)

        # keep clusters ≥0.5% if available
        if hide_small.value and df["frac_in_dataset"].notna().any():
            big = (df[["cluster","frac_in_dataset"]].dropna()
                   .groupby("cluster", as_index=False)["frac_in_dataset"].max())
            keep = set(big.loc[big["frac_in_dataset"] >= 0.005, "cluster"])
            df = df[df["cluster"].isin(keep)]

        # subset to panel genes
        sub = df[df["names"].isin(panel_genes)].copy()
        if sub.empty:
            display(HTML(f"<i>No matching genes in {res_key} for {ctype_dd.value}.</i>"))
            return

        # ---- NEW: gene filters based on mean_in across clusters ----
        sub, gene_stats = _apply_gene_mean_filters(sub, panel_genes)
        if sub.empty:
            display(HTML("<i>All genes filtered out by mean thresholds.</i>"))
            return

        genes_kept = [g for g in panel_genes if g in sub["names"].unique()]

        # ---- build matrix and scale ----
        mat = _build_matrix(sub, genes_kept, value_dd.value, agg_dd.value)
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

        # Row/col ordering
        if not row_cl_cb.value:
            order = np.argsort(-np.nanmean(mat.values, axis=1))
            mat = mat.iloc[order, :]
        if not col_cl_cb.value:
            mat = mat.loc[:, genes_kept]

        vmin = None if np.isnan(vmin_ft.value) else float(vmin_ft.value)
        vmax = None if np.isnan(vmax_ft.value) else float(vmax_ft.value)

        # Header showing how many genes survived
        kept_info = f"<small>Kept {len(genes_kept)}/{len(panel_genes)} genes"
        if not np.isnan(mean_any_min.value):  kept_info += f"; max(mean) ≥ {float(mean_any_min.value)}"
        if not np.isnan(mean_mean_min.value): kept_info += f"; mean(mean) ≥ {float(mean_mean_min.value)}"
        kept_info += "</small>"
        display(HTML(kept_info))

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
          mean_any_min, mean_mean_min]:
    w.observe(_draw, names="value")

display(ui, out)
_refresh_celltypes()
_draw()
