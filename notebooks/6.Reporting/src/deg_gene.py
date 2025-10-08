# -*- coding: utf-8 -*-
from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
from IPython.display import display, HTML
import ipywidgets as W
import matplotlib.pyplot as plt

# Read BASE_DIR from notebook (fallback = CWD)
BASE_DIR = globals().get("BASE_DIR", Path.cwd())

def _find_samples_de(base_dir: Path):
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
refresh_btn = W.Button(description="↻ Refresh", layout=W.Layout(width="110px"))
info_html = W.HTML()
header = W.HBox([sample_dd, refresh_btn])

def _set_info():
    if not _SAMPLES:
        info_html.value = f"<div style='color:#b00020'>No DE folders found under <code>{BASE_DIR}</code></div>"
    else:
        info_html.value = f"<small>Base: <code>{BASE_DIR}</code> — {len(_SAMPLES)} sample(s) with DE</small>"

def _res_from_path(p: Path) -> str:
    return p.stem.replace("DE_", "")   # e.g., resolution_leiden_0.5

def _refresh_samples(_=None):
    global _SAMPLES, resolutions, res2file
    _SAMPLES = _find_samples_de(BASE_DIR)
    sample_dd.options = sorted(_SAMPLES.keys())
    if sample_dd.options and sample_dd.value not in sample_dd.options:
        sample_dd.value = sample_dd.options[0]
    _set_info()
    # refresh resolutions for current sample
    if not sample_dd.value:
        resolutions = []; res2file = {}
        res_dd.options = []
        return
    files = _SAMPLES[sample_dd.value]["de_files"]
    resolutions = [_res_from_path(p) for p in files]
    res2file = { _res_from_path(p): p for p in files }
    res_dd.options = resolutions
    if resolutions:
        res_dd.value = resolutions[0]

refresh_btn.on_click(_refresh_samples)
_set_info()

# ---- Panels ----
# NSCLC / TLS marker panel (now includes Immune cells, Tumor, Stromal roll-ups)
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

# P.A. panel (as provided)
PANEL_PA = {
    "All" : ["KRT18","IRX2","NAPSA","SPINK13","KRT7","CAPN8","GPRC5A","TP63","CALML3","KRT5","PKP1","TESC","SPINK1","C9orf152","CD3D","CD3E","CD8A","CD8B","BCL6","CD81","POU2AF1","RGS13","IGHA1","IGHG1","JCHAIN","AKNA","ARHGAP25","CCL21","CD180","CD2","CD27","CD37","CD38","CLEC17A","CLEC9A","CLECL1","FAIM3","FAM65B","GIMAP4","MAP4K1","PAX5","TNFRSF17","TRAF3IP3","CD209","S100A1","S100A10","S100A11","S100A12","S100A13","S100A14","S100A16","S100A2","S100A3","S100A4","S100A5","S100A6","S100A7","S100A7L1","S100A8","S100A9","S100B","S100G","S100P","S100PBP","S100Z","ANLN","BRIP1","BUB1B","CASC5","CCNB1","CCNB2","CCNE2","CEP55","CKAP2L","DLGAP5","DTL","E2F8","ECT2","ESCO2","EXO1","EXOC6","FBXO5","FIGNL1","HELLS","HMMR","IARS","KIF11","KIF18A","KIF20A","KNTC1","MAD2L1","MASTL","MTHFD2","NCAPG2","NCAPH","NEIL3","NUF2","PRC1","PSAT1","RTKN2","ADRM1","AHSA1","C1GALT1C1","CCT5","CCT6B","CETN3","CSE1L","EIF2S1","GAL","GEMIN6","GPT2","KIAA0101","MND1","MPZL1","MRPS16","PCNA","PTRH2","RFC5","SPC25","TIMM13","TIMM8B","TK1","TUBB","TXNDC17","FCGR3A","FCN1","CD68","MSR1","APOE","C1QA","XCR1","CLEC9A","CD1C","CLEC10A","FCER1A","LILRA4","SPIB","IRF7","KIT","CPA3","CDH5","PECAM1","VWF","CCL21","FLT4","LYVE1","PROX1","ACTA2","COL13A1","COL14A1","MYH11","PDGFRA","PDGFRB","VIM","CNN1","CRYAB","DES","SYNPO2","ABTB1","AMPD2","CAMP","EMR4P","FPR1","FPR2","GPR77","MAEA","PROK2","SEC14L1","SEPX1","SLC25A37","TNFSF14","TREML4","VNN2","XPO6","BATF","CCR10","CCR8","CTLA4","FOXP3","IKZF2","IKZF4","IL10RA","IL2RA","IL32","TIGIT","TNFRSF18"] ,
    "Tumor cells 1": ["KRT18","IRX2","NAPSA","SPINK13","KRT7","CAPN8","GPRC5A"],
    "Tumor cells 2": ["TP63","CALML3","KRT5","PKP1","TESC","SPINK1","C9orf152","KRT7"],
    "T cells": ["CD3D","CD3E","CD8A","CD8B"],
    "Germinal_center_Bcells": ["BCL6","CD81","POU2AF1","RGS13"],
    "Plasma cells": ["IGHA1","IGHG1","JCHAIN"],
    "B_cells": ["AKNA","ARHGAP25","CCL21","CD180","CD2","CD27","CD37","CD38","CLEC17A","CLEC9A","CLECL1","FAIM3","FAM65B","GIMAP4","MAP4K1","PAX5","TNFRSF17","TRAF3IP3"],
    "Dendritic_cells": ["CD209","S100A1","S100A10","S100A11","S100A12","S100A13","S100A14","S100A16","S100A2","S100A3","S100A4","S100A5","S100A6","S100A7","S100A7L1","S100A8","S100A9","S100B","S100G","S100P","S100PBP","S100Z"],
    "Activated_CD4_T_cell": ["ANLN","BRIP1","BUB1B","CASC5","CCNB1","CCNB2","CCNE2","CEP55","CKAP2L","DLGAP5","DTL","E2F8","ECT2","ESCO2","EXO1","EXOC6","FBXO5","FIGNL1","HELLS","HMMR","IARS","KIF11","KIF18A","KIF20A","KNTC1","MAD2L1","MASTL","MTHFD2","NCAPG2","NCAPH","NEIL3","NUF2","PRC1","PSAT1","RTKN2"],
    "Activated_CD8_T_cell": ["ADRM1","AHSA1","C1GALT1C1","CCT5","CCT6B","CETN3","CSE1L","EIF2S1","GAL","GEMIN6","GPT2","KIAA0101","MND1","MPZL1","MRPS16","PCNA","PTRH2","RFC5","SPC25","TIMM13","TIMM8B","TK1","TUBB","TXNDC17"],
    "Myeloid cells": ["FCGR3A","FCN1","FCGR3A","FCN1","CD68","MSR1","APOE","C1QA","XCR1","CLEC9A","CD1C","CLEC10A","FCER1A","LILRA4","SPIB","IRF7","KIT","CPA3"],
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

def _current_panel(panel_name: str):
    return PANEL_NSCLC if panel_name == "NSCLC panel" else PANEL_PA

# ---------- Widgets ----------
res_dd     = W.Dropdown(options=[], description="Resolution")
panel_dd   = W.Dropdown(options=["NSCLC panel","P.A. panel"], value="NSCLC panel", description="Markers")
ctype_dd   = W.Dropdown(options=[], description="Cell type")
value_dd   = W.Dropdown(options=["scores","logfoldchanges","pct_diff","pct_in_cluster","pct_in_rest","-log10(pvals_adj)"],
                        value="scores", description="Value")
agg_dd     = W.Dropdown(options=["max","mean","median"], value="max", description="Aggregate")

scale_dd   = W.Dropdown(
    options=["raw","zscore_by_gene","zscore_by_cluster","zscore_global",
             "minmax_by_gene","minmax_by_cluster","minmax_global"],
    value="zscore_by_gene", description="Scale"
)
cmap_dd    = W.Dropdown(
    options=["RdBu_r","Spectral","viridis","plasma","magma","cividis","coolwarm",
             "bwr","seismic","turbo","cubehelix","rocket","mako","YlOrRd","YlGnBu"],
    value="RdBu_r", description="Colormap"
)

hide_small = W.Checkbox(value=True, description="Hide clusters <0.5%")
row_cl_cb  = W.Checkbox(value=True,  description="Cluster rows")
col_cl_cb  = W.Checkbox(value=True,  description="Cluster cols")
vmin_ft    = W.FloatText(value=np.nan, description="vmin")
vmax_ft    = W.FloatText(value=np.nan, description="vmax")

ui = W.VBox([
    W.HBox([sample_dd, refresh_btn]),
    info_html,
    W.HBox([res_dd, panel_dd, ctype_dd, value_dd, agg_dd]),
    W.HBox([scale_dd, cmap_dd, hide_small, row_cl_cb, col_cl_cb, vmin_ft, vmax_ft]),
])
out = W.Output()

# ---------- Helpers ----------
resolutions = []
res2file = {}

def _load_deg(res_key: str) -> pd.DataFrame:
    df = pd.read_csv(res2file[res_key])
    for c in ["names","scores","logfoldchanges","pvals_adj","pct_diff","pct_in_cluster","pct_in_rest","cluster","frac_in_dataset"]:
        if c not in df.columns: df[c] = np.nan
    df["cluster"] = df["cluster"].astype(str)
    df["names"]   = df["names"].astype(str)
    return df

def _build_matrix(df: pd.DataFrame, genes, val_col, agg):
    if val_col == "-log10(pvals_adj)":
        vals = -np.log10(df["pvals_adj"].astype(float))
        vals.replace([np.inf, -np.inf], np.nan, inplace=True)
        dfv = df.assign(V=vals)
    else:
        dfv = df.rename(columns={val_col: "V"})
    aggdf = dfv.groupby(["cluster","names"], as_index=False)["V"].agg({"max":"max","mean":"mean","median":"median"}[agg])
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

def _refresh_celltypes(*_):
    panel = _current_panel(panel_dd.value)
    opts = list(panel.keys())
    default = "Tumor" if "Tumor" in panel else ("Tumor/Epithelial (pan)" if "Tumor/Epithelial (pan)" in panel else opts[0])
    ctype_dd.options = opts
    ctype_dd.value = default

def _refresh_resolutions(*_):
    if not sample_dd.value:
        res_dd.options = []; return
    files = _SAMPLES[sample_dd.value]["de_files"]
    keys = [_res_from_path(p) for p in files]
    global resolutions, res2file
    resolutions = keys
    res2file = { _res_from_path(p): p for p in files }
    res_dd.options = keys
    if keys:
        res_dd.value = keys[0]

def _draw(*_):
    with out:
        out.clear_output(wait=True)
        if not (sample_dd.value and res_dd.value):
            display(HTML("<i>Select a sample and resolution.</i>"))
            return
        res_key = res_dd.value
        genes   = _current_panel(panel_dd.value)[ctype_dd.value]
        df      = _load_deg(res_key)

        if hide_small.value and df["frac_in_dataset"].notna().any():
            big = (df[["cluster","frac_in_dataset"]].dropna()
                   .groupby("cluster", as_index=False)["frac_in_dataset"].max())
            keep = set(big.loc[big["frac_in_dataset"] >= 0.005, "cluster"])
            df = df[df["cluster"].isin(keep)]

        sub = df[df["names"].isin(genes)].copy()
        if sub.empty:
            display(HTML(f"<i>No matching genes in {res_key} for {ctype_dd.value}.</i>"))
            return

        mat = _build_matrix(sub, genes, value_dd.value, agg_dd.value)
        mat = _apply_scale(mat, scale_dd.value)

        # --- Clean numeric matrix to avoid ValueError ---
        mat = mat.replace([np.inf, -np.inf], np.nan)
        mat = mat.dropna(axis=0, how='all').dropna(axis=1, how='all')
        if mat.empty:
            display(HTML("<i>Matrix became empty after removing NaNs — nothing to plot.</i>"))
            return
        if np.all(np.isnan(mat.values)):
            display(HTML("<i>All values are NaN — check filtering or scaling.</i>"))
            return
        mat = mat.fillna(0)

        # Keep sensible order if clustering off
        if not row_cl_cb.value:
            order = np.argsort(-np.nanmean(mat.values, axis=1))
            mat = mat.iloc[order, :]
        if not col_cl_cb.value:
            mat = mat.loc[:, genes]

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
            g.fig.suptitle(f"{sample_dd.value} — {res_key} — {ctype_dd.value} — {panel_dd.value}", y=1.02)
            display(g.fig)
            plt.close(g.fig)
        except ValueError as e:
            display(HTML(f"<b style='color:red'>Error:</b> {e}<br><i>Try disabling clustering or change scaling.</i>"))

# ---------- Reactivity ----------
sample_dd.observe(_refresh_resolutions, names="value")
panel_dd.observe(_refresh_celltypes, names="value")
for w in [res_dd, ctype_dd, value_dd, agg_dd, scale_dd, cmap_dd,
          hide_small, row_cl_cb, col_cl_cb, vmin_ft, vmax_ft]:
    w.observe(_draw, names="value")

display(ui, out)
_refresh_samples()
_refresh_celltypes()
_draw()
