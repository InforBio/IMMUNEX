import os
import scanpy as sc
import pandas as pd; import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.sparse
sys.path.append("/opt/Banksy_py")
import banksy as banksy
import time; import random
from banksy.initialize_banksy import initialize_banksy
from banksy.run_banksy import run_banksy_multiparam
from banksy_utils.color_lists import spagcn_color
from banksy_utils.plot_utils import plot_qc_hist, plot_cell_positions
from banksy_utils.load_data import load_adata, display_adata
from banksy_utils.filter_utils import filter_cells
from banksy_utils.filter_utils import normalize_total, filter_hvg, print_max_min
from banksy_utils.plot_utils import plot_qc_hist, plot_cell_positions
from banksy_utils.load_data import display_adata
from banksy_utils.filter_utils import filter_cells, normalize_total
from datetime import datetime
import numpy as np
import seaborn as sns

# Define paths
base_dir = "/scratch/Projects/IMMUNEX/results/intermediate/segmented_adatas"
sample_dirs = [
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d))
]

for sample_id in tqdm(sample_dirs, desc="Banksy QC + Clustering"):
    try:
        sample_path = os.path.join(base_dir, sample_id, f"adata_{sample_id}_cells.h5ad")
        output_dir = os.path.join(base_dir, sample_id)

        if not os.path.exists(sample_path):
            print(f"Skipping {sample_id}: file not found.")
            continue
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Load adata
        adata_cells = sc.read_h5ad(sample_path)

        if not scipy.sparse.issparse(adata_cells.X):
            adata_cells.X = scipy.sparse.csr_matrix(adata_cells.X)

        display_adata(adata_cells)

        # QC
        adata_cells.var_names_make_unique()
        adata_cells.var["mt"] = adata_cells.var_names.str.startswith("MT-")
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        sc.pp.calculate_qc_metrics(adata_cells, qc_vars=["mt"], log1p=True, inplace=True)

        hist_bin_options = ['auto', 50, 80, 100]
        plot_qc_hist(adata_cells, 800, 1000, 0)
        plt.savefig(os.path.join(output_dir, f"{sample_id}_qc_raw_hist.png"))
        plt.close()

        adata_filtred = filter_cells(adata_cells.copy(), min_count=20, max_count=1500, MT_filter=50, gene_filter=3)
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        plot_qc_hist(adata_filtred, 800, 1000, 0)
        plt.savefig(os.path.join(output_dir, f"{sample_id}_qc_filtered_hist.png"))
        plt.close()

        # Spatial plotting
        adata_filtred.obs['spatial_0'] = adata_filtred.obsm['spatial'][:, 0]
        adata_filtred.obs['spatial_1'] = adata_filtred.obsm['spatial'][:, 1]
        plot_cell_positions(
            adata=adata_filtred,
            raw_x=adata_filtred.obsm['spatial'][:, 0],
            raw_y=adata_filtred.obsm['spatial'][:, 1],
            coord_keys=('spatial_0', 'spatial_1'),
            s=1,
            alpha=0.3,
            label1="Original adata",
            label2="Filtered adata"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{sample_id}_spatial_qc_filtered.png"))
        plt.close()
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Plot filtered-out cells
        original_indices = set(adata_cells.obs_names)
        filtered_indices = set(adata_filtred.obs_names)
        filtered_out_indices = original_indices - filtered_indices
        adata_out = adata_cells[list(filtered_out_indices)].copy()
        adata_out.obs['spatial_0'] = adata_out.obsm['spatial'][:, 0]
        adata_out.obs['spatial_1'] = adata_out.obsm['spatial'][:, 1]

        plt.figure(figsize=(8, 8))
        plt.scatter(adata_out.obs['spatial_0'], adata_out.obs['spatial_1'], s=1, c='red', label='Filtered cells')
        plt.axis("equal")
        plt.title("Spatial positions of filtered cells")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{sample_id}_filtered_cells_pos.png"))
        plt.close()
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                        
        output_base_dir = "/scratch/Projects/IMMUNEX/results/intermediate/segmented_adatas/"
        sample_output_dir = os.path.join(output_base_dir, sample_id)
        save_dir = sample_output_dir
        os.makedirs(save_dir, exist_ok=True)
        os.chdir(save_dir) 
        # Embedding plots before clustering
        sc.pl.embedding(
            adata_cells,
            basis="spatial",
            color=['total_counts_mt', 'n_counts', 'n_bins', 'n_genes_by_counts'],
            size=5,
            alpha=0.6,
            show=False,
            save=f"{sample_id}_embedding_precluster.png"
        )
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Normalize and HVG
        adata_normalized = normalize_total(adata_filtred.copy())
        sc.pp.log1p(adata_normalized)
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        sc.pp.highly_variable_genes(adata_normalized, n_top_genes=1000, flavor='seurat')

        nsclc_marker_genes = sorted(list(set([
            # T cells
            'CD3D', 'CD3E', 'CD3G', 'CD2', 'CD7', 'TRAC', 'JUNB', 'S100A4', 'CD52', 'PFN1P1',
            'CD81', 'EEF1B2P3', 'CXCR4', 'CREM', 'IL32', 'TGIF1',
        
            # CD4+ T cells
            'CD4', 'IL7R', 'CCR7', 'SELL', 'FOXP3', 'CTLA4', 'TIGIT',
        
            # CD8+ T cells
            'CD8A', 'CD8B', 'GZMA', 'GZMB', 'PRF1', 'LAG3', 'PDCD1',
        
            # Regulatory T cells
            'IL2RA', 'IKZF2',
        
            # B cells
            'CD19', 'MS4A1', 'CD79A', 'CD79B', 'IGHD', 'IGHM', 'CD22', 'CD24',
            'CD74', 'HMGA1', 'CD52', 'PTPRC', 'HLA-DRA', 'CXCR4', 'SPCS3', 'LTB', 'IGKC',
        
            # Plasma cells
            'MZB1', 'JCHAIN', 'SDC1', 'XBP1',
        
            # NK cells
            'NCAM1', 'KLRB1', 'KLRD1', 'NKG7', 'GNLY', 'PRF1', 'FGFBP2',
            'IL32', 'FHL2', 'IL2RG', 'CD69', 'HOPX',
        
            # Dendritic cells
            'ITGAX', 'CD1C', 'CLEC9A', 'CD83', 'LAMP3', 'CCR7', 'HLA-DRA', 'BATF3',
        
            # Monocytes / Macrophages
            'CD14', 'CD68', 'LYZ', 'FCN1', 'S100A8', 'S100A9', 'ITGAM', 'MRC1', 'CD163',
            'NOS2', 'IL1B', 'TNF', 'ARG1', 'CD206', 'IL10',
        
            # Neutrophils
            'FCGR3B', 'CSF3R', 'ELANE', 'MPO', 'CEACAM8',
        
            # Endothelial
            'PECAM1', 'VWF', 'CDH5', 'KDR', 'CLDN5', 'ESAM', 'ENG', 'CD34', 'PROM1', 'PDPN',
            'TEK', 'FLT1', 'VCAM1', 'PTPRC', 'MCAM', 'ICAM1', 'FLT4',
        
            # Fibroblasts / CAFs
            'PDGFRA', 'PDGFRB', 'COL1A1', 'COL1A2', 'COL3A1', 'COL5A2', 'ACTA2', 'TAGLN', 'FAP', 'POSTN',
            'TCF21', 'FN',
        
            # Pericytes / Smooth muscle
            'RGS5', 'MYL9', 'MYLK', 'FHL2', 'ITGA1', 'EHD2', 'OGN', 'SNCG', 'FABP4',
        
            # Epithelial / Alveolar / Cancer cells
            'EPCAM', 'KRT8', 'KRT18', 'KRT19', 'SFTPC', 'SFTPB', 'AGER', 'PDPN', 'KRT7', 'KRT5', 'KRT14',
            'TP63', 'NGFR', 'MUC1', 'MUC16', 'BIRC5', 'MYC', 'SOX2', 'CCND1', 'CDKN2A', 'PD-L1',
            'EGFR', 'KRAS', 'TP53', 'TTF-1', 'NKX2-1', 'CEACAM5', 'CDH1', 'CLDN1',
        
            # Club/secretory cells
            'SCGB1A1', 'SCGB3A2', 'MUC5B',
        
            # Ciliated cells
            'FOXJ1', 'TPPP3', 'PIFO', 'DNAH5',
        
            # Goblet cells
            'MANF', 'KRT7', 'AQP3', 'AGR2', 'BACE2', 'TFF3', 'PHGR1', 'MUC4', 'MUC13', 'GUCA2A',
        
            # Enterocytes
            'CD55', 'ELF3', 'PLIN2', 'GSTM3', 'KLF5', 'CBR1', 'APOA1', 'CA1', 'PDHA1', 'EHF',
        
            # Enteroendocrine
            'NUCB2', 'FABP5', 'CPE', 'ALCAM', 'GCG', 'SST', 'CHGB', 'IAPP', 'CHGA', 'ENPP2',
        
            # Crypt cells / stem-like
            'HOPX', 'SLC12A2', 'MSI1', 'SMOC2', 'OLFM4', 'ASCL2', 'PROM1', 'BMI1', 'EPHB2', 'LRIG1',
        
            # Cancer stem cells
            'CD44', 'PROM1', 'ALDH1A1', 'ITGA6',
        
            # Immune checkpoints / tumor markers
            'PDCD1', 'CTLA4', 'HAVCR2', 'TIGIT', 'LAG3', 'CD274', 'PDCD1LG2',
            'HLA-A', 'HLA-B', 'HLA-C', 'HLA-DQA1', 'HLA-DPB1'
        ])))

        hvg_genes = adata_normalized.var_names[adata_normalized.var['highly_variable']].tolist()
        combined_genes = list(set(hvg_genes + nsclc_marker_genes))
        genes_to_keep = [g for g in combined_genes if g in adata_normalized.var_names]
        adata_normalized = adata_normalized[:, genes_to_keep]

        
        display_adata(adata_normalized)
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        adata_normalized.obs['n_bins'] = adata_normalized.obs['n_bins'].astype(float)
        sns.scatterplot(x=adata_normalized.obs['n_bins'], y=adata_normalized.obs['total_counts'])
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(output_dir, f"{sample_id}-bins_count_before_regression.png"))
        plt.close()
        
        adata_normalized.obs['n_bins'] = adata_normalized.obs['n_bins'].astype(float)
        sc.pp.regress_out(adata_normalized, ['n_bins'])
        
        sc.pp.calculate_qc_metrics(adata_normalized, inplace=True)
        sns.scatterplot(x=adata_normalized.obs['n_bins'], y=adata_normalized.obs['total_counts'])

        sns.scatterplot(x=adata_normalized.obs['n_bins'], y=adata_normalized.obs['total_counts'])
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(output_dir, f"{sample_id}-bins_count_after_regression.png"))
        plt.close()


        # PCA
        sc.tl.pca(adata_normalized, svd_solver='arpack', n_comps=100)
        sc.pl.pca_variance_ratio(adata_normalized, log=True, n_pcs=100)
        plt.savefig(os.path.join(output_dir, f"{sample_id}_pca_variance_ratio.png"))
        plt.close()
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Clustering
        NNei = 30
        res = 0.5
        nPCs = 15

        cumulative_variance = np.cumsum(adata_normalized.uns['pca']['variance_ratio'])
        n_pcs = np.argmax(cumulative_variance >= 0.90) + 1
        print(f"Using {n_pcs} PCs to explain 90% of variance.")

        sc.pp.neighbors(adata_normalized, n_neighbors=NNei, n_pcs=n_pcs)
        leiden_key = f'nn_{NNei} leiden_{res}'
        sc.tl.leiden(adata_normalized, resolution=res, key_added=leiden_key)
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        sc.pl.embedding(
            adata_normalized,
            basis="spatial",
            color=leiden_key,
            size=5,
            alpha=0.6,
            title=f"Spatial clusters (Leiden res={res})",
            show=False,
            save=f"{sample_id}_Spatial_Clusters.png"

        )
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(output_dir, f"{sample_id}_spatial_clusters_leiden.png"))
        plt.close()
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        sc.tl.umap(adata_normalized)
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        sc.pl.umap(
            adata_normalized,
            color=leiden_key,
            title=f'nn_{NNei} Leiden {res}',
            size=10,
            show=False,
            save=f"{sample_id}_UMAP.png"
        )
        plt.savefig(os.path.join(output_dir, f"{sample_id}_umap_clusters_leiden.png"))
        plt.close()
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Save final adata
        adata_normalized.write(os.path.join(output_dir, f"{sample_id}_clustered.h5ad"))
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    except Exception as e:
        print(f"❌ Error in {sample_id}: {e}")
