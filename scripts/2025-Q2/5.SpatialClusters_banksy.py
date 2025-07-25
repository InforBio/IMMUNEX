import os; import sys
from datetime import datetime
import matplotlib.pyplot as plt
import scanpy as sc
import scipy.sparse
from tqdm import tqdm
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
    
from banksy_utils.load_data import display_adata
from banksy_utils.plot_utils import plot_qc_hist
from banksy_utils.filter_utils import filter_cells, normalize_total
from banksy.initialize_banksy import initialize_banksy
from banksy.embed_banksy import generate_banksy_matrix
from banksy_utils.umap_pca import pca_umap
from banksy.cluster_methods import run_Leiden_partition
from banksy.plot_banksy import plot_results

# Define input/output paths
input_dir = "/scratch/Projects/IMMUNEX/results/intermediate/segmented_adatas"
output_dir = "/scratch/Projects/IMMUNEX/results/intermediate/banksy_clustered/"
os.makedirs(output_dir, exist_ok=True)

skip_samples = []
# Initialize a dictionary to hold loaded AnnData objects
adata_dict = {}

# Loop over folders and look for adata files
for folder_name in os.listdir(input_dir):
    sample_folder = os.path.join(input_dir, folder_name)
    if os.path.isdir(sample_folder):
        for file_name in os.listdir(sample_folder):
            file = file_name
            sample_id = file.replace("adata_", "").replace("_cells.h5ad", "")
            if sample_id in skip_samples:
                print('⏭️ Skiping sample:', sample_id, '⏭️','\n'*8)
                continue
            if file.endswith("_cells.h5ad") and file_name.startswith("adata_"):
                sample_id = file.replace("adata_", "").replace("_cells.h5ad", "")
                file_path = os.path.join(sample_folder, file_name)
                print(f"📦 Loading {sample_id} from: {file_path}")
                        
                sample_output_dir = os.path.join(input_dir, sample_id)
                os.makedirs(sample_output_dir, exist_ok=True)
        
                print(f"\n🔄 Processing sample: {sample_id} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
                file_path = os.path.join(sample_output_dir, file)
                adata_cells = sc.read(file_path)
        
                if not scipy.sparse.issparse(adata_cells.X):
                    print("⚙️  Converting expression matrix to sparse format")
                    adata_cells.X = scipy.sparse.csr_matrix(adata_cells.X)
        
                print("📋 Initial AnnData summary:")
                display_adata(adata_cells)
        
                print("🧪 Calculating QC metrics...")
                adata_cells.var_names_make_unique()
                adata_cells.var["mt"] = adata_cells.var_names.str.startswith("MT-")
                sc.pp.calculate_qc_metrics(adata_cells, qc_vars=["mt"], log1p=True, inplace=True)
        
                plot_qc_hist(adata_cells, 800, 1000, 0)
                plt.savefig(os.path.join(sample_output_dir, f"{sample_id}_qc_raw_hist.png"))
                plt.close()
        
                print("🔍 Filtering low-quality cells...")
                adata_filtred = filter_cells(adata_cells.copy(), min_count=20, max_count=1500, MT_filter=50, gene_filter=3)
        
                adata_filtred.obs['spatial_0'] = adata_filtred.obsm['spatial'][:, 0]
                adata_filtred.obs['spatial_1'] = adata_filtred.obsm['spatial'][:, 1]
        
                print("🧼 Normalizing and log-transforming data...")
                adata_normalized = normalize_total(adata_filtred.copy())
                sc.pp.log1p(adata_normalized)
        
                print("🔬 Selecting highly variable genes (HVGs)...")
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
                print("📋 HVG-filtered AnnData summary:")
                display_adata(adata_normalized)
        
                print("📉 Regressing out 'n_bins' to reduce segmentation bias...")
                adata_normalized.obs['n_bins'] = adata_normalized.obs['n_bins'].astype(float)
                sc.pp.regress_out(adata_normalized, ['n_bins'])
        
                # BANKSY
                print("🧭 Initializing BANKSY...")
                k_geom = 7
                max_m = 1
                nbr_weight_decay = "scaled_gaussian"
                coord_keys = ('spatial_0', 'spatial_1', 'spatial')
        
                banksy_dict = initialize_banksy(
                    adata_normalized,
                    coord_keys,
                    k_geom,
                    nbr_weight_decay=nbr_weight_decay,
                    max_m=max_m,
                    plt_edge_hist=False,
                    plt_nbr_weights=False,
                    plt_agf_angles=False,
                    plt_theta=False,
                )
        
                lambda_list = [0.2]
                pca_dims = [20]
                resolutions = [1]
        
                print("🧮 Generating BANKSY matrix...")
                banksy_dict, banksy_matrix = generate_banksy_matrix(
                    adata_normalized, banksy_dict, lambda_list, max_m
                )
        
                print("📊 Performing PCA/UMAP embedding...")
                pca_umap(banksy_dict, pca_dims=pca_dims, add_umap=True, plt_remaining_var=False)
        
                print("🔗 Running Leiden clustering...")
                results_df, max_num_labels = run_Leiden_partition(
                    banksy_dict,
                    resolutions,
                    num_nn=25,
                    num_iterations=-1,
                    partition_seed=123,
                    match_labels=True,
                )
        
                print("🖼️ Saving cluster plots...")
                weights_graph = banksy_dict['scaled_gaussian']['weights'][0]
                plot_results(
                    results_df,
                    weights_graph,
                    c_map='rainbow',
                    match_labels=True,
                    coord_keys=coord_keys,
                    max_num_labels=max_num_labels,
                    save_path=os.path.join(sample_output_dir, 'banksy_results'),
                    save_fig=True,
                    save_seperate_fig=True,
                )
        
                print("💾 Saving final clustered data...")
                # adata_banksy = banksy_dict['scaled_gaussian']['weights']["adata"]
                # Automatically extract the first adata from banksy_dict
                print(banksy_dict)

                try:
                    first_decay = next(iter(banksy_dict))
                    first_lambda = next(k for k in banksy_dict[first_decay] if isinstance(k, (float, int)))
                    adata_banksy = banksy_dict[first_decay][first_lambda]['adata']
                except:
                    adata_banksy = banksy_dict['scaled_gaussian'][0.2]['adata']
                adata_banksy.write(os.path.join(sample_output_dir, f"{sample_id}_clustered_banksy.h5ad"))
                print(f"✅ Done: {sample_id} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n🎉 All samples processed successfully with the BANKSY pipeline.")
