%reset -f

import resource

# Set memory limit to 8 GB (in bytes)
soft_limit = 400 * 1024 ** 3  # X GB
hard_limit = 550 * 1024 ** 3
resource.setrlimit(resource.RLIMIT_AS, (soft_limit, hard_limit))


#!/usr/bin/env python
import os
import re
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import bin2cell as b2c
import tifffile
import itertools


sample_ids = ['IMMUNEX012','IMMUNEX004','IMMUNEX015','IMMUNEX001']
# Base directory for input and output
base_dir = Path("/scratch/Projects/IMMUNEX/results/intermediate/bin2cell_segmentation")
output_dir = Path("/scratch/Projects/IMMUNEX/results/intermediate/nuclei_adatas")

# Mask region to apply (same across samples for now)
mask_region = {{ "x": 250, "y": 250, "w": 100, "h": 100 }}

seg_config = f"{sample_id}_he0001_gex01"



for sample_id in sample_ids:

    rint(f"➡️  Processing sample: {sample_id}")
    sample_dir = base_dir / f"{sample_id}_{seg_config}"
    output_dir = output_dir /  f"{sample_id}_{seg_config}"
    output_dir.mkdir(exist_ok=True)

    # Load and filter bin data
    print("📥 Loading bin data...")
    adata_bins = sc.read(sample_dir / f"bins_{sample_id}.h5ad")
    adata_bins = adata_bins[ ~adata_bins.obs['on_tissue'] ]
    adata_bins.obs['source'] = (adata_bins.obs['labels_he'] != 0).astype(int) + (adata_bins.obs['labels_he_expanded'] != 0).astype(int)

    sc.pp.calculate_qc_metrics(adata_bins, inplace=True)
    sc.pl.violin(adata_bins, ['total_counts', 'n_genes_by_counts'], jitter=0.4, multi_panel=True, show=False)
    plt.savefig(output_dir / "violin_plot.png")
    plt.close()

    counts = adata_bins.obs['total_counts']
    sorted_counts = np.sort(counts)[::-1]
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(sorted_counts)), sorted_counts)
    plt.yscale('log')
    plt.xlabel('Barcode rank')
    plt.ylabel('Count depth')
    plt.title('Barcode Rank Plot')
    plt.savefig(output_dir / "barcode_rank_bins.png")
    plt.close()

    # Load nuclear cell data
    print("📥 Loading nuclear cell data...")
    adata_nuc = sc.read(sample_dir / f"cells_{sample_id}.h5ad")
    adata_nuc.var['mt'] = adata_nuc.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata_nuc, qc_vars=['mt'], inplace=True)

    counts = adata_nuc.obs['total_counts']
    sorted_counts = np.sort(counts)[::-1]
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(sorted_counts)), sorted_counts)
    plt.yscale('log')
    plt.xlabel('Barcode rank')
    plt.ylabel('Count depth')
    plt.title('Barcode Rank Plot (Nuclear)')
    plt.savefig(output_dir / "barcode_rank_nuclear.png")
    plt.close()

    # QC and filtering
    adata_nuc_f = adata_nuc.copy()
    print("🧹 Filtering genes and cells...")
    sc.pp.filter_genes(adata_nuc_f, min_cells=50)
    sc.pp.filter_cells(adata_nuc_f, min_counts=10)
    sc.pp.filter_cells(adata_nuc_f, min_genes=10)
    sc.pp.filter_cells(adata_nuc_f, max_genes=700)
    adata_nuc_f.var['mt'] = adata_nuc_f.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata_nuc_f, qc_vars=['mt'], inplace=True)
    adata_nuc_f = adata_nuc_f[adata_nuc_f.obs.pct_counts_mt < 10, :]

    # cutoff = np.percentile(adata_nuc_f.obs['total_counts'], 95)
    # adata_nuc_f = adata_nuc_f[adata_nuc_f.obs['total_counts'] < cutoff, :]

    # Normalize and find HVGs
    print("📊 Normalizing and scaling...")
    sc.pp.normalize_total(adata_nuc_f, target_sum=1e4)
    sc.pp.log1p(adata_nuc_f)
    adata_nuc_f.raw = adata_nuc_f.copy()
    sc.pp.highly_variable_genes(adata_nuc_f, n_top_genes=5000, flavor='cell_ranger')

    adata_hvgs = adata_nuc_f[:, adata_nuc_f.var['highly_variable']].copy()
    sc.pp.scale(adata_hvgs, zero_center=False)
    sc.tl.pca(adata_hvgs, svd_solver='arpack')

    sdata = adata_hvgs.copy()
    mask = (
        (sdata.obs['array_row'] >= mask_region["y"]) &
        (sdata.obs['array_row'] < mask_region["y"] + mask_region["h"]) &
        (sdata.obs['array_col'] >= mask_region["x"]) &
        (sdata.obs['array_col'] < mask_region["x"] + mask_region["w"])
    )

    # Clustering
    sc.pp.neighbors(sdata, n_neighbors=20, n_pcs=12)
    print("🔗 Running Leiden clustering...")

    clustered_adata = Path(f"{output_dir}/_bins_clusters.h5ad")
    if not clustered_adata.exists():
            
        import resolutiontree as rt
        resolutions = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
        
        # If you don't want to modify the original AnnData object, make a copy first
        
        # Step 1: Find optimal resolution with DEG analysis
        rt.cluster_resolution_finder(adata_new,
                                     resolutions=resolutions,
                                     n_top_genes=3,
                                     min_cells=2,
                                     deg_mode="within_parent"
                                     )
        
        
        # Step 2: Visualize the hierarchical clustering tree
        rt.cluster_decision_tree(adata_new, resolutions=resolutions, 
                                output_settings = {
                                    "output_path": f"{output_dir}/cluster_decision_tree.png",
                                    "draw": False,
                                    "figsize": (12, 6),
                                    "dpi": 300
                                    },
                                node_style = {
                                    "node_size": 500,
                                    "node_colormap": None,
                                    "node_label_fontsize": 12
                                    },
                                edge_style = {
                                    "edge_color": "parent",
                                    "edge_curvature": 0.01,
                                    "edge_threshold": 0.01,
                                    "show_weight": True,
                                    "edge_label_threshold": 0.05,
                                    "edge_label_position": 0.8,
                                    "edge_label_fontsize": 8
                                    },
                                gene_label_settings = {
                                    "show_gene_labels": True,
                                    "n_top_genes": 2,
                                    "gene_label_threshold": 0.001,
                                    "gene_label_style": {"offset":0.5, "fontsize":8},
                                    },
                                level_label_style = {
                                    "level_label_offset": 15,
                                    "level_label_fontsize": 12
                                    },
                                title_style = {
                                    "title": "Hierarchical Leiden Clustering",
                                    "title_fontsize": 20
                                    },
                                layout_settings = {
                                    "node_spacing": 5.0,
                                    "level_spacing": 1.5
                                    },
                                clustering_settings = {
                                    "prefix": "leiden_res_",
                                    "edge_threshold": 0.05
                                    }
                            )
    
        
        adata_new.write( clustered_adata )
    else:
        adata_new = sc.read( clustered_adata )
        
    sdata = adata_new.copy()
    
    print("🌐 Computing UMAP...")
    sc.tl.umap(sdata, key_added = 'leiden_res_0.1')
    sc.pl.umap(sdata, show=False)
    plt.close()


    # Save output files (optional)
    print("💾 Saving annotated bin data...")
