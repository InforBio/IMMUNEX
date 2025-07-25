import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from anndata import AnnData
import scipy.sparse as sp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from datetime import datetime

# Directory containing the h5ad files
data_dir = "/scratch/Projects/IMMUNEX/segmentation/bin2cell/bin2cell_output_he0.01_gex0.05"


# Collect all subdirectories (one per sample) that contain h5ad files
h5ad_paths = []
for sample_folder in os.listdir(data_dir):
    sample_path = os.path.join(data_dir, sample_folder, "adata_processed.h5ad")
    if os.path.isfile(sample_path):
        h5ad_paths.append((sample_folder, sample_path))
print(h5ad_paths)

to_skip = []


# Loop through each file
for sample_id, path in tqdm(h5ad_paths, desc="Loading samples"):
    if sample_id in to_skip:
        print(f'skipping {sample_id}')
        continue
    else:
                
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        adata = sc.read(path)
        adata.raw = adata
        adata.uns['sample_id'] = sample_id
        
        # --- Your custom processing here ---
        print(f"Loaded {sample_id}: {adata.shape}")
        print(adata.obs.head())
        
        # Load the metadata
        metadata = pd.read_csv("~/rawdata/IMMUNEX/data/VisiumHD_18_2024_NSCLC.csv")
        print(metadata.sample())
        metadata_sample = metadata[metadata['Sample_code'] == sample_id]
        adata.uns["sample_metadata"] = metadata_sample.iloc[0].to_dict()
        print(adata.obs.sample()) # Display the first few rows of the observation data (cell metadata)


        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        #### Assess the seg.
        
        # Define a base output directory
        output_base_dir = "/scratch/Projects/IMMUNEX/results/intermediate/segmented_adatas/"
        sample_output_dir = os.path.join(output_base_dir, sample_id)
        os.makedirs(sample_output_dir, exist_ok=True)
        
        # Boolean masks from adata
        is_he = adata.obs['labels_he'] != 0
        is_he_exp = adata.obs['labels_he_expanded'] != 0
        is_gex = adata.obs['labels_gex'] != 0
        
        # Hierarchical categorization
        conditions = {
            "Shared": is_he_exp & is_gex,
            "HE (Nuclei) ": is_he & ~is_gex,
            "HE expansion only": is_he_exp & ~is_gex & ~is_he,
            "GEx only": is_gex & ~is_he_exp,
            "Unassigned": ~is_he_exp & ~is_gex
        }
        
        # Count bins in each category
        counts = {k: v.sum() for k, v in conditions.items()}
        
        # Create DataFrame
        df = pd.DataFrame(list(counts.items()), columns=["Category", "Count"])
        df["Percentage"] = df["Count"] / df["Count"].sum() * 100
        
        # Ensure specific order
        df["Category"] = pd.Categorical(df["Category"], categories=[
            "Shared", "HE (Nuclei) ", "HE expansion only", "GEx only", "Unassigned"
        ], ordered=True)
        df = df.sort_values("Category")
        
        # Define custom colors
        colors = {
            "Shared": "#66c2a5",
            "HE (Nuclei) ": "#fc8d62",
            "HE expansion only": "#ffd92f",
            "GEx only": "#8da0cb",
            "Unassigned": "#dddddd"
        }
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 2))
        left = 0
        for _, row in df.iterrows():
            ax.barh(y=0, width=row["Count"], left=left,
                    color=colors[row["Category"]], label=row["Category"])
            ax.text(left + row["Count"]/2, 0,
                    f"{int(row['Count'])}\n({row['Percentage']:.1f}%)",
                    ha="center", va="center", fontsize=9, color="black")
            left += row["Count"]
        
        ax.set_yticks([])
        ax.set_xlabel("Number of Bins")
        ax.set_title("Hierarchical Segmentation Overlap")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(sample_output_dir, f"{sample_id}_hierarchical_overlap.png")
        fig = plt.gcf()  # Get the current figure
        fig.savefig(output_path, dpi=300)
        # Optional: show the plot
        plt.show()

        
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Count bins per segmentation (excluding label 0 as background)
        label_col = 'labels_joint'
        adata_obs = adata.obs.copy()
        cell_bin_counts = adata_obs[label_col].value_counts()
        cell_bin_counts = cell_bin_counts[cell_bin_counts.index != 0]  # exclude background
        
        # Plot distribution before filtering
        plt.figure(figsize=(6, 4))
        plt.hist(cell_bin_counts.values, bins=100, color='skyblue')
        plt.axvline(np.percentile(cell_bin_counts.values, 99), color='red', linestyle='--', label='fltr. percentile')
        plt.title("Cell Size Distribution from HE+GEx (Before Filtering)")
        plt.xlabel("Number of Bins per Cell")
        plt.ylabel("Cell Count")
        plt.legend()
        plt.tight_layout()
        output_path = os.path.join(sample_output_dir, f"{sample_id}_cell dist_ from GEx and HE.png")
        fig = plt.gcf()  # Get the current figure
        fig.savefig(output_path, dpi=300)
        # Optional: show the plot
        plt.show()

        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Count bins per segmentation (excluding label 0 as background)
        label_col = 'labels_he_expanded'
        cell_bin_counts = adata_obs[label_col].value_counts()
        cell_bin_counts = cell_bin_counts[cell_bin_counts.index != 0]  # exclude background
        
        # Plot distribution before filtering
        plt.figure(figsize=(6, 4))
        plt.hist(cell_bin_counts.values, bins=100, color='skyblue')
        plt.axvline(np.percentile(cell_bin_counts.values, 99), color='red', linestyle='--', label='90th percentile')
        plt.title("Cell Size from HE (Before Filtering)")
        plt.xlabel("Number of Bins per Cell")
        plt.ylabel("Cell Count")
        plt.legend()
        plt.tight_layout()
        output_path = os.path.join(sample_output_dir, f"{sample_id}_cell dist_ from HE.png")
        fig = plt.gcf()  # Get the current figure
        fig.savefig(output_path, dpi=300)
        # Optional: show the plot
        plt.show()

        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Filtering logic
        threshold = np.percentile(cell_bin_counts.values, 99)
        valid_labels = cell_bin_counts[cell_bin_counts <= threshold].index
        mask = adata_obs[label_col].isin(valid_labels)
        
        # New filtered counts
        filtered_counts = adata_obs[mask][label_col].value_counts()
        
        # Plot distribution after filtering
        plt.figure(figsize=(6, 4))
        plt.hist(filtered_counts.values, bins=100, color='lightgreen')
        plt.title("Cell Size Distribution (After Filtering)")
        plt.xlabel("Number of Bins per Cell")
        plt.ylabel("Cell Count")
        plt.tight_layout()
        output_path = os.path.join(sample_output_dir, f"{sample_id}_cell dist After Filtering.png")
        fig = plt.gcf()  # Get the current figure
        fig.savefig(output_path, dpi=300)
        # plt.show()
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Filtering data
        mask = adata.obs['labels_joint'].isin(filtered_counts.index)
        adata_filtered = adata[mask].copy()


        print('ploting nuclei')
        # Parameters
        crop_axis_fraction = 0.02  # Fraction of the tissue area to crop around the center
        
        # Get spatial coordinates
        spatial_coords = adata_filtered.obsm['spatial'].copy()
        
        # Compute tissue bounds and center
        x_min, x_max = spatial_coords[:, 0].min(), spatial_coords[:, 0].max()
        y_min, y_max = spatial_coords[:, 1].min(), spatial_coords[:, 1].max()
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Define crop bounds
        dx = (x_max - x_min) * crop_axis_fraction / 2
        dy = (y_max - y_min) * crop_axis_fraction / 2
        x0, x1 = center_x - dx, center_x + dx
        y0, y1 = center_y - dy, center_y + dy
        
        # Subset AnnData to cropped region
        within_crop = (
            (spatial_coords[:, 0] >= x0) & (spatial_coords[:, 0] <= x1) &
            (spatial_coords[:, 1] >= y0) & (spatial_coords[:, 1] <= y1)
        )
        adata_cropped_bins = adata_filtered[within_crop, :].copy()
        
        # Keep only nuclei
        mask = adata_cropped_bins.obs['labels_he'] != 0
        adata_cropped_bins = adata_cropped_bins[mask].copy()
        
        # # ---- Plot 1: full image with crop box ----
        # fig, ax = plt.subplots(figsize=(6, 6))
        # sc.pl.spatial(
        #     adata_filtered,
        #     ax=ax,
        #     color=None,
        #     show=False,
        #     alpha_img=0.8,
        #     size=1
        # )
        # # Add rectangle showing crop area
        # rect = Rectangle(
        #     (x0, y0),
        #     x1 - x0,
        #     y1 - y0,
        #     linewidth=2,
        #     edgecolor='red',
        #     facecolor='none'
        # )
        # ax.add_patch(rect)
        # ax.set_title("Full image with crop area")
        # plt.tight_layout()

        # plt.tight_layout()
        # output_path = os.path.join(sample_output_dir, f"{sample_id}_crop pos.png")
        # fig = plt.gcf()  # Get the current figure
        # fig.savefig(output_path, dpi=300)
        
        # print('Crop position ploted')
        
        print('Ploting nuclei #1')
        # ---- Plot 2: cropped region with nuclei labels ----
        sc.pl.spatial(
            adata_cropped_bins,
            color=['labels_he'],
            size=1,
            cmap='nipy_spectral',
            alpha_img=0.9,
            show=False,
            title="Zoom on Nuclei (HE)"
        )
        plt.tight_layout()
        output_path = os.path.join(sample_output_dir, f"{sample_id} nuclei preview.png")
        fig = plt.gcf()  # Get the current figure
        fig.savefig(output_path, dpi=300)


        

        
        # Aggregation of bins
        adata_filtered.obs['cell_id'] = adata_filtered.obs['labels_joint'].astype(str)
        adata_cells = adata_filtered[adata_filtered.obs['labels_joint'] > 0].copy()
        adata_cells.var['mt'] = adata_cells.var_names.str.upper().str.startswith('MT-')
        
        # Use CSR for efficient row slicing
        X = adata_cells.X.tocsr()
        cell_ids = adata_cells.obs['cell_id'].values
        print('Use CSR for efficient row slicing')
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Map cell_id to integer index
        unique_ids, inverse_idx = np.unique(cell_ids, return_inverse=True)
        n_cells = len(unique_ids)
        n_genes = X.shape[1]
        print('Map cell_id to integer index')
        
        # Initialize empty matrix to store the result
        # Result will be dense at the end, but small: (n_cells x n_genes)
        result = np.zeros((n_cells, n_genes))
        print('Initialize empty matrix to store the result')
        
        # Efficient sparse row-by-row summing
        for i in tqdm(range(X.shape[0])):
            result[inverse_idx[i]] += X[i].toarray()[0]
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Convert to DataFrame
        grouped_expr = pd.DataFrame(result, index=unique_ids, columns=adata_cells.var_names)

        print('Aggregation from csr')
        adata_cells = AnnData(X=sp.csr_matrix(grouped_expr.values))
        adata_cells.obs_names = grouped_expr.index
        adata_cells.var_names = grouped_expr.columns # Gene Names
        print('Aggregation done')
        
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # and 'cell_id' is the grouping key used to generate grouped_expr
        # 1. Get a mapping from cell_id → average spatial position
        cell_coords = adata_filtered.obs[['cell_id']].copy()
        cell_coords['x'] = adata_filtered.obsm['spatial'][:, 0]
        cell_coords['y'] = adata_filtered.obsm['spatial'][:, 1]
        
        # 2. Average spatial position per cell
        mean_coords = cell_coords.groupby('cell_id')[['x', 'y']].mean()
        
        # 3. Match to your grouped_expr.index (cell IDs)
        mean_coords = mean_coords.loc[grouped_expr.index]
        
        # 4. Assign to obsm
        adata_cells.obsm['spatial'] = mean_coords.values
        
        # Copy the spatial dictionary to cdata
        library_id = list(adata_filtered.uns['spatial'].keys())[0]
        adata_cells.uns['spatial'] = {
            library_id: adata_filtered.uns['spatial'][library_id]
        }

        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Step 1: Copy gene metadata (var) for the grouped expression matrix
        adata_cells.var = adata_filtered.var.loc[grouped_expr.columns].copy()
        adata_cells.var_names_make_unique()
        
        # Step 2: Aggregate metadata per reconstructed cell
        meta = adata_filtered.obs.groupby('cell_id').agg({
            'n_counts': 'sum',
        })
        # - Number of bins per cell (count of occurrences)
        meta['n_bins'] = adata_filtered.obs['cell_id'].value_counts()
        meta.index.name = 'cell_id'
        
        # Step 3: Reorder metadata to match cell ordering in grouped_expr
        meta = meta.loc[grouped_expr.index]
        
        # Step 4: Assign aggregated metadata as the new obs for reconstructed cells
        adata_cells.obs = meta
        
        # Step 5: Copy over select entries from uns
        for k in ['sample_id', 'sample_metadata', 'bin2cell']:
            if k in adata.uns:
                adata_cells.uns[k] = adata.uns[k]
        
        # Step 6: Copy full spatial metadata
        adata_cells.uns['spatial'] = adata.uns['spatial']
        
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        crop_axis_fraction = 0.01  # how much to crop around the center
        
        print('Cropped vizualisation',crop_axis_fraction,' ratio' )
        # Get spatial coordinates
        spatial_coords = adata_cells.obsm['spatial'].copy()
        
        # Compute coordinate bounds and center
        x_min, x_max = np.min(spatial_coords[:, 0]), np.max(spatial_coords[:, 0])
        y_min, y_max = np.min(spatial_coords[:, 1]), np.max(spatial_coords[:, 1])
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Define crop bounds around center
        x0 = center_x - (x_max - x_min) * crop_axis_fraction / 2
        x1 = center_x + (x_max - x_min) * crop_axis_fraction / 2
        y0 = center_y - (y_max - y_min) * crop_axis_fraction / 2
        y1 = center_y + (y_max - y_min) * crop_axis_fraction / 2
        
        # Find cells within crop
        within_crop = (
            (spatial_coords[:, 0] >= x0) & (spatial_coords[:, 0] <= x1) &
            (spatial_coords[:, 1] >= y0) & (spatial_coords[:, 1] <= y1)
        )
        adata_cropped = adata_cells[within_crop, :].copy()
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Normalize n_bins to define spot sizes (you can adjust scaling)
        sizes = adata_cropped.obs['n_bins']
        sizes_normalized = 10 * (sizes / sizes.max())  # scale to 0–20

        save = f"{sample_id}_cropped_viz_size_and_counts.png"
        
        output_dir = os.path.join("figures", "segmentation", sample_id)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)  # <- temporarily switch to save location

        # Plot
        sc.pl.spatial(
            adata_cropped,
            color=['n_counts',None],
            size=sizes_normalized,
            cmap='Spectral_r',
            alpha_img=0.6,
            show=False,
            save=f"{sample_id}_cropped_viz_size_and_counts.png"
        )
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        #Visualize the UMI (total counts)
        #distribution per cell and the expression distributions of specific marker genes
        
        adata_cells.obs['total_counts'] = np.array(adata_cells.X.sum(axis=1)).flatten()
        adata_cells.obs['n_genes_by_counts'] = np.array((adata_cells.X > 0).sum(axis=1)).flatten()
        adata_cells.obs
        
        plt.figure(figsize=(14, 4))
        sns.histplot(adata_cells.obs['total_counts'], bins=555, kde=True)
        plt.xlabel("Total UMI counts per cell")
        plt.ylabel("Frequency")
        plt.title("Distribution of Total UMIs per Cell")
        plt.xlim(0, 1000)  # limit x-axis
        plt.tight_layout()
        output_path = os.path.join(sample_output_dir, f"{sample_id}_cell Distribution of UMIs.png")
        fig = plt.gcf()  # Get the current figure
        fig.savefig(output_path, dpi=300)
        # Optional: show the plot
        plt.show()

        

        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Boolean masks from adata
        is_he = adata_filtered.obs['labels_he'] != 0
        is_he_exp = adata_filtered.obs['labels_he_expanded'] != 0
        is_gex = adata_filtered.obs['labels_gex'] != 0
        
        # Hierarchical categorization
        conditions = {
            "Shared": is_he_exp & is_gex,
            "HE (Nuclei) ": is_he & ~is_gex,
            "HE expansion only": is_he_exp & ~is_gex & ~is_he,
            "GEx only": is_gex & ~is_he_exp,
            "Unassigned": ~is_he_exp & ~is_gex
        }
        
        # Count bins in each category
        counts = {k: v.sum() for k, v in conditions.items()}
        
        # Create DataFrame
        df = pd.DataFrame(list(counts.items()), columns=["Category", "Count"])
        df["Percentage"] = df["Count"] / df["Count"].sum() * 100
        
        # Ensure specific order
        df["Category"] = pd.Categorical(df["Category"], categories=[
            "Shared", "HE (Nuclei) ", "HE expansion only", "GEx only", "Unassigned"
        ], ordered=True)
        df = df.sort_values("Category")
        
        # Define custom colors
        colors = {
            "Shared": "#66c2a5",
            "HE (Nuclei) ": "#fc8d62",
            "HE expansion only": "#ffd92f",
            "GEx only": "#8da0cb",
            "Unassigned": "#dddddd"
        }
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 2))
        left = 0
        for _, row in df.iterrows():
            ax.barh(y=0, width=row["Count"], left=left,
                    color=colors[row["Category"]], label=row["Category"])
            ax.text(left + row["Count"]/2, 0,
                    f"{int(row['Count'])}\n({row['Percentage']:.1f}%)",
                    ha="center", va="center", fontsize=9, color="black")
            left += row["Count"]
        
        ax.set_yticks([])
        ax.set_xlabel("Number of Bins")
        ax.set_title("Hierarchical Segmentation Overlap")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()

        output_path = os.path.join(sample_output_dir, f"{sample_id}_hierarchical_overlap post filtering.png")
        fig = plt.gcf()  # Get the current figure
        fig.savefig(output_path, dpi=300)
        # Optional: show the plot
        plt.show()
        
        sizes = adata_cells.obs['n_bins']
        sizes_normalized = 10 * (sizes / sizes.max())  # scale to 0–20
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Plot
        sc.pl.spatial(
            adata_cells,
            color=['n_counts',None],
            size=sizes_normalized,
            cmap='Spectral_r',
            alpha_img=0.6,
            show=False,
            save=f"{sample_id}_ full image projection.png"
        )
        
        file_path = os.path.join(sample_output_dir, f"adata_{sample_id}_cells.h5ad")
        adata_cells.write(file_path)
        print(" =============== ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


