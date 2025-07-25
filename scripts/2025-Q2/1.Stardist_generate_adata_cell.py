import resource

# Set maximum memory usage to  x GB (in bytes)
max_memory_bytes = 600 * 1024**3 
resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
import geopandas as gpd
import scanpy as sc
from tifffile import imread, imwrite
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from shapely.geometry import Polygon, Point
from scipy import sparse
from matplotlib.colors import ListedColormap

# Configuration for inline plotting



import os
from glob import glob

# Define root paths
tiff_root = r"/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/IMAGE/HE_nanozoomer_tif"
sampleids = [['IMMUNEX004']]

for sample_ids in sampleids:
    
        
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import anndata
    import geopandas as gpd
    import scanpy as sc
    from tifffile import imread, imwrite
    from csbdeep.utils import normalize
    from stardist.models import StarDist2D
    from shapely.geometry import Polygon, Point
    from scipy import sparse
    from matplotlib.colors import ListedColormap

    # Automatically find the TIFF file for one of the known samples
    tiff_path = None
    for sample_id in sample_ids:
        pattern = os.path.join(tiff_root, f"{sample_id}*.tif")
        matches = glob(pattern)
        if matches:
            tiff_path = matches[0]
            print(f"Using TIFF file: {tiff_path}")
            break
        else:
            print('No matching TIFF FOUND !')
    
    import gc
    import shutil
    
    # Prepare output directory for the sample
    sample_output_dir = "/scratch/Projects/IMMUNEX/results/intermediate/stardist_segmentation/" + sample_id
    os.makedirs(sample_output_dir, exist_ok=True)
    from pathlib import Path
    sample_output_dir = Path(sample_output_dir)
    
    # Update matplotlib saving to use sample-specific output folder
    def save_fig(name):
        path = os.path.join(sample_output_dir, name)
        plt.savefig(path)
        print(f"Saved figure: {path}")
        plt.close()
    
    
    
    # Load full image and normalize
    img = imread(tiff_path)
    print('image loaded')
    
    # Load pretrained model
    model = StarDist2D.from_pretrained('2D_versatile_he')
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    save_fig(Path(sample_output_dir) / '_preview.png')
    
    
    img = normalize(img, 5, 95)
    print('Image Normalized ')
    
    # Load adata
    
    adata = sc.read_10x_h5(f"/home/mounim/rawdata/IMMUNEX/OUTPUT/Visium_NSCLC_{sample_id}/outs/binned_outputs/square_002um/filtered_feature_bc_matrix.h5")
    adata.raw = adata
    adata.uns['sample_id'] = sample_id
    print(adata)
    
    
    # Load coordinates from Space Ranger output (adjust path as needed)
    parquet_path = f"/home/mounim/rawdata/IMMUNEX/OUTPUT/Visium_NSCLC_{sample_id}/outs/binned_outputs/square_002um/spatial/tissue_positions.parquet"
    coords = pd.read_parquet(parquet_path)
    print(coords.head())
    
    # Set barcode index if not done
    coords.set_index("barcode", inplace=True)
    
    # Join without suffix — no conflict now
    adata.obs = adata.obs.join(coords, how="left")
    print(adata.obs)
    
    # Keep only bins with spatial coordinates
    adata = adata[adata.obs["pxl_row_in_fullres"].notnull()].copy()
    
    # Add to obsm
    adata.obsm["spatial"] = adata.obs[["pxl_col_in_fullres","pxl_row_in_fullres"]].values
    
    adata.obs["library_id"] = sample_id
    adata.uns["spatial"] = {
        sample_id: {
            "images": {"hires": None},
            "scalefactors": {
                "tissue_hires_scalef": 1.0,
                "spot_diameter_fullres": 1.0
            }
        }
    }
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    print("🎨 Plotting spatial scatter...")
    
    
    
    import seaborn as sns
    
    # Prepare DataFrame for plotting
    df_plot = pd.DataFrame(
        adata.obsm["spatial"], 
        columns=["x", "y"],
        index=adata.obs.index
    )
    
    
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df_plot, x="x", y="y", s=2, color="black")
    plt.gca().invert_yaxis()  # Optional: match image orientation
    plt.title(f"{sample_id} - Spatial spots (2µm bins)")
    plt.xlabel("pxl_col_in_fullres")
    plt.ylabel("pxl_row_in_fullres")
    plt.tight_layout()
    
    save_fig(Path(sample_output_dir) / 'scatterplot2.png')
    
    
    import numpy as np
    
    # Get x and y coordinates
    coords = adata.obsm["spatial"]
    x_min, y_min = coords.min(axis=0).astype(int)
    x_max, y_max = coords.max(axis=0).astype(int)
    
    # Add margin if needed
    margin = 100  # adjust as needed
    x_min = max(x_min - margin, 0)
    y_min = max(y_min - margin, 0)
    x_max += margin
    y_max += margin
    
    img_crop = img[y_min:y_max, x_min:x_max, :]
    
    # Plot cropped image
    plt.figure(figsize=(10, 10))
    plt.imshow(img_crop)
    plt.title("Cropped Visium Region")
    plt.axis("off")
    
    save_fig(Path(sample_output_dir) / 'cropped.png')
    
    img_height, img_width = img_crop.shape[:2]
    
    print(f'Image h = {img_height} and w = {img_width}')
    
    sample_id = tiff_path.split('/')[-1].split('_')[0]
    print(sample_id)
    
    # !rm -rf /scratch/Projects/IMMUNEX/notebooks/6.Full\ Pipeline/IMMUNEX004/roi_*
    # !rm /scratch/Projects/IMMUNEX/notebooks/6.Full\ Pipeline/IMMUNEX0*/roi*
    img = img_crop
    
    from matplotlib.colors import ListedColormap
    import os
    import cv2
    from tqdm import tqdm
    from shapely.affinity import translate
    from scipy.ndimage import gaussian_filter
    import matplotlib.pyplot as plt
    
    img = img_crop
    
    cmap = ListedColormap(['grey'])  # or any other single-color mask
    
    # print("Step 1: Displaying original image...")
    # plt.figure(figsize=(8, 8))
    # plt.imshow(img)
    # plt.tight_layout()
    # save_fig(f"image_preview.png", dpi=150)
    # # save_fig(os.path.join("/scratch/Projects/IMMUNEX/results/intermediate/stardist_segmentation", "figure_{len(os.listdir('/scratch/Projects/IMMUNEX/results/intermediate/stardist_segmentation'))}.png"))
    # plt.close()
    
    # --- Tissue detection preprocessing ---
    print("Step 2: Converting image to grayscale...")
    lower_thresh = 222
    roi_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    print("Step 3: Thresholding grayscale image to get tissue mask...")
    _, tissue_mask = cv2.threshold(roi_gray, lower_thresh, 255, cv2.THRESH_BINARY_INV)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(tissue_mask, cmap='gray')
    # save_fig(f"tissue_mask.png", dpi=150)
    # save_fig(os.path.join("/scratch/Projects/IMMUNEX/results/intermediate/stardist_segmentation", "figure_{len(os.listdir('/scratch/Projects/IMMUNEX/results/intermediate/stardist_segmentation'))}.png"))
    
    print("Step 4: Applying Gaussian blur to mask...")
    tissue_mask_blur = cv2.GaussianBlur(tissue_mask, (5, 5), sigmaX=2)
    
    print("Step 5: Binarizing blurred mask...")
    _, tissue_mask_bin = cv2.threshold(tissue_mask_blur, 200, 255, cv2.THRESH_BINARY)
    
    print("Step 6: Inverting binary mask...")
    tissue_mask_inv = cv2.bitwise_not(tissue_mask_bin)
    
    print("Step 7: Performing morphological opening...")
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tissue_mask_inv_clean = cv2.morphologyEx(tissue_mask_inv, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    print("Step 8: Performing morphological closing...")
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    tissue_mask_clean = cv2.morphologyEx(tissue_mask_inv_clean, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Optional: Visualize cleaned mask
    # print("Step 9: Displaying cleaned tissue mask...")
    # plt.figure(figsize=(6, 6))
    # plt.imshow(tissue_mask_clean, cmap='gray')
    # save_fig(f"tissue_mask_clean.png", dpi=150)
    # save_fig(os.path.join("/scratch/Projects/IMMUNEX/results/intermediate/stardist_segmentation", "figure_{len(os.listdir('/scratch/Projects/IMMUNEX/results/intermediate/stardist_segmentation'))}.png"))
    
    print("Step 9: Overlaying tissue mask on original ROI image...")
    # plt.figure(figsize=(8, 8))
    # plt.imshow(roi_img)
    # plt.imshow(tissue_mask_clean, cmap='Purples', alpha=0.7)
    # plt.title(f"ROI Tissue Mask Overlay")
    # plt.axis('off')
    # plt.tight_layout()
    # save_fig(f"tissue_mask_overlay.png", dpi=150)
    # save_fig(os.path.join("/scratch/Projects/IMMUNEX/results/intermediate/stardist_segmentation", "figure_{len(os.listdir('/scratch/Projects/IMMUNEX/results/intermediate/stardist_segmentation'))}.png"))
    
    print("✅ Tissue mask generation complete.")
    
    import gc
    gc.collect()
    
    # Segmentation parameters
    block_size = 1024 * 3
    prob_thresh = 0.0001
    nms_thresh = 0.001
    min_overlap = 128
    context = 128
    
    # # --- Run segmentation ---
    labels, polys = model.predict_instances_big(
        img_crop, axes='YXC', 
        block_size=block_size,
        prob_thresh=prob_thresh, 
        nms_thresh=nms_thresh,
        min_overlap=min_overlap, 
        context=context,
        normalizer=None, 
        n_tiles=(2,2,1)
    )
    
    # --- Convert local polygon coordinates to global ---
    # --- Filter out nuclei in white area ---
    
    
    from shapely.geometry import Polygon
    from shapely.affinity import translate
    import geopandas as gpd
    import os
    
    # --- Step 1: Reconstruct local polygons ---
    geometries = []
    
    for nuclei in tqdm( range(len(polys['coord'])) ):
        ys = polys['coord'][nuclei][0]
        xs = polys['coord'][nuclei][1]
        coords = list(zip(xs, ys))  # (x, y) format for polygons
        geometries.append(Polygon(coords))
    
    # --- Step 2: Convert to GeoDataFrame ---
    gdf = gpd.GeoDataFrame(geometry=geometries)
    gdf['id'] = [f"ID_{i+1}" for i in range(len(gdf))]
    
    # --- Step 3: Translate local polygons to global image coordinates ---
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: translate(geom, xoff=x_min, yoff=y_min))
    
    # --- Step 4: Save the global-coordinate segmentation ---
    outdir = 'segmentation/'
    os.makedirs(outdir, exist_ok=True)
    gdf.to_pickle(f"{outdir}{sample_id}_all_segmentation_{prob_thresh}_{nms_thresh}_{min_overlap}.pkl")
    
    outdir = 'segmentation/'
    gdf = pd.read_pickle(f"{outdir}{sample_id}_all_segmentation_{prob_thresh}_{nms_thresh}_{min_overlap}.pkl")
    gdf.head()
    
    print("Segmentation polygon X range:", gdf.total_bounds[[0, 2]])
    print("Segmentation polygon Y range:", gdf.total_bounds[[1, 3]])
    print("Visium coords X range:", adata.obsm["spatial"][:, 0].min(), adata.obsm["spatial"][:, 0].max())
    print("Visium coords Y range:", adata.obsm["spatial"][:, 1].min(), adata.obsm["spatial"][:, 1].max())
    
    from shapely.geometry import Polygon
    from shapely.affinity import translate
    from tqdm import tqdm
    import geopandas as gpd
    import matplotlib.pyplot as plt
    
    selected_geometries = []
    rejected_geometries = []
    
    # --- Loop over all polygons from local segmentation ---
    for nuclei in tqdm(range(len(polys['coord']))):
        ys = polys['coord'][nuclei][0]
        xs = polys['coord'][nuclei][1]
    
        # Local centroid (for tissue mask)
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
    
        # Original local coordinates
        coords_local = [(x, y) for x, y in zip(xs, ys)]
    
        # Shift to global coordinates for output
        coords_global = [(x + x_min, y + y_min) for x, y in coords_local]
    
        # Check tissue mask (still in local coords!)
        if 0 <= cy < tissue_mask_clean.shape[0] and 0 <= cx < tissue_mask_clean.shape[1]:
            if tissue_mask_clean[cy, cx] < 50:
                selected_geometries.append(Polygon(coords_global))  # save global
            else:
                rejected_geometries.append(Polygon(coords_global))
        else:
            rejected_geometries.append(Polygon(coords_global))
    
    print(f"Cropped image, prob_thresh={prob_thresh} → selected: {len(selected_geometries)} | rejected: {len(rejected_geometries)}")
    
    # --- GeoDataFrames in global coordinates ---
    gdf_selected = gpd.GeoDataFrame(geometry=selected_geometries)
    gdf_rejected = gpd.GeoDataFrame(geometry=rejected_geometries)
    gdf_selected['id'] = [f"ID_sel_{j+1}" for j in range(len(gdf_selected))]
    gdf_rejected['id'] = [f"ID_rej_{j+1}" for j in range(len(gdf_rejected))]
    
    outdir = 'segmentation/'
    os.makedirs(outdir, exist_ok=True)
    
    # --- Save segmentation result ---
    # gdf_selected.to_pickle(f"{outdir}{sample_id}_segmentation_{prob_thresh}_{nms_thresh}_{min_overlap}.pkl")
    
    # # --- Visualization ---
    # fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    # # Subplot 1: Original ROI image
    # axes[0].imshow(img)
    # axes[0].set_title("Original Image")
    # # Subplot 3: Overlay on H&E
    # axes[1].imshow(img)
    # gdf_selected.plot(cmap=cmap, ax=axes[1], alpha=0.1)
    # gdf_selected.boundary.plot(ax=axes[1], edgecolor='yellow', linewidth=.005)
    # axes[1].set_title("H&E + Segmentation")
    # # Save and show
    # plt.tight_layout()
    # save_fig(f"{outdir}segmentation_plot_{prob_thresh}.png", dpi=200)
    # save_fig(os.path.join("/scratch/Projects/IMMUNEX/results/intermediate/stardist_segmentation", "figure_{len(os.listdir('/scratch/Projects/IMMUNEX/results/intermediate/stardist_segmentation'))}.png"))
    
    
    # --- Standalone plot: H&E + Segmentation overlay only ---
    
    
    
    from shapely.affinity import translate
    
    gdf_local = gdf_selected.copy()
    gdf_local['geometry'] = gdf_local['geometry'].apply(
        lambda geom: translate(geom, xoff=-x_min, yoff=-y_min)
    )
    
    # Start from gdf_selected (global coords)
    gdf_selected_plot = gdf_selected.copy()
    gdf_selected_plot['geometry'] = gdf_selected_plot['geometry'].apply(
        lambda geom: translate(geom, xoff=-x_min, yoff=-y_min)
    )
    
    gdf_rejected_plot = gdf_rejected.copy()
    gdf_rejected_plot['geometry'] = gdf_rejected_plot['geometry'].apply(
        lambda geom: translate(geom, xoff=-x_min, yoff=-y_min)
    )
    
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, alpha=.2)
    gdf_selected_plot.plot(cmap=cmap, ax=ax, alpha=0.1, edgecolor='cyan')
    gdf_rejected_plot.boundary.plot(ax=ax, edgecolor='yellow', alpha=0.1,linewidth=1)
    
    plt.tight_layout()
    # save_fig(f"{outdir}rejected_segmentations_{prob_thresh}.png", dpi=111)
    
    save_fig(Path(sample_output_dir) / f'rejected_segmentations_{prob_thresh}.png')
    
    # save_fig(os.path.join("/scratch/Projects/IMMUNEX/results/intermediate/stardist_segmentation", "figure_{len(os.listdir('/scratch/Projects/IMMUNEX/results/intermediate/stardist_segmentation'))}.png"))
    plt.close()
    
    
    
    
    
    
    
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point
    
    print("Step 1: Create GeoDataFrame for Visium spots in pixel space...")
    gdf_spots = gpd.GeoDataFrame(
        adata.obs.copy(),
        geometry=gpd.points_from_xy(
            adata.obsm["spatial"][:, 0],  # x
            adata.obsm["spatial"][:, 1]   # y
        )
    )
    gdf_spots.crs = None
    gdf.crs = None  # make sure both are CRS-less or equal
    
    print(f"  → {len(gdf_spots)} spots ready for spatial join.")
    
    
    # --- Spatial Join: Assign each spot to a segmentation polygon ---
    print("Step 2: Matching spots with nuclei (gpd.sjoin)...")
    joined = gpd.sjoin(gdf_spots, gdf[['geometry', 'id']], how='left', predicate='within')
    
    # Flags and assignments
    joined['isNuclei'] = ~joined['index_right'].isna()
    joined['nucleus_id'] = joined['id']  # optional: keep polygon ID
    
    n_matched = joined['isNuclei'].sum()
    print(f"  → Matched {n_matched} spots to nuclei.")
    
    
    # --- Tissue Mask Check: isEmpty ---
    print("Step 3: Checking tissue mask for empty spots...")
    spot_y_img = (adata.obsm["spatial"][:, 1] - y_min).astype(int)
    spot_x_img = (adata.obsm["spatial"][:, 0] - x_min).astype(int)
    
    # Check bounds
    in_bounds = (
        (spot_y_img >= 0) & (spot_y_img < tissue_mask_clean.shape[0]) &
        (spot_x_img >= 0) & (spot_x_img < tissue_mask_clean.shape[1])
    )
    
    isEmpty = np.ones(len(adata), dtype=bool)
    isEmpty[in_bounds] = tissue_mask_clean[spot_y_img[in_bounds], spot_x_img[in_bounds]] >= 50
    
    print(f"  → Found {isEmpty.sum()} empty spots (outside tissue).")
    
    joined
    
    print("Step 4: Resolving duplicates and updating adata.obs...")
    
    # Drop duplicate spot matches — keep first match only
    joined_unique = joined[~joined.index.duplicated(keep='first')].copy()
    
    # Ensure indices match types
    joined_unique.index = joined_unique.index.astype(str)
    adata.obs.index = adata.obs.index.astype(str)
    
    # Merge safely
    adata.obs['isNuclei'] = joined_unique['isNuclei'].reindex(adata.obs.index).fillna(False)
    adata.obs['nucleus_id'] = joined_unique['nucleus_id'].reindex(adata.obs.index)
    adata.obs['isEmpty'] = isEmpty
    
    print("✅ Done. Columns added to adata.obs: ['isNuclei', 'nucleus_id', 'isEmpty']")
    
    print("Step 5: Expanding nucleus assignment by ±2 Visium bins...")
    
    # Start from spots that are already assigned to a nucleus
    nucleus_map = adata.obs[['nucleus_id', 'array_row', 'array_col']].dropna()
    
    # Prepare map of nucleus → set of (row, col) grid locations to expand from
    expanded_assignments = {}
    
    for nucleus_id, group in tqdm(nucleus_map.groupby('nucleus_id')):
        rows = group['array_row']
        cols = group['array_col']
    
        # Expand each (row, col) by ±2 in both directions
        for r, c in zip(rows, cols):
            for dr in range(-2, 3):       # -2 to +2
                for dc in range(-2, 3):
                    key = (r + dr, c + dc)
                    # Only assign if not already taken
                    if key not in expanded_assignments:
                        expanded_assignments[key] = nucleus_id
    
    # Map each spot in adata to a nucleus if it's in the expanded map
    expanded_ids = []
    
    for r, c in zip(adata.obs['array_row'], adata.obs['array_col']):
        expanded_ids.append(expanded_assignments.get((r, c), np.nan))
    
    # Add to adata.obs
    adata.obs['nucleus_id_expanded'] = expanded_ids
    
    adata.obs['isNuclei_expanded'] = adata.obs['nucleus_id_expanded'].notna()
    
    print(f"  → Expanded assignments: {adata.obs['isNuclei_expanded'].sum()} spots assigned via ±2 bin expansion.")
    print(adata.obs.head())
    
    r_center = 100
    c_center = 100
    
    # Define crop size (±20 bins)
    window = 10
    
    # Crop by row and column range
    bdata = adata[
        (adata.obs['array_row'] >= r_center - window) & (adata.obs['array_row'] <= r_center + window) &
        (adata.obs['array_col'] >= c_center - window) & (adata.obs['array_col'] <= c_center + window),
        :
    ].copy()
    
    
    import matplotlib.pyplot as plt
    
    print(f"Plotting Visium crop around row={r_center}, col={c_center}...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    x = bdata.obsm['spatial'][:, 0]
    y = bdata.obsm['spatial'][:, 1]
    
    # Color by nucleus_id_expanded
    codes = bdata.obs['nucleus_id'].astype('category').cat.codes
    
    sc = ax.scatter(x, y, c=codes, cmap='rainbow', s=55, alpha=0.9)
    
    plt.colorbar(sc, ax=ax, shrink=0.7, label='Nucleus ID (coded)')
    ax.set_title("Visium Crop (±2 bins) Colored by Nucleus")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # ax.invert_yaxis()
    plt.axis('equal')
    plt.tight_layout()
    save_fig(Path(sample_output_dir) / f'Visium Crop nuc.png')
    
    # save_fig(os.path.join("/scratch/Projects/IMMUNEX/results/intermediate/stardist_segmentation", "figure_{len(os.listdir('/scratch/Projects/IMMUNEX/results/intermediate/stardist_segmentation'))}.png"))
    
    # exporting the adata segmented
    import os
    
    outdir = "segmentation/"
    os.makedirs(outdir, exist_ok=True)
    adata.write_h5ad(f"{outdir}segmented_adata.h5ad")
    
    filtered_adata = adata[adata.obs['nucleus_id'].isna()].copy()
    filtered_adata
    
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from tqdm import tqdm
    import anndata as ad
    
    print("Step 1: Preparing input...")
    
    # Filter barcodes assigned to nuclei
    adata_in_cells = filtered_adata[filtered_adata.obs['nucleus_id_expanded'].notna()].copy()
    n_barcode = adata_in_cells.n_obs
    print(f"  → {n_barcode:,} barcodes assigned to a nucleus")
    
    # Extract nucleus IDs
    nucleus_ids = adata_in_cells.obs['nucleus_id_expanded'].astype("category")
    nucleus_codes = nucleus_ids.cat.codes
    n_unique = len(nucleus_ids.cat.categories)
    print(f"  → {n_unique:,} unique expanded nuclei")
    
    # Convert to sparse matrix
    X = adata_in_cells.X.tocsr()
    
    # Prepare empty result matrix and centroid matrix
    print("Step 2: Summing expression and computing centroids...")
    summed_counts = sparse.lil_matrix((n_unique, X.shape[1]), dtype=X.dtype)
    centroids = np.zeros((n_unique, 2))  # X, Y
    
    # Use the high-res spatial coordinates (can also be "spatial" or "spatial_pixel")
    spatial_coords = adata_in_cells.obsm["spatial"]
    
    # Loop over each nucleus
    for nucleus_idx in tqdm(range(n_unique), desc="  → Processing nuclei"):
        idx = np.where(nucleus_codes == nucleus_idx)[0]
        if len(idx) > 0:
            summed_counts[nucleus_idx] = X[idx].sum(axis=0)
            centroids[nucleus_idx] = spatial_coords[idx].mean(axis=0)
    
    print("Step 3: Building DataFrame from results...")
    # Wrap counts into DataFrame
    counts_grouped = pd.DataFrame.sparse.from_spmatrix(
        summed_counts.tocsr(),
        index=nucleus_ids.cat.categories,
        columns=adata_in_cells.var_names
    )
    print(f"✅ Done: counts_grouped has shape {counts_grouped.shape}")
    
    print("Step 4: Creating AnnData for nuclei...")
    # Create new AnnData
    adata_cells = ad.AnnData(
        X=sparse.csr_matrix(counts_grouped.values),
        obs=pd.DataFrame(index=counts_grouped.index),
        var=adata_in_cells.var.copy()
    )
    
    # Add metadata
    adata_cells.obs["nucleus_id"] = adata_cells.obs.index
    adata_cells.obsm["spatial"] = centroids  # ← Store centroid coordinates
    
    print("✅ Created `adata_cells` with .obsm['spatial'] included.")
    
    adata_cells.obsm['spatial']
    
    del img
    del img_crop
    
    
    
    import os
    
    outdir = str(sample_output_dir)
    os.makedirs(outdir, exist_ok=True)
    
    # Export filtered_adata if not already saved
    filtered_adata.write_h5ad(f"{outdir}/filtered_adata.h5ad", compression="gzip")
    
    # Export adata_cells (summed per nucleus)
    adata_cells.write_h5ad(f"{outdir}/adata_cells_summed_by_nucleus.h5ad", compression="gzip")
    
    # Export gdf (original segmentation polygons)
    gdf.to_file(f"{outdir}/segmentation_polygons.geojson", driver="GeoJSON")
    
    # Export gdf_selected (if available — tissue-filtered nuclei)
    if 'gdf_selected' in globals():
        gdf_selected.to_file(f"{outdir}/segmentation_selected.geojson", driver="GeoJSON")
    
    print("✅ All files exported:")
    print(f"  - Filtered AnnData: {outdir}/filtered_adata.h5ad")
    print(f"  - Nucleus-level counts: {outdir}/adata_cells_summed_by_nucleus.h5ad")
    print(f"  - Full segmentation polygons: {outdir}/segmentation_polygons.geojson")
    print(f"  - Selected segmentation (if any): {outdir}/segmentation_selected.geojson")
    
