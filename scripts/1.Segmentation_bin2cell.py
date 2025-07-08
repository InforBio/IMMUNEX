import os
import re
import time
import logging
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import bin2cell as b2c
import tifffile
from tqdm import tqdm
import importlib

# -------------------------------
# USER-DEFINED PARAMETERS
# -------------------------------

# Input paths
base_sample_path = Path("/home/mounim/rawdata/IMMUNEX/tools")
# base_sample_path = Path("/home/mounim/rawdata/IMMUNEX/OUTPUT")
base_he_image_path = Path("/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/IMAGE/HE_nanozoomer_tif")

# Output path
base_output_dir = Path("segmentation/bin2cell/bin2cell_output_he0005_gex005")
os.makedirs(base_output_dir, exist_ok=True)

# Microns per pixel
mpp = 0.6

# Define a rectangle for the region of interest
mask_region = {
    "x": 500,   # col
    "y": 750,  # row
    "w": 220, ## width
    "h": 220 ## height
}


# StarDist settings for H&E
stardist_he_params = {
    "stardist_model": "2D_versatile_he",
    "block_size": 4096,
    "min_overlap": 128,
    "context": 128,
    "prob_thresh": 0.0005
}

# StarDist settings for GEX
stardist_gex_params = {
    "stardist_model": "2D_versatile_fluo",
    "prob_thresh": 0.05,
    "nms_thresh": 0.1,
    "block_size": 4096,
    "min_overlap": 250,
    "context": 128,
    "show_progress": True
}

# Gene filtering thresholds
gene_filter_min_cells = 3
cell_filter_min_counts = 1

# Image smoothing for grid_image
grid_image_sigma = 5

# Logging configuration
import os, sys
log_path = base_output_dir / "batch_processing.log"
logging.basicConfig(filename=log_path, level = logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------
# PATCH TIFF LOADER FOR LARGE FILES
# -------------------------------

def patched_load_image(image_path, **kwargs):
    
    print(f"Loading image via tifffile: {image_path}")
    img = tifffile.imread(image_path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[0] == 3 and img.ndim == 3:
        img = np.moveaxis(img, 0, -1)
    return img




def print_log(message, type ="info"):
    print(message)
    if type == "error":
        logging.error(message)
    elif type == "warning":
        logging.warning(message)
    elif type == "success":
        logging.success(message)
    else:
        logging.info(message)

# -------------------------------
# START BATCH PROCESSING
# -------------------------------

# Match available HE images to IMMUNEX IDs
nanozoomer_tif = {}
for file in tqdm(base_he_image_path.glob("*.tif")) :
    match = re.match(r"(IMMUNEX\d+)(.*)\.tif", file.name)
    if match:
        sample_id, suffix = match.groups()
        nanozoomer_tif[sample_id] = suffix

sample_folders = sorted(base_sample_path.glob("Visium_NSCLC_*"))

print(sample_folders)

sample_to_skip = [
    'IMMUNEX001',
    'IMMUNEX002',
    'IMMUNEX003',
    'IMMUNEX004',
    'IMMUNEX005',
    'IMMUNEX006',
    'IMMUNEX007',
    'IMMUNEX008',
    'IMMUNEX009',
    'IMMUNEX010',
    'IMMUNEX011',
    'IMMUNEX012',
    'IMMUNEX013',
    'IMMUNEX014',
    'IMMUNEX015',
    # 'IMMUNEX017',
    # 'IMMUNEX018',
]

print(sample_folders)

for sample_folder in tqdm(sample_folders) :
    b2c.bin2cell.load_image = patched_load_image

    IMMUNEXID = sample_folder.name.split("_")[-1]
    
    print_log(f"Starting processing for {IMMUNEXID}")
    start_time_sample = time.time()

    if IMMUNEXID not in nanozoomer_tif or IMMUNEXID in sample_to_skip:
        print_log(f"Skipping {IMMUNEXID}: No matching HE image found", type="warning")
        continue

    if 1:
        step_time = time.time()
        path = sample_folder / "outs/binned_outputs/square_002um/"
        source_image_path = base_he_image_path / f"{IMMUNEXID}{nanozoomer_tif[IMMUNEXID]}.tif"
        output_dir = base_output_dir / IMMUNEXID
        output_dir.mkdir(parents=True, exist_ok=True)

        stardist_dir = output_dir / "stardist"
        stardist_dir.mkdir(exist_ok=True)
        he_img_out = stardist_dir / "he.tiff"
        he_seg_out = stardist_dir / "he.npz"
        gex_img_out = stardist_dir / "gex.tiff"
        gex_seg_out = stardist_dir / "gex.npz"

        
        try:
            importlib.reload(b2c.bin2cell) 
            adata = b2c.read_visium(path, source_image_path=source_image_path)
        except:
            b2c.bin2cell.load_image = patched_load_image
            adata = b2c.read_visium(path, source_image_path=source_image_path)
        
        adata.var_names_make_unique()
        sc.pp.filter_genes(adata, min_cells=gene_filter_min_cells)
        sc.pp.filter_cells(adata, min_counts=cell_filter_min_counts)
        
        print_log(f"Loaded data for {IMMUNEXID} in {time.time() - step_time:.2f}s")
        step_time = time.time()

        b2c.destripe(adata)
        print('destriped')

        
        try:
            importlib.reload(b2c.bin2cell) 
            b2c.scaled_he_image(adata, mpp=mpp, save_path=he_img_out)
        except:
            b2c.bin2cell.load_image = patched_load_image
            b2c.scaled_he_image(adata, mpp=mpp, save_path=he_img_out)
            

        
        print_log(f"Destriping and scaled image completed in {time.time() - step_time:.2f}s")
        step_time = time.time()

        mask = (
            (adata.obs['array_row'] >= mask_region["y"]) &
            (adata.obs['array_row'] <  mask_region["y"] + mask_region["h"]) &
            (adata.obs['array_col'] >= mask_region["x"]) &
            (adata.obs['array_col'] <  mask_region["x"] + mask_region["w"])
        )

        bdata = adata[mask]

        sc.set_figure_params(figsize=[10,10], dpi=100)
        sc.pl.spatial(bdata, color=[None, "n_counts", "n_counts_adjusted"], img_key=f"{mpp}_mpp_150_buffer", basis="spatial_cropped_150_buffer", cmap='Reds')
        plt.savefig(output_dir / "spatial_destriping.pdf")
        plt.close()

        
        print_log(f"Spatial plot saved in {time.time() - step_time:.2f}s")
        step_time = time.time()

        b2c.stardist(str(he_img_out), str(he_seg_out), **stardist_he_params)
        b2c.insert_labels(adata, str(he_seg_out), basis="spatial", spatial_key="spatial_cropped_150_buffer", mpp=mpp, labels_key="labels_he")

        
        print_log(f"H&E segmentation completed in {time.time() - step_time:.2f}s")
        step_time = time.time()

        b2c.expand_labels(adata, labels_key="labels_he", expanded_labels_key="labels_he_expanded")

        print_log(f"Label expansion completed in {time.time() - step_time:.2f}s")
        step_time = time.time()

        crop = b2c.get_crop(adata[mask], basis="spatial", spatial_key="spatial_cropped_150_buffer", mpp=mpp)

        # Generate the visualization
        rendered = b2c.view_stardist_labels(
            image_path=he_img_out, 
            labels_npz_path=he_seg_out, 
            crop=crop,
            alpha_boundary=1, 
            normalize_img=True,
            alpha=0.1
        )

        # Show the image
        plt.imshow(rendered)
        plt.axis("off")
        plt.tight_layout()
        # Save to file
        plt.savefig(output_dir / "he_segmentation_overlay.png", bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()



         # Generate GEX grid image and export
        import importlib
        importlib.reload(b2c.bin2cell) # using default
        img = b2c.grid_image(adata, "n_counts_adjusted", mpp=mpp, sigma=grid_image_sigma) 
        cv2.imwrite(str(gex_img_out), img)

        print(" 🚀 GEx generated and exported to ", gex_img_out)
        
        b2c.stardist(str(gex_img_out), str(gex_seg_out), **stardist_gex_params)

        print(f" GEx segmented 🔎 ")
        b2c.insert_labels(adata, str(gex_seg_out), basis="array", mpp=mpp, labels_key="labels_gex")
        print_log(f"GEX segmentation completed in {time.time() - step_time:.2f}s")
        step_time = time.time()

        b2c.salvage_secondary_labels(adata, primary_label="labels_he_expanded", secondary_label="labels_gex", labels_key="labels_joint")

        
        print_log(f"Label fusion completed in {time.time() - step_time:.2f}s")
        step_time = time.time()

        cdata = b2c.bin_to_cell(adata, labels_key="labels_joint", spatial_keys=["spatial", "spatial_cropped_150_buffer"])
        sc.pl.spatial(cdata, color=["bin_count"], basis="spatial_cropped_150_buffer", img_key=f"{mpp}_mpp_150_buffer", show=False)
        plt.savefig(output_dir / "spatial_cell_density.pdf")
        plt.close()

        print_log(f"spatial_cell_density exported")

        try:
            fig = sc.pl.spatial(
                bdata,
                color=[None, "labels_joint_source", "labels_joint"],
                img_key=f"{mpp}_mpp_150_buffer",
                basis="spatial_cropped_150_buffer",
                show=False, return_fig=True
            )

            fig.savefig(output_dir / "salvage_labels.svg", bbox_inches="tight")
            plt.close()
        except:
            pass


        # Export final processed files
        adata.write(output_dir / "adata_processed.h5ad", compression="gzip")
        cdata.write(output_dir / "cdata_cell_segmented.h5ad", compression="gzip")
        
        print_log(f"Exported adata and cdata to {output_dir}")

        
        print_log(f"Final cell density plot completed in {time.time() - step_time:.2f}s")
        elapsed = time.time() - start_time_sample
        
        print_log(f"SUCCESS: {IMMUNEXID} processed in {elapsed:.2f}s")

    else:
        pass
    # except Exception as e:
    #     print_log(f"XXXXXXXXXXXXXXXX\nXXXXXXXXXXXXXXXX FAILURE: {IMMUNEXID} failed with error: {str(e)} XXXXXXXXXXXXXXXX\nXXXXXXXXXXXXXXXX ", type='error')
