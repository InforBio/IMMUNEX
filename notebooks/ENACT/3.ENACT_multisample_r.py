from enact.pipeline import ENACT
import os
import gc

images = os.listdir('/scratch/IMMUNEX/PJ2410310_250214/IMAGE/HE_nanozoomer_tif')
skipsamples = ['IMMUNEX017', 'IMMUNEX001']
export_dir = '/scratch/Projects/IMMUNEX/enact_cache'

for image in images[::-1]:
    try:
        sample = image[:10]
        out_name = f"run01_{sample}"
    
        sample_export_path = os.path.join(export_dir, out_name)
        if os.path.exists(sample_export_path):
            print(f"Skipping {sample}: output already exists.")
            continue
    
        print(f"Processing {sample}")
        so_hd = ENACT(
            cache_dir=export_dir,
            wsi_path=f"/scratch/IMMUNEX/PJ2410310_250214/IMAGE/HE_nanozoomer_tif/{image}",
            visiumhd_h5_path=f"/scratch/IMMUNEX/OUTPUT/Visium_NSCLC_{sample}/outs/binned_outputs/square_002um/filtered_feature_bc_matrix.h5",
                tissue_positions_path=f"/scratch/IMMUNEX/OUTPUT/Visium_NSCLC_{sample}/outs/binned_outputs/square_002um/spatial/tissue_positions.parquet",
            cell_typist_model="Human_Colorectal_Cancer",
            analysis_name=out_name,
            patch_size = 8000
        )
    
        so_hd.run_enact()
        del so_hd
        gc.collect()
    except:
        pass