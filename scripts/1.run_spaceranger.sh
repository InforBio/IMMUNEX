sudo chmod -R u+rwX,go+rX "$INSTALL"
sudo chmod 755 "$(dirname "$INSTALL")"
OUT_BASE=/mnt/data/IMMUNEX/OUTPUT
ID=Visium_NSCLC_IMMUNEX012_HD_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT_BASE"


"/opt/spaceranger-4.0.1/bin/spaceranger" count \
  --id=Visium_NSCLC_IMMUNEX012_HD_$(date +%Y%m%d_%H%M%S)\
  --output-dir=/mnt/data/IMMUNEX/spaceranger_output/spaceranger_IMMUNEX012 \
  --transcriptome=/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/REF/refdata-gex-GRCh38-2020-A \
  --create-bam=false \
  --fastqs=/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/RAWDATA/IMMUNEX012 \
  --sample=IMMUNEX012 \
  --image=/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/IMAGE/HE_nanozoomer_tif/IMMUNEX012_Visium_HE_x40_z0.tif \
  --cytaimage=/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/IMAGE/CytAssist/assay_CAVG10047_NJ306_15janv2025_H1-NCFHZKX_1736942000_CytAssist/IMMUNEX012.tif \
  --slide=H1-NCFHZKX --area=A1 \
  --loupe-alignment=/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/IMAGE/json_allignment/H1-NCFHZKX-A1-fiducials-image-registration.json \
  --probe-set=/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/Visium_Human_Transcriptome_Probe_Set_v2.0_GRCh38-2020-A.csv \
  --filter-probes=false \
  --nucleus-segmentation=true \
  --nucleus-expansion-distance-micron=8 \
  --localcores=100 --localmem=400




# IMMUNEX001_Visium_HE_x40_z0.tif     IMMUNEX006_Visium_HE_20_x20_z0.tif  IMMUNEX011_Visium_HE_x40_z0.tif  IMMUNEX016_Visium_HE_x40_z0.tif
# IMMUNEX002_Visium_HE_x40_z0.tif     IMMUNEX007_Visium_HE_x40_z0.tif     IMMUNEX012_Visium_HE_x40_z0.tif  IMMUNEX017_Visium_HE_x40_z0.tif
# IMMUNEX003_Visium_HE_20_x20_z0.tif  IMMUNEX008_Visium_HE_x40_z0.tif     IMMUNEX013_Visium_HE_x40_z0.tif  IMMUNEX018_Visium_HE_x40_z0.tif
# IMMUNEX004_Visium_HE_20_x20_z0.tif  IMMUNEX009_Visium_HE_20_x20_z0.tif  IMMUNEX014_Visium_HE_x40_z0.tif
# IMMUNEX005_Visium_HE_20_x20_z0.tif  IMMUNEX010_Visium_HE_20_x20_z0.tif  IMMUNEX015_Visium_HE_x40_z0.tif





"/opt/spaceranger-4.0.1/bin/spaceranger" count \
  --id="$ID" \
  --output-dir=/mnt/data/IMMUNEX/spaceranger_output/spaceranger_IMMUNEX004 \
  --transcriptome=/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/REF/refdata-gex-GRCh38-2020-A \
  --create-bam=false \
  --fastqs=/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/RAWDATA/IMMUNEX004 \
  --sample=IMMUNEX004 \
  --image=/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/IMAGE/HE_nanozoomer_tif/IMMUNEX004_Visium_HE_20_x20_z0.tif \
  --cytaimage=/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/IMAGE/CytAssist/assay_CAVG10047_NJ289-14112024_H1-ZVB979D_1731671385_CytAssist/IMMUNEX004.tif \
  --slide=H1-X9BBRQR --area=A1 \
  --loupe-alignment=/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/IMAGE/json_allignment/H1-X9BBRQR-D1-fiducials-image-registration.json \
  --probe-set=/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/Visium_Human_Transcriptome_Probe_Set_v2.0_GRCh38-2020-A.csv \
  --filter-probes=false \
  --nucleus-segmentation=true \
  --nucleus-expansion-distance-micron=8 \
  --localcores=100 --localmem=400




"/opt/spaceranger-4.0.1/bin/spaceranger" count \
  --id=Visium_NSCLC_IMMUNEX015_HD_$(date +%Y%m%d_%H%M%S)\
  --output-dir=/mnt/data/IMMUNEX/spaceranger_output/spaceranger_IMMUNEX015 \
  --transcriptome=/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/REF/refdata-gex-GRCh38-2020-A \
  --create-bam=false \
  --fastqs=/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/RAWDATA/IMMUNEX015 \
  --sample=IMMUNEX015 \
  --image=/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/IMAGE/HE_nanozoomer_tif/IMMUNEX015_Visium_HE_x40_z0.tif \
  --cytaimage=/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/IMAGE/CytAssist/assay_CAVG10047_NJ309_22janv2025_H1-7BFQC6K_1737542699_CytAssist/IMMUNEX012.tif \
  --slide=H1-7BFQC6K --area=D1 \
  --loupe-alignment=/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/IMAGE/json_allignment/H1-7BFQC6K-D1-fiducials-image-registration.json \
  --probe-set=/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/Visium_Human_Transcriptome_Probe_Set_v2.0_GRCh38-2020-A.csv \
  --filter-probes=false \
  --nucleus-segmentation=true \
  --nucleus-expansion-distance-micron=8 \
  --localcores=100 --localmem=400
