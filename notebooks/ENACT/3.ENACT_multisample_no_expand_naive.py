import os, sys
# sys.path.append('/home/mounim/pipelines/enact-pipeline')
sys.path.append('/home/mounim/pipelines/enact-pipeline/src')

from enact.pipeline import ENACT
import os
import gc

images = os.listdir('/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/IMAGE/HE_nanozoomer_tif')
runonly = []
skipsamples = ['']

export_dir = '/scratch/Projects/IMMUNEX/results/enact_results'


cell_markers = {
    "NSCLC_Drivers": [
        "EGFR","KRAS","BRAF","ALK","RET","ROS1","MET","ERBB2","PIK3CA",
        "TP53","KEAP1","STK11","NF1","CDKN2A","PTEN","RB1","NFE2L2","NOTCH1",
        "TERT","MLL2","FAT1","BRCA2","APC","CTNNB1","BAP1","SMARCA4","CDK4","CDK6","CCND1","MYC"
    ],
    "EMT_and_Metastasis": [
        "CDH1","CDH2","VIM","FN1","SNAI1","SNAI2","TWIST1","TWIST2","ZEB1","ZEB2","TGFBR1","TGFBR2","TGFB1",
        "TGFB2","MMP2","MMP3","MMP9","SERPINE1","SPARC","ACTA2","FAP","POSTN","COL1A1","COL1A2","COL3A1",
        "SPHK1","S100A4","LOX","LOXL2","PDPN","ITGA5","ITGB1"
    ],
    "Immune_Checkpoints_Cytokines": [
        "PDCD1","CD274","CTLA4","LAG3","HAVCR2","TIGIT","IDO1","IDO2","VISTA","CD80","CD86","CD40","CD40LG",
        "IFNG","TNF","IL1B","IL2","IL4","IL6","IL10","IL12A","IL12B","IL17A","TGFB1",
        "CXCL9","CXCL10","CXCL11","CXCL12","CCL2","CCL3","CCL4","CCL5","CCL17","CCL22"
    ],
    "T_Cells": [
        "CD3D","CD3E","CD3G","CD2","CD7","CD4","CD8A","CD8B","GZMA","GZMB","PRF1","NKG7",
        "LAG3","PDCD1","HAVCR2","ICOS","TNFRSF9","CTLA4","FOXP3","TIGIT","ENTPD1","IL2RA"
    ],
    "B_Cells_and_Plasma": [
        "CD19","MS4A1","CD79A","CD79B","CD22","SDC1","MZB1","XBP1","PRDM1","JCHAIN","TNFRSF17",
        "IGHM","IGHD","IGHG1","IGHG2","IGHA1","IGKC"
    ],
    "NK_Cells": [
        "KLRD1","KLRF1","KLRB1","KLRC1","KLRC2","FCGR3A","NKG7","GNLY","GZMM","GZMH","PRF1","SPON2","NCR1"
    ],
    "Myeloid_Macrophage": [
        "CD68","CD14","CD163","CD86","CD80","MRC1","CSF1R","ITGAM","LYZ","S100A8","S100A9","S100A12","MARCO",
        "CD36","VSIG4","TREM2","APOE","MSR1","CD11c","CD11b","CD16","CD64"
    ],
    "Dendritic_Cells": [
        "ITGAX","CD1C","CLEC9A","CLEC10A","HLA-DRA","HLA-DPB1","IRF8","IRF4","BATF3","XCR1","LAMP3","TCF4",
        "IL3RA","SIGLEC6","NRP1"
    ],
    "Fibroblasts_CAFs": [
        "ACTA2","FAP","PDGFRA","PDGFRB","COL1A1","COL1A2","COL3A1","COL6A3","DCN","THY1","SPARC","LUM","TAGLN",
        "CAV1","MMP2","MMP9","MMP11","POSTN","SRPX","PDPN","PDGFA"
    ],
    "Endothelial_Cells": [
        "PECAM1","CDH5","ENG","VWF","CD34","ESAM","FLT1","KDR","TIE1","ECSCR","PLVAP","CLDN5","MCAM","VEGFA",
        "VEGFB","ANGPT1","ANGPT2","TEK","FLT4","LYVE1","PROX1","PDPN","CCL21"
    ],
    "Proliferation_CellCycle": [
        "MKI67","TOP2A","PCNA","BIRC5","CCNB1","CDK1","AURKA","CENPF","UBE2C","TK1","STMN1","NUSAP1","MCM2",
        "MCM4","CDC20","CDCA8","CDKN3","E2F1","E2F2","E2F3"
    ],
    "DNA_Repair": [
        "BRCA1","BRCA2","RAD51","ATM","ATR","CHEK1","CHEK2","PARP1","MLH1","MSH2","MSH6","PMS2","FANCD2","BARD1",
        "NBN","XPA","ERCC1","POLQ","POLB"
    ],
    "Metabolism": [
        "SLC2A1","GLUT1","HK2","LDHA","PKM2","IDH1","IDH2","G6PD","PGD","SHMT2","ACLY","SCD","FASN","PPARG",
        "CPT1A","LDHB"
    ],
    "Hypoxia_Angiogenesis": [
        "HIF1A","CA9","ADM","EGLN3","BNIP3","NDRG1","VEGFA","VEGFB","ANGPT2","ANGPT1","TEK","FLT1","KDR"
    ],
    "ISGs_Interferon": [
        "STAT1","STAT2","IRF1","ISG15","IFI6","IFIT1","IFIT3","OAS1","MX1","BST2","LY6E","CXCL9","CXCL10","CXCL11"
    ],
    "Lung_Epithelial_Differentiation": [
        "AGER","SFTPA1","SFTPA2","SFTPC","SFTPD","SCGB1A1","SCGB3A2","NAPSA","CLDN18","HOPX","FOXA2","ABCA3",
        "LAMP3","CLDN4","CLDN7","TJP1","FOXJ1","TPPP3"
    ],
    "Cell_Stress_Apoptosis": [
        "BAX","BCL2","BCL2L1","BCL2L11","CASP3","CASP8","CASP9","FAS","FASLG","DR4","DR5","TP53","MDM2","BBC3","GADD45A"
    ],
    "Other_Pathways": [
        "NOTCH1","WNT1","WNT5A","CTNNB1","HEDGEHOG","GLI1","PTEN","PIK3CA","AKT1","MTOR","MAPK1","MAPK3","JUN","FOS"
    ]
}


if len(runonly):
    imagestokeep = []
    for tokeep in runonly:
        for image in images:
            if tokeep in image:
                imagestokeep.append(image)
    images = imagestokeep

for image in images[::-1]:
    sample = image[:10]
    
    out_name = f"250627_NoExpNaive-{sample}"

    sample_export_path = os.path.join(export_dir, out_name)
    if os.path.exists(sample_export_path) or sample in skipsamples:
        if len(images) > 2:
            print(f"Skipping {sample}: output already exists.")
            continue
    print(f"Processing {sample}")
    so_hd = ENACT(
        cache_dir=export_dir,
        wsi_path=f"/home/mounim/rawdata/IMMUNEX/PJ2410310_250214/IMAGE/HE_nanozoomer_tif/{image}",
        visiumhd_h5_path=f"/home/mounim/rawdata/IMMUNEX/OUTPUT/Visium_NSCLC_{sample}/outs/binned_outputs/square_002um/filtered_feature_bc_matrix.h5",
            tissue_positions_path=f"/home/mounim/rawdata/IMMUNEX/OUTPUT/Visium_NSCLC_{sample}/outs/binned_outputs/square_002um/spatial/tissue_positions.parquet",
        cell_typist_model="Immune_All_Low",
        analysis_name=out_name,
        n_hvg = 1000,
        cell_markers = cell_markers,
        patch_size = 5000,
        nucleus_expansion=False,
        expand_by_nbins=0,
        bin_to_cell_method="naive",

    )

    so_hd.run_enact()
    del so_hd
    gc.collect()