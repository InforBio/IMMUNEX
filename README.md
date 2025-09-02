# IMMUNEX
Analysis of Visium-HD data from 18 NSCLC samples.

## Pipeline:
### 1. Segmentation with HE mask

This script processes **Visium HD (2 µm bins)** samples with the stardist workflow.  
It runs QC, filtering, destriping, Stardist segmentation, and generates tissue masks and cropped visualizations.

---

## 📂 Script

File: `1.bin2cell_segment_and_binary_mask.py`

---

## ⚙️ Dependencies

- Python ≥ 3.8  
- Packages:
  - `scanpy`, `anndata`, `numpy`, `pandas`, `scipy`
  - `matplotlib`, `tifffile`, `cv2` (OpenCV)
  - `skimage`, `stardist`, `csbdeep`
  - `tqdm`

Install environment (example with conda):

```bash
conda create -n bin2cell python=3.8
conda activate bin2cell
pip install scanpy anndata numpy pandas scipy matplotlib tifffile opencv-python scikit-image stardist csbdeep tqdm
