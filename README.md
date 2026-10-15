# cortex-medullary Identification
An automated whole-slide image analysis pipeline for identifying and delineating cortical and medullary regions in kidney histopathology images using patch-level feature extraction and Support Vector Machine (SVM) classification.

## Installation

On HiPerGator:

```bash
git clone https://github.com/UFerminJamie/cortex-medullary-identification.git 
cd cortex_medullary_identification
module load conda
conda create -n venv python=3.10 -y
conda activate venv
git clone https://github.com/mahmoodlab/TRIDENT.git TRIDENT
cd TRIDENT
pip install -e .
cd ..
pip install numpy h5py shapely
```

## 1. Run WSI processing with Trident

See the [Trident WSI processing guide](https://trident-docs.readthedocs.io/en/latest/quickstart.html) for processing options. From the project directory on a HiPerGator login node, submit the batch job:

```bash
sbatch run_batch_slide.sh ./wsi_dir ./output_dir
```

`wsi_dir` contains the slides (`.svs` by default). `output_dir` receives the Trident results, including tile feature H5 files. Wait for this job to finish before labeling.

## 2. Label tiles from annotations

Place annotation masks in folders named after each WSI file stem (for example, `sample.svs` uses `sample/`):

```text
annotation_dir/
  sample/
    cortical_gt.json
    medullary_gt.json
```

The JSON files contain cortex and medullary regions in level-zero WSI coordinates. Run:

```bash
sbatch batch_labelling.sh ./annotation_dir ./output_dir
```

This adds a `labels` dataset to each matching feature H5 file: `0` = unlabeled, `1` = cortex, `2` = medullary. The run report is `output_dir/batch_labelling_report.csv`.
