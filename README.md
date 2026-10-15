# Cortex-Medullary Identification
An automated whole-slide image analysis pipeline for identifying and delineating cortical and medullary regions in kidney histopathology images using patch-level feature extraction and Support Vector Machine (SVM) classification.

## Installation

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
pip install -r requirements-svc.txt
```

## 1. Run WSI processing with Trident

See the [Trident WSI processing guide](https://trident-docs.readthedocs.io/en/latest/quickstart.html) for processing options. From the project directory on a HiPerGator login node, submit the batch job:

```bash
sbatch run_batch_slide.sh ./wsi_dir ./trident_processed
```

`wsi_dir` contains the slides (`.svs` by default). `trident_processed` receives the Trident results, including tile feature H5 files. Wait for this job to finish before labeling.

## 2. Label tiles from annotations

Place annotation masks in folders named after each WSI file stem (for example, `sample.svs` uses `sample/`):

```text
annotation_dir/
  sample1/
    cortical_gt.json
    medullary_gt.json
  sample2/
    cortical_gt.json
```

The JSON files contain cortex and medullary regions in level-zero WSI coordinates. Run:

```bash
sbatch batch_labelling.sh ./annotation_dir ./path/to/trident_processed
```

This adds a `labels` dataset to each matching feature H5 file: `0` = unlabeled, `1` = cortex, `2` = medullary. The run report is `/path/to/trident_processed/batch_labelling_report.csv`.

## 3. Create train, validation, and test splits

After labeling finishes, create a CSV manifest from the labeled H5 files:

```bash
conda activate trident
python -m svc.split_h5 --job-dir ./output_dir --output ./output_dir/splits.csv
```

`splits.csv` lists each WSI, its feature H5 path, and its assigned split. The default is an approximately 70/15/15 split by WSI with seed 42. Review the file before training; slides from the same patient should stay in one split. If multiple feature sets exist, add `--features-dir PATH_TO_FEATURES_DIR`.

## 4. Train and evaluate the SVC

```bash
sbatch scripts/train_and_eval.sbatch ./output_dir/splits.csv ./models
```

The job fits on **train**, evaluates **val**, and reports final performance on held-out **test**. It writes `models/svc_pipeline.pkl`, `train_metrics.json`, `val_metrics.json`, `test_metrics.json`, `test_confusion_matrix.png`, and `test_tile_predictions.csv`.
