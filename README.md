# cortex-medullary Identification

## Installation

On HiPerGator:

```bash
git clone https://github.com/UFerminJamie/cortex-medullary-identification.git cortex_medullary_identification
cd cortex_medullary_identification
module load conda
conda create -n trident python=3.10 -y
conda activate trident
git clone https://github.com/mahmoodlab/TRIDENT.git TRIDENT
cd TRIDENT
pip install -e .
cd ..
pip install numpy h5py shapely
mkdir -p logs
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

This adds a `labels` dataset to each matching feature H5 file: `0` = unlabeled, `1` = cortex, `2` = medullary. Check `output_dir/batch_labelling_report.csv` for missing or failed samples before splitting.

## 3. Create train, validation, and test splits

After labeling finishes, create a CSV manifest from the labeled H5 files:

```bash
conda activate trident
python -m svc.split_h5 --job-dir ./output_dir --output ./output_dir/splits.csv
```

`splits.csv` lists each WSI, its feature H5 path, and its assigned split. The default is an approximately 70/15/15 split by WSI with seed 42. Review the file before training; slides from the same patient should stay in one split. If multiple feature sets exist, add `--features-dir PATH_TO_FEATURES_DIR`.

## 4. Train and evaluate the SVC

Use a separate `rapids_svc` Conda environment with cuML and CuPy installed for HiPerGator's CUDA version (see the [RAPIDS installation guide](https://docs.nvidia.com/datascience/install/)). Install the remaining Python packages, then submit the GPU job from the project directory:

```bash
conda activate rapids_svc
pip install -r requirements-svc.txt
sbatch scripts/train_and_eval.sbatch ./output_dir/splits.csv ./models
```

The job fits on **train**, evaluates **val** without fitting on it, and reports final performance on held-out **test**. It writes `models/svc_pipeline.pkl`, `train_metrics.json`, `val_metrics.json`, `test_metrics.json`, `test_confusion_matrix.png`, and `test_tile_predictions.csv`. Set `SVC_CONDA_ENV` if your RAPIDS environment has a different name or path. The default PCA size is 1024; set `N_COMPONENTS` to at most the smaller of the training tile count and feature width if needed.
