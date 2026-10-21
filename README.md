# ComBat-Robust

This repository implements **Robust ComBat**, a method designed to detect and remove outliers prior to harmonization. The goal is to improve the robustness of ComBat-based harmonization when datasets contain pathological or anomalous samples that could bias parameter estimation.

This repository is a fork of [Clinical-ComBAT](https://github.com/scil-vital/clinical-ComBAT). All detailed documentation for harmonization, quality control, and visualization is available there; this README remains intentionally short and focuses on the robust ComBat variant. The data used in the paper are available on [Zenodo](https://doi.org/10.5281/zenodo.18802256) and match the dataset used in our experiments.

## References

- David, Y., Jodoin, P.-M., et al. (2026). *Robust-ComBat: Mitigating Outlier Effects in Diffusion MRI Data Harmonization*. arXiv:2603.17968. https://arxiv.org/abs/2603.17968
- Girard, G., Edde, M., Dumais, F., et al. (2025). *Clinical-ComBAT: a diffusion-weighted MRI harmonization method for clinical applications*. arXiv:2511.04871. https://arxiv.org/abs/2511.04871
- Jodoin, P.-M., Edde, M., Girard, G., et al. (2025). Challenges and best practices when using ComBAT to harmonize diffusion MRI data. *Scientific Reports*, 15, 41508. https://www.nature.com/articles/s41598-025-25400-x
- Fortin, J.-P., Parker, D., Tunç, B., et al. (2017). Harmonization of multi-site diffusion tensor imaging data. *NeuroImage*, 161, 149–170. https://doi.org/10.1016/j.neuroimage.2017.08.047

## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

You can copy, redistribute, and adapt this work, but only for **non-commercial purposes**. You must give the original creator credit (Attribution), and if you adapt the work, your new version must be shared under the same or a compatible license (ShareAlike). The work cannot be used for commercial gain, meaning for activities primarily intended to generate money. In such case, please contact [Pierre-Marc Jodoin](mailto:pierre-marc.jodoin@usherbrooke.ca).

## Quick installation

These commands were tested on a fresh Ubuntu 22.04 and 24.04 with Python 3.10 to 3.12. We recommend installing [uv](https://docs.astral.sh/uv/getting-started/installation/) to speed up the install.

```bash
git clone https://github.com/scil-vital/ComBat-Robust.git
cd ComBat-Robust
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Without uv, create the environment and install with pip instead (Debian and Ubuntu need the `python3-venv` package):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Main dependencies: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `torch`. The install takes about 5 GB on disk on Linux (about 3 GB downloaded), mostly the CUDA build of `torch` (the code runs on CPU). Scripts accept compressed or uncompressed CSV files.

To run the tests, from the repository root (they take a few minutes):
```bash
uv pip install -e ".[dev]"
pytest src
```
Without uv, install with `pip install -e ".[dev]"` instead.

The file `requirements-lock.txt` pins the exact versions tested (Linux, Python 3.12 only, some pinned versions need Python 3.11 or later). To install them:
```bash
uv pip install -r requirements-lock.txt
uv pip install -e . --no-deps
```
Without uv, run `pip install -r requirements-lock.txt` and then `pip install -e . --no-deps`.

## Expected data format

CSV files must contain at least the columns below:

```
sid,site,bundle,metric,mean,age,sex,handedness,disease
```

- `sid`: subject identifier
- `site`: site name
- `bundle`: bundle or region name
- `metric`: diffusion metric (e.g., `md`, `fa`)
- `mean`: numeric value per bundle (mean, median, etc.)
- `age`, `sex`, `handedness`: covariates
  - use integer values (1 or 2) for `sex` and `handedness`; if unknown, fill with `1` and the scripts will disable that effect
- `disease`: flag; any row whose value is not `HC` is dropped before model fitting if the HC robust method is used

Complete examples (`CamCAN.md.raw.csv.gz` and `ModifiedCamCAN.md.raw.csv.gz`) are available in `src/clinical_combat/data/`. The paper data are available on [Zenodo](https://doi.org/10.5281/zenodo.18802256).

## Quick example

The scripts below rewrite the CSV files they receive. We therefore work on copies of the example files in a `demo/` folder, so that the files tracked in `src/clinical_combat/data/` stay untouched. Run all commands from the repository root:
```bash
mkdir -p demo
cp src/clinical_combat/data/70_pct_example_*.csv \
   src/clinical_combat/data/CamCAN.rd.raw.csv.gz \
   src/clinical_combat/data/ModifiedCamCAN.md.raw.csv.gz \
   src/clinical_combat/data/CamCAN.md.raw.csv.gz demo/
```

### Step 1: Compute outlier scores (Robust ComBat)

First, run the `combat_robust` script with the outlier method of your choice (here, `MLP_TEST`):
```bash
combat_robust \
    demo/70_pct_example_*.csv \
    --method MLP_TEST
```
This adds an `MLP_TEST` outlier score column to each file. You can then filter observations based on this score before harmonization.

`MLP_TEST` is the model used in the paper. It needs the 10 metrics (one file per metric) and the 43 bundles of the example data. The files are rewritten in place and their rows can be reordered. MLP weights are read from `Pytorch_models/` in the current directory, so run MLP methods from the repository root.

### Step 2: Run the standard ComBat pipeline

Then run the `combat_pipeline` script to harmonize the data using the outlier method defined above:
```bash
combat_pipeline \
    demo/CamCAN.rd.raw.csv.gz \
    demo/70_pct_example_rd.csv \
    --method pairwise \
    --robust MLP_TEST \
    --out_dir quicktest_demo_MLP/ \
    --bundles "mni_ICP_R"
```
In a single command, the pipeline chains fit → apply → figures → QC. Main outputs include the model, the harmonized data, and QC metrics/figures inside `quicktest_demo_MLP/`.

### Comparison without filtering

You can compare these results with the standard pipeline without outlier filtering:
```bash
combat_pipeline \
    demo/CamCAN.rd.raw.csv.gz \
    demo/70_pct_example_rd.csv \
    --method pairwise \
    --robust NO \
    --out_dir quicktest_demo_NO/ \
    --bundles "mni_ICP_R"
```
## Main scripts

### Robust outlier scoring (`combat_robust`)

`combat_robust` applies the selected outlier detection method to the provided CSV files, computes an outlier score for each row, and appends these columns to the files before harmonization. All files must be from the same site. The files are saved back with the same name, and no rows are dropped at this stage. Actual filtering is performed later in the pipeline script, where you specify the robust method and optionally the associated threshold.

- `data` *(required, one or more from the same site)*: CSV paths or glob patterns (`*`) to process.
- `--method` (default `MAD`, case-insensitive): outlier detection method.
  - Bundle-level methods (one score per subject and bundle): `ZS`, `MAD`, `SN`, `QN`, `VS`, `MMS`.
  - Subject-level methods (one score per subject, computed jointly on all metrics and bundles of the site): `G_ZS`, `G_MAD`.
  - `PCA_ZS`, `PCA_MAD`: `G_ZS` and `G_MAD` applied to the first 15 principal components of the standardized features of the site, which limits the weight of collinear metrics.
  - `MLP*`: neural network–based outlier detection (subject level).
    - `MLP_TEST`: uses the pre-trained model of the paper provided in this repository (it needs the 10 metrics and the 43 bundles of the example data).
    - `MLP_<model_name>`: uses a custom trained MLP model (for example `MLP_EXAMPLE`, trained in notebook 4).
  - `IQR`, `HC` and `NO` add no column, because they act when the model is fitted (`--robust` of `combat_fit` and `combat_pipeline`).
    - `IQR`: removes values outside the interquartile fences of each bundle.
    - `HC`: uses only Healthy Controls (ideal scenario).
    - `NO`: no outlier filtering, all subjects are used as-is.
- `--metrics` (default `ad adt afd fa fat fw md mdt rd rdt`): allowed metrics. Each file must hold a single metric, and files containing others are rejected.
- `-v` (default `WARNING`, `INFO` with `-v`): logging level.
- Writes back to the same files, adding a column named after `--method` with the score/flag.

Example:
```bash
combat_robust \
    demo/ModifiedCamCAN.md.raw.csv.gz \
    --method MAD
```

### Combined workflow (`combat_pipeline`)

`combat_pipeline` runs fit → apply → figures → QC and logs each spawned command with `-v`.

- `ref_data` *(required)*: reference-site CSV (`*.raw.csv[.gz]`).
- `mov_data` *(required)*: moving-site CSV to harmonize.
- `--out_dir` (default `./`): root directory for models, harmonized data, QC, and figures.
- `--output_model_filename` (default auto): custom model filename.
- `--output_results_filename` (default auto): custom harmonized CSV filename.
- `--save_curves_json` (default false): also export regression curves/percentiles as JSON.
- `--method {clinical,pairwise,gam,covbat}` (default `clinical`): harmonization strategy.
- `--ignore_sex`, `--ignore_handedness` (default false): drop these covariates.
- `--limit_age_range` (default false): restrict reference subjects to the moving-site age span.
- `--no_empirical_bayes` (default false): skip empirical Bayes estimates.
- `--robust` (default `HC`): outlier filtering on moving data before fit. Any `combat_robust` method (its score column must already be in the file), or `IQR`, `HC`, `NO`.
- `--robust_threshold` (default method-specific, see below): override the robust filtering threshold.
- `--regul_ref` (default 0, clinical only): ridge penalty on the reference regression.
- `--regul_mov` (default -1, clinical only): moving-site penalty, -1 for auto-tuning.
- `--degree` (default 2 clinical, 1 pairwise): polynomial degree for age.
- `--nu` (default 5, clinical): variance hyperparameter for the moving site.
- `--tau` (default 2, clinical): covariate hyperparameter for the moving site.
- `--smooth_terms` (default `age`, GAM only): covariates to smooth; use `none` to disable.
- `--df_spline` (default 10): number of spline basis functions for each GAM term.
- `--spline_degree` (default 3): B-spline degree for GAM.
- `--smooth_lower/--smooth_upper` (default None): optional GAM knot bounds.
- `--covbat_pve` (default 0.95): CovBat cumulative variance threshold for retained PCs.
- `--covbat_max_components` (default None): maximum PCs to use in CovBat.
- `--bundles` (default first skeleton bundle): bundles to plot (`all` for every bundle).
- `--degree_qc` (default 0): QC model degree (0 reuses harmonization degree).
- `-v` (default `WARNING`, `INFO` with `-v`): logging verbosity.
- `-f` (default false): allow overwriting existing files.

Default thresholds of `--robust`. A row is removed when its score is greater than or equal to the threshold.

| Method | Default threshold |
|---|---|
| `ZS`, `MAD`, `SN`, `QN` | 3.0 |
| `IQR` | 1.5 (fence multiplier) |
| `VS`, `MMS` | flagged rows (binary flag) |
| `G_ZS`, `G_MAD`, `PCA_ZS`, `PCA_MAD` | 1.0 |
| `MLP*` | 0.6 |

Sequence:
1. Fits the model with `combat_fit` (applying robust filtering if set) and saves `*.model.csv`.
2. Applies it with `combat_apply` to produce harmonized `*.harmonized.csv[.gz]`.
3. Generates figures with `combat_visualize_model` and `combat_visualize_harmonization` (optionally also JSON curves).
4. Runs `combat_QC` on raw and harmonized data to report distances/QC metrics.

Example:

```bash
combat_pipeline demo/CamCAN.md.raw.csv.gz \
    demo/ModifiedCamCAN.md.raw.csv.gz \
    --method clinical \
    --out_dir results/clinical_pipeline/ \
    --robust MAD \
    --bundles all \
    -v INFO
```
This example needs the `MAD` column that the `combat_robust` example above adds to `demo/ModifiedCamCAN.md.raw.csv.gz`.

## Notebooks

The notebooks below rerun most of the experiments reported in the paper from the raw data.

1. Download `DATA.zip` from the [Zenodo record](https://doi.org/10.5281/zenodo.18802256) and unzip it at the repository root, so that `DATA/raw` and `DATA/raw_camcan` exist.
2. Install Jupyter in the activated environment (`uv pip install jupyter`, or `pip install jupyter` without uv).
3. Start Jupyter from the repository root, with the environment activated, and run the notebooks in order, from 0 to 7. Each notebook uses the outputs of the previous ones and never the stored results of the paper. The results go to `DATA/processed` and `RESULTS`. Two kinds of files are written elsewhere. Notebook 1 rewrites the files of three sites in `DATA/raw` in place (see below), and notebook 4 saves the trained model as `Pytorch_models/mlp_example_weights.pt` and `Pytorch_models/mlp_example_params.json`.

Notebook 4 trains the MLP (`MLP_EXAMPLE`) used by notebooks 5 to 7, while the trained model of the paper is shipped as `MLP_TEST`. The notebooks are not fully seeded. Notebook 4 seeds numpy, random and torch, but the synthetic sites drawn in parallel (notebook 3, and the training sites of the MLP in notebook 4) and the Optuna search are not seeded, so the numbers differ slightly from the paper. Three sites of the Zenodo data (`35343`, `35426` and `MRN`) store the handedness as True/False, and notebook 1 converts it to 1 and 2 in place.

- `0-CamCAN_Cleaning.ipynb`: removes the outlier subjects from the raw CamCAN cohort.
- `1-Harmonized_Dataset_Generation.ipynb`: harmonizes every site with at least 8 HC and 5 patients to the cleaned CamCAN.
- `2-Dataset_Split_Augment.ipynb`: splits the data into MLP training and harmonization testing sets, and augments the data.
- `3-Synthetic_Site_Generation.ipynb`: generates synthetic sites to test harmonization and outlier handling.
- `4-MLP_Training.ipynb`: trains the MLP outlier detector (`MLP_EXAMPLE`).
- `5-STD_MAE_Calculations.ipynb`: computes STD_MAE metrics across outlier handling methods and synthetic sites, with pairwise ComBat. Setting `COMBAT_VARIANTS = True` in the notebook adds Clinical-ComBat, ComBat-GAM and CovBat, but ComBat-GAM stops on the `HC` method when patients are outside the age range of the healthy controls.
- `6-STD_MAE_Plots.ipynb`: creates the plots used in the manuscript from the results obtained in notebook 5.
- `7-Effect_Size_Preservation.ipynb`: reads the outputs of notebooks 3 and 5 to measure how well each outlier method preserves the effect size of each disease (Cohen's d against CamCAN), and writes `RESULTS/EFFECT_SIZE/`.
