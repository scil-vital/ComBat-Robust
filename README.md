# ComBat-Robust

This repository implements **Robust ComBat**, a method designed to detect and remove outliers prior to harmonization. The goal is to improve the robustness of ComBat-based harmonization when datasets contain pathological or anomalous samples that could bias parameter estimation.

This repository is a fork of [Clinical-ComBAT](https://github.com/scil-vital/clinical-ComBAT). All detailed documentation for harmonization, quality control, and visualization is available there; this README remains intentionally short and focuses on the robust ComBat variant. Data used in the paper will be available on  [Zenodo](https://doi.org/10.5281/zenodo.18802256) and will match the dataset used in our experiments.

## References

- ComBat-Robust reference: forthcoming.
- Girard, G., Edde, M., Dumais, F., et al. (2025). *Clinical-ComBAT: a diffusion MRI harmonization method for clinical normative modeling applications*. Submitted to Medical Image Analysis.
- Jodoin, P.-M., Edde, M., Girard, G., et al. (2025). Challenges and best practices when using ComBAT to harmonize diffusion MRI data. *Nature Scientific Reports*, 15, 41508. https://www.nature.com/articles/s41598-025-25400-x
- Fortin, J.-P., Parker, D., Tun¸c, B., et al. (2017). Harmonization of multi-site diffusion tensor imaging data. *NeuroImage*, 161, 149–170. https://doi.org/10.1016/j.neuroimage.2017.08.047

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

:warning: We recommend installing uv to speed up the install: https://docs.astral.sh/uv/getting-started/installation/

:point_up: If you do not want to use uv, simply remove `uv` from the commands below.

Update `pip`:
```bash
uv pip install --upgrade pip
```

Then:
```bash
# 1) create a Python >= 3.9 environment
python -m venv .venv
source .venv/bin/activate

# 2) install clinical_combat
uv pip install -e .
```

Main dependencies: `numpy`, `pandas`, `matplotlib`, `seaborn`. Scripts accept compressed or uncompressed CSV files.

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

Complete examples (`CamCAN.md.raw.csv.gz` and `ModifiedCamCAN.md.raw.csv.gz`) are available in `src/clinical_combat/data/`. Paper data can be retrieved on Zenodo (DOI to be published).

## Quick example

### Step 1: Compute outlier scores (Robust ComBat)

First, run the `combat_robust` script with the outlier method of your choice (here, `MLP_TEST`):
```bash
combat_robust \
    src/clinical_combat/data/70_pct_example_*.csv \
    --method MLP_TEST
```
This adds an `MLP_TEST` outlier score column to each file. You can then filter observations based on this score before harmonization.

### Step 2: Run the standard ComBat pipeline

Then run the `combat_pipeline` script to harmonize the data using the outlier method defined above:
```bash
combat_pipeline \
    src/clinical_combat/data/CamCAN.rd.raw.csv.gz \
    src/clinical_combat/data/70_pct_example_rd.csv \
    --method pairwise \
    --robust MLP_TEST \
    --out_dir quicktest_demo_MLP/ \
    --bundle "mni_ICP_R"
```
In a single command, the pipeline chains fit → apply → QC → figures. Main outputs include the model, the harmonized data, and QC metrics/figures inside `quickstart_demo_MLP/`.

### Comparison without filtering

You can compare these results with the standard pipeline without outlier filtering:
```bash
combat_pipeline \
    src/clinical_combat/data/CamCAN.rd.raw.csv.gz \
    src/clinical_combat/data/70_pct_example_rd.csv \
    --method pairwise \
    --robust NO \
    --out_dir quicktest_demo_NO/ \
    --bundle "mni_ICP_R"
```
## Main scripts

### Robust outlier scoring (`combat_robust`)

`combat_robust` applies the selected outlier detection method to the provided CSV files, computes an outlier score for each row, and appends these columns to the files before harmonization. All files must be from the same site. The files are saved back with the same name, and no rows are dropped at this stage. Actual filtering is performed later in the pipeline script, where you specify the robust method and optionally the associated threshold.

- `data` *(required, one or more from the same site)*: CSV paths or glob patterns (`*`) to process.
- `--method` (default `HC`): outlier detection method.
  - Statistical methods: `ZS`, `MAD`, `IQR`, `SN`, `QN`, `VS`, `MMS`, `G_ZS`, `G_MAD`.
  - `MLP*`: neural network–based outlier detection.
    - `MLP_TEST`: uses the pre-trained model provided in this repository.
    - `MLP_<model_name>`: uses a custom trained MLP model.
  - `HC`: uses only Healthy Controls (ideal scenario).
  - `NO`: no outlier filtering, all subjects are used as-is.
- `--metrics` (default `ad adt afd fa fat fw md mdt rd rdt`): allowed metrics; files containing others are rejected.
- `--verbose/-v` (default `WARNING`; `INFO` with `-v`): logging level.
- Writes back to the same files, adding a column named after `--method` with the score/flag.

Example:
```bash
combat_robust \
     src/clinical_combat/data/ModifiedCamCAN.md.raw.csv.gz \
    --method MAD
```

### Combined workflow (`combat_pipeline`)

`combat_pipeline` runs fit → apply → QC → figures and logs each spawned command.

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
- `--robust` (default `HC`): outlier filtering on moving data before fit (MAD, IQR, VS, MLP2_ALL, HC, NO).
- `--robust_threshold` (default method-specific): override the robust filtering threshold.
- `--regul_ref` (default 0): ridge penalty on the reference regression.
- `--regul_mov` (default -1 for clinical, 0 for pairwise): moving-site penalty or auto-tuning.
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
- `--verbose/-v` (default `WARNING`; `INFO` with `-v`): logging verbosity.
- `--overwrite/-f` (default false): allow overwriting existing files.

Sequence:
1. Fits the model with `combat_fit` (applying robust filtering if set) and saves `*.model.csv`.
2. Applies it with `combat_apply` to produce harmonized `*.harmonized.csv[.gz]`.
3. Generates figures with `combat_visualize_model` and `combat_visualize_harmonization` (optionally also JSON curves).
4. Runs `combat_QC` on raw and harmonized data to report distances/QC metrics.

Example:

```bash
combat_pipeline src/clinical_combat/data/CamCAN.md.raw.csv.gz \
    src/clinical_combat/data/ModifiedCamCAN.md.raw.csv.gz \
    --method clinical \
    --out_dir results/clinical_pipeline/ \
    --robust MAD \
    --bundles all \
    -v INFO
```

## Notebooks

The notebooks below allow you to rerun most of the experiments reported in the paper by executing them sequentially:

- `0-CamCAN_Cleaning.ipynb`: cleans the raw CamCAN cohort (drops problematic subjects/bundles).
- `1-Harmonized_Dataset_Generation.ipynb`: builds the harmonized datasets using all available sites.
- `2-Dataset_Split_Augment.ipynb`: splits the data into MLP training and harmonization testing sets, and augments the data.
- `3-Synthetic_Site_Generation.ipynb`: generates synthetic sites to test harmonization and outlier handling.
- `4-MLP_Training.ipynb`: trains the MLP outlier detector variants.
- `5-STD_MAE_Calculations.ipynb`: computes STD_MAE metrics across harmonization methods, outlier handling methods, and synthetic sites.
- `6-STD_MAE_Plots.ipynb`: creates the plots used in the manuscript from the results obtained in notebook 5.
