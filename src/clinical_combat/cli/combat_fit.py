#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compute the transfer function from a moving site to a reference site.

Harmonization methods:
    pairwise: uses both moving and reference data to fit the covariate
             regression parameters (Beta_mov).
             Jodoin et al., 2025 method,
             see https://www.nature.com/articles/s41598-025-25400-x
    clinical: uses a priori from the reference site to fit the moving site
            (Beta_mov, variance)
    gam: fits the covariate effect with a generalized additive model (spline on age)
         while estimating site effects with gamma/delta as in ComBat-GAM.
    covbat: runs pairwise ComBat then aligns covariance structure in a shared
            principal component space (Chen et al., 2021).

Examples:
# Use the pairwise method to harmonize the moving site data to
# the reference site data (linear)
combat_fit reference_site.raw.csv.gz moving_site.raw.csv.gz \
                 --method pairwise

# Use the clinical method to harmonize the moving site data to
# the reference site data (non-linear)
combat_fit reference_site.raw.csv.gz moving_site.raw.csv.gz \
                 --method clinical
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from clinical_combat.robust.robust import add_outlier_column, filter_outliers_by_threshold, DEFAULT_THRESHOLDS
from clinical_combat.harmonization import from_model_name
from clinical_combat.utils.scilpy_utils import (
    add_overwrite_arg,
    add_verbose_arg,
    assert_outputs_exist,
)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("ref_data",
                   help="Path to the reference site data.")
    p.add_argument("mov_data",
                   help="Path to the moving site data.")
    p.add_argument("--out_dir",
                   default="./",
                   help="Output directory.[%(default)s]")
    p.add_argument("-o", "--output_model_filename",
                   default="",
                   help="Output CSV model filename."
                        "['ref_site-moving-site.model.metric_name.method.model.csv']")
    p.add_argument("-m", "--method",
                   default="clinical",
                   choices=["pairwise", "clinical", "gam", "covbat"],
                   help="Harmonization method.")
    p.add_argument("--ignore_sex",
                   action="store_true",
                   help="If set, ignore the sex covariate in the data.")
    p.add_argument("--ignore_handedness",
                   action="store_true",
                   help="If set, ignore the handedness covariate in the data.")
    p.add_argument("--limit_age_range",
                   action="store_true",
                   help="If set, exclude reference site subjects with age"
                        " outside the range of the moving site subject ages.")
    p.add_argument("--no_empirical_bayes",
                   action="store_true",
                   help="If set, skip empirical Bayes estimator"
                        " for alpha and sigma estimation.")
    p.add_argument("--robust",
                   default="HC",
                   help="Robust outlier method applied to the moving site before fitting "
                        "(e.g., MAD, IQR, VS, MLP2_ALL, HC). Use 'NO' to disable. "
                        "[%(default)s]")
    p.add_argument("--robust_threshold",
                   type=float,
                   default=None,
                   help="Override the robust filtering threshold. "
                        "If omitted, method defaults are used.")
    p.add_argument("--regul_ref",
                   type=float,
                   default=0,
                   help="Regularization parameter for"
                        " the reference site data. [%(default)s]")
    p.add_argument("--regul_mov",
                   type=float,
                   help="Regularization parameter for"
                        " the moving site data. "
                        "Set to '-1' for automatic tuning "
                        "[default=0 for pairwise; -1 for clinical]")
    p.add_argument("--degree",
                   type=int,
                   help="Degree of the polynomial fit in Combat."
                        " Default is linear [default=1 for pairwise;"
                        " 2 for clinical].")
    p.add_argument("--nu",
                   type=float,
                   default=5,
                   help="Combat Clinical hyperparameter for"
                        " the standard deviation estimation of the moving "
                        "site data. It must be >=0.  [%(default)s]")
    p.add_argument("--tau",
                   type=float,
                   default=2,
                   help="Combat Clinical hyperparameter for "
                        "the covariate fit of the moving site data. "
                        "It must be >= 1. [%(default)s]")
    p.add_argument("--smooth_terms",
                   nargs="+",
                   default=["age"],
                   help="Covariates to smooth with GAM. Use 'none' to disable smoothing. [%(default)s]")
    p.add_argument("--df_spline",
                   type=int,
                   default=10,
                   help="Number of spline basis functions for each GAM smooth term. [%(default)s]")
    p.add_argument("--spline_degree",
                   type=int,
                   default=3,
                   help="Degree of the GAM B-spline basis. [%(default)s]")
    p.add_argument("--smooth_lower",
                   type=float,
                   default=None,
                   help="Optional lower bound for GAM spline knots.")
    p.add_argument("--smooth_upper",
                   type=float,
                   default=None,
                   help="Optional upper bound for GAM spline knots.")
    p.add_argument("--covbat_pve",
                   type=float,
                   default=0.95,
                   help="CovBat: cumulative variance threshold for PCs. [%(default)s]")
    p.add_argument("--covbat_max_components",
                   type=int,
                   default=None,
                   help="CovBat: optional maximum number of PCs to use.")
    p.add_argument("--ignore_bundles",
                   nargs="+",
                   default=['left_ventricle', 'right_ventricle'],
                   help="List of bundle to ignore.")
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.getLevelName(args.verbose))
    
    if args.regul_mov is None:
        if args.method in ["pairwise", "gam", "covbat"]:
            args.regul_mov = 0
        else:
            args.regul_mov = -1

    if args.degree is None:
        if args.method in ["pairwise", "gam", "covbat"]:
            args.degree = 1
        else:
            args.degree = 2

    if args.smooth_terms == ["none"]:
        args.smooth_terms = []

    ref_data = pd.read_csv(args.ref_data)
    ref_data = ref_data[~ref_data['bundle'].isin(args.ignore_bundles)]
    mov_data = pd.read_csv(args.mov_data)
    mov_data = mov_data[~mov_data['bundle'].isin(args.ignore_bundles)]

    logging.info("Bundles: %s will be ignored.", args.ignore_bundles)

    # Check if moving site is a string
    if mov_data.site.dtype != "str":
        mov_data.site = mov_data.site.astype(str)
    if ref_data.site.dtype != "str":
        ref_data.site = ref_data.site.astype(str)

    if len(np.unique(ref_data["site"])) != 1:
        raise AssertionError("The reference data contains more than one site.")
    if len(np.unique(mov_data["site"])) != 1:
        raise AssertionError("The moving data contains more than one site.")
    if np.unique(ref_data["metric"]) != np.unique(mov_data["metric"]):
        raise AssertionError("Data file have different metrics.")

    robust_method = str(args.robust).upper() if args.robust else "NO"
    if robust_method != "NO":
        logging.info(
            "Applying robust filtering with method=%s and threshold=%s",
            robust_method,
            args.robust_threshold if args.robust_threshold is not None else "default",
        )
        mov_data = filter_outliers_by_threshold(
            mov_data, robust_method, args.robust_threshold)

    cols = list(DEFAULT_THRESHOLDS.keys())
    cols_to_drop = [c for c in cols if c in mov_data.columns]
    mov_data =  mov_data.drop(columns=cols_to_drop)

    if args.output_model_filename == "":
        output_filename = os.path.join(
            args.out_dir,
            str(np.unique(mov_data["site"])[0])
            + "-"
            + str(np.unique(ref_data["site"])[0])
            + "."
            + str(np.unique(ref_data["metric"])[0])
            + "."
            + args.method.lower()
            + ".model.csv",
        )
    else:
        output_filename = os.path.join(args.out_dir,
                                       args.output_model_filename)
    os.makedirs(args.out_dir, exist_ok=True)
    assert_outputs_exist(parser, args, output_filename, check_dir_exists=True)

    QC = from_model_name(
        args.method.lower(),
        ignore_handedness_covariate=args.ignore_handedness,
        ignore_sex_covariate=args.ignore_sex,
        use_empirical_bayes=not args.no_empirical_bayes,
        limit_age_range=args.limit_age_range,
        degree=args.degree,
        regul_ref=args.regul_ref,
        regul_mov=args.regul_mov,
        nu=args.nu,
        tau=args.tau,
        covbat_pve=args.covbat_pve,
        covbat_max_components=args.covbat_max_components,
        smooth_terms=args.smooth_terms,
        smooth_term_bounds=(args.smooth_lower, args.smooth_upper),
        df_spline=args.df_spline,
        spline_degree=args.spline_degree,
    )

    QC.fit(ref_data, mov_data)

    logging.info("Saving file: %s", output_filename)
    QC.save_model(output_filename)


if __name__ == "__main__":
    main()
