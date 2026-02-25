# -*- coding: utf-8 -*-
from __future__ import absolute_import

import ast

from clinical_combat.harmonization.CombatClinical import CombatClinical
from clinical_combat.harmonization.Covbat import CovBat
from clinical_combat.harmonization.CombatGAM import CombatGAM
from clinical_combat.harmonization.CombatPairwise import CombatPairwise


def from_model_name(
    name,
    ignore_sex_covariate=False,
    ignore_handedness_covariate=False,
    use_empirical_bayes=True,
    limit_age_range=False,
    degree=1,
    regul=0,
    regul_ref=0,
    regul_mov=0,
    nu=0,
    tau=1,
    covbat_pve=0.95,
    covbat_max_components=None,
    smooth_terms=("age",),
    smooth_term_bounds=(None, None),
    df_spline=10,
    spline_degree=3,
):

    if name == "pairwise":
        QC = CombatPairwise(
            ignore_sex_covariate=ignore_sex_covariate,
            ignore_handedness_covariate=ignore_handedness_covariate,
            use_empirical_bayes=use_empirical_bayes,
            limit_age_range=limit_age_range,
            degree=degree,
            regul=regul
        )
    elif name == "clinical":
        QC = CombatClinical(
            ignore_sex_covariate=ignore_sex_covariate,
            ignore_handedness_covariate=ignore_handedness_covariate,
            use_empirical_bayes=use_empirical_bayes,
            limit_age_range=limit_age_range,
            degree=degree,
            regul_ref=regul_ref,
            regul_mov=regul_mov,
            nu=nu,
            tau=tau,
        )
    elif name == "gam":
        QC = CombatGAM(
            ignore_sex_covariate=ignore_sex_covariate,
            ignore_handedness_covariate=ignore_handedness_covariate,
            use_empirical_bayes=use_empirical_bayes,
            limit_age_range=limit_age_range,
            degree=degree,
            regul_ref=regul_ref,
            regul_mov=regul_mov,
            smooth_terms=smooth_terms,
            smooth_term_bounds=smooth_term_bounds,
            df_spline=df_spline,
            spline_degree=spline_degree,
        )
    elif name == "covbat":
        QC = CovBat(
            ignore_sex_covariate=ignore_sex_covariate,
            ignore_handedness_covariate=ignore_handedness_covariate,
            use_empirical_bayes=use_empirical_bayes,
            limit_age_range=limit_age_range,
            degree=degree,
            regul=regul,
            covbat_pve=covbat_pve,
            covbat_max_components=covbat_max_components,
        )
    else:
        raise AssertionError(
            name + " is an invalid value for the harmonization method."
        )
    return QC


def from_model_filename(model_filename):
    with open(model_filename) as f:
        model_params = ast.literal_eval(f.readline()[2:])

    model = from_model_name(model_params["name"])
    model.initialize_from_model_params(model_filename)
    return model
