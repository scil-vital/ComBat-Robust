# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from statsmodels.gam.api import GLMGam, BSplines

from clinical_combat.harmonization.Combat import Combat


class CombatGam(Combat):
    """
    ComBat GAM: Harmonize the moving site to the reference site.
    The covariate regression is estimated with a GAM (splines), in the spirit of
    neuroHarmonize ComBat-GAM, while site effects are still handled by gamma/delta.

    Notes
    -----
    - We do not include SITE in the covariate regression here because site effects
      are modeled by gamma/delta.
    - We fit one GAM per bundle (feature), then compute sigma from pooled residuals.
    """

    def __init__(
            self,
            bundle_names=None,
            model_params=None,
            ignore_sex_covariate=False,
            ignore_handedness_covariate=False,
            use_empirical_bayes=True,
            limit_age_range=False,
            degree=1,
            regul_ref=0,
            regul_mov=0,
            smooth_terms=("age",),
            smooth_term_bounds=(None, None),
            df_spline=10,
            spline_degree=3,
            gam_params=None,
            gam_linear_cols=None,
            gam_smooth_cols=None,
            gam_bspline_cfg=None,
            sigma=None,
            gamma_ref=None,
            delta_ref=None,
            gamma_mov=None,
            delta_mov=None,
            ):
        """
        regul_ref: float
            Regularization parameter for reference site (if used by parent).
        regul_mov: float
            Regularization parameter for moving site (if used by parent).

        smooth_terms: tuple[str]
            Covariate names to smooth (e.g. ("age",)).
        smooth_term_bounds: tuple[float|None, float|None]
            Lower and upper bounds for knots for 1D smoothing (optional).
        df_spline: int
            Number of spline basis functions for each smooth term.
        spline_degree: int
            B-spline degree.

        gam_params: Array
            Fitted parameters for the GAM, one row per bundle.
        gam_linear_cols: list[str]
            Names of the linear design columns used in the GAM (including "x0").
        gam_smooth_cols: list[str]
            Raw covariate names used as smooth terms.
        gam_bspline_cfg: dict
            Persisted configuration to rebuild splines at apply time.

        sigma: Array
            Standard deviation per bundle, computed from pooled residuals.
        gamma_ref: Array
            Additive bias of the reference site.
        delta_ref: Array
            Multiplicative bias of the reference site.
        gamma_mov: Array
            Additive bias of the moving site.
        delta_mov: Array
            Multiplicative bias of the moving site.
        """
        super().__init__(
            bundle_names=bundle_names,
            model_params=model_params,
            ignore_sex_covariate=ignore_sex_covariate,
            ignore_handedness_covariate=ignore_handedness_covariate,
            use_empirical_bayes=use_empirical_bayes,
            limit_age_range=limit_age_range,
            degree=degree,
        )

        self.regul_ref = regul_ref
        self.regul_mov = regul_mov
        self.smooth_terms = list(smooth_terms) if smooth_terms is not None else []
        self.smooth_term_bounds = smooth_term_bounds
        self.df_spline = int(df_spline)
        self.spline_degree = int(spline_degree)

        self.gam_params = gam_params
        self.gam_linear_cols = gam_linear_cols
        self.gam_smooth_cols = gam_smooth_cols
        self.gam_bspline_cfg = gam_bspline_cfg
        self.sigma = sigma
        self.gamma_ref = gamma_ref
        self.delta_ref = delta_ref
        self.gamma_mov = gamma_mov
        self.delta_mov = delta_mov

        self._bsplines_constructor = None

    def _build_covars_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build a covariates DataFrame from input data.

        data: DataFrame
            Must contain at least "age". Optionally contains "sex" and "handedness"
            depending on ignore flags.

        Returns
        -------
        covars: DataFrame
            Columns are raw covariates (not polynomialized).
        """
        cols = {}

        if "age" not in data.columns:
            raise ValueError("CombatGam: missing column 'age' in data.")
        cols["age"] = data["age"].astype(float).to_numpy()

        if not self.ignore_sex_covariate:
            if "sex" not in data.columns:
                raise ValueError("CombatGam: missing column 'sex' in data.")
            cols["sex"] = data["sex"].astype(float).to_numpy()

        if not self.ignore_handedness_covariate:
            if "handedness" not in data.columns:
                raise ValueError("CombatGam: missing column 'handedness' in data.")
            cols["handedness"] = data["handedness"].astype(float).to_numpy()

        return pd.DataFrame(cols)

    def _make_bspline(self, X_spline: np.ndarray) -> BSplines:
        """
        Create a BSplines smoother for the given smooth covariates matrix.

        X_spline: array
            Shape (n_samples, n_smooth_terms).

        Returns
        -------
        bs: BSplines
            Statsmodels BSplines object.
        """
        if X_spline.ndim != 2:
            raise ValueError("CombatGam: X_spline must be 2D.")
        if X_spline.shape[1] == 1:
            return BSplines(
                X_spline,
                df=self.df_spline,
                degree=self.spline_degree,
                knot_kwds=[{
                    "lower_bound": self.smooth_term_bounds[0],
                    "upper_bound": self.smooth_term_bounds[1],
                }],
            )
        return BSplines(
            X_spline,
            df=[self.df_spline] * X_spline.shape[1],
            degree=[self.spline_degree] * X_spline.shape[1],
        )

    def _fit_gam_params(self, covars: pd.DataFrame, y: np.ndarray):
        """
        Fit a GAM and return fitted parameters.

        covars: DataFrame
            Raw covariates for a single bundle.
        y: array
            Observations for that bundle.

        Returns
        -------
        params: array
            Concatenated parameters [linear..., spline...].
        linear_cols: list[str]
            Names of linear design columns used by the GAM (includes "x0").
        smooth_cols: list[str]
            Raw covariate names used for smoothing.
        bs: BSplines|None
            Fitted smoother (None if smoothing disabled).
        """
        smooth_cols = [c for c in self.smooth_terms if c in covars.columns]
        linear_raw = [c for c in covars.columns if c not in smooth_cols]

        df_gam = {}
        formula = "y ~ "

        df_gam["x0"] = np.ones(len(covars), dtype=float)
        formula += "x0 + "

        for c in linear_raw:
            name = f"c_{c}"
            df_gam[name] = covars[c].astype(float).to_numpy()
            formula += f"{name} + "

        formula = formula[:-3] + " - 1"
        df_gam = pd.DataFrame(df_gam)
        df_gam["y"] = y.astype(float)

        if len(smooth_cols) > 0:
            X_spline = covars[smooth_cols].astype(float).to_numpy()
            bs = self._make_bspline(X_spline)

            alpha_init = np.array([1.0] * len(smooth_cols), dtype=float)
            gam_bs = GLMGam.from_formula(formula, data=df_gam, smoother=bs, alpha=alpha_init)
            _ = gam_bs.fit()
            gam_bs.alpha = gam_bs.select_penweight_kfold()[0]
            res = gam_bs.fit()
            params = np.asarray(res.params, dtype=float)
        else:
            bs = None
            X_lin = df_gam.drop(columns=["y"]).to_numpy()
            params, *_ = np.linalg.lstsq(X_lin, y.astype(float), rcond=None)
            params = np.asarray(params, dtype=float)

        linear_cols = ["x0"] + [f"c_{c}" for c in linear_raw]
        return params, linear_cols, smooth_cols, bs

    def _predict_gam_mean(self, covars: pd.DataFrame, params: np.ndarray, bs) -> np.ndarray:
        """
        Predict GAM mean mu(covars) for a single bundle.

        covars: DataFrame
            Raw covariates for that bundle.
        params: array
            Stored GAM parameters for that bundle.
        bs: BSplines|None
            Smoother. If None and smoothing is enabled, we will rebuild it.

        Returns
        -------
        mu: array
            Predicted mean for each row.
        """
        if self.gam_linear_cols is None:
            raise AssertionError("CombatGam: gam_linear_cols is not initialized.")

        do_smooth = bool(self.gam_bspline_cfg and self.gam_bspline_cfg.get("perform_smoothing", False))
        if bs is None and do_smooth:
            X_spline = covars[self.gam_smooth_cols].astype(float).to_numpy()
            bs = self._make_bspline(X_spline)

        X_lin_df = pd.DataFrame(index=covars.index)
        X_lin_df["x0"] = 1.0
        for col in self.gam_linear_cols:
            if col == "x0":
                continue
            raw = col.replace("c_", "")
            X_lin_df[col] = covars[raw].astype(float).to_numpy()

        X_lin = X_lin_df[self.gam_linear_cols].to_numpy()
        n_lin = X_lin.shape[1]
        mu = X_lin @ params[:n_lin]

        if bs is not None:
            X_spline = covars[self.gam_smooth_cols].astype(float).to_numpy()
            bs_basis = bs.transform(X_spline)
            mu = mu + bs_basis @ params[n_lin:]

        return np.asarray(mu, dtype=float)

    def standardize_moving_data(self, covars_df_list, y_list):
        """
        Standardize the data (Y). ComBat standardizes the data with
        the estimated GAM covariate effect and standard deviation.

        covars_df_list: list[DataFrame]
            One covariates DataFrame per bundle.
        y_list: list[array]
            One observation vector per bundle.

        Returns
        -------
        s_y: list[array]
            Standardized values per bundle.
        """
        s_y = []
        for i in range(len(y_list)):
            bs = None
            if isinstance(self._bsplines_constructor, (list, tuple)):
                bs = self._bsplines_constructor[i]
            else:
                bs = self._bsplines_constructor
            mu = self._predict_gam_mean(covars_df_list[i], self.gam_params[i], bs)
            s_y.append((np.asarray(y_list[i], dtype=float) - mu) / self.sigma[i])
        return s_y

    def remove_covariate_effect(self, covars_df_list, y_list):
        """
        Remove GAM covariate effect.

        covars_df_list: list[DataFrame]
            One covariates DataFrame per bundle.
        y_list: list[array]
            One observation vector per bundle.

        Returns
        -------
        resid: list[array]
            Residuals (Y - mu) per bundle.
        """
        out = []
        for i in range(len(y_list)):
            bs = None
            if isinstance(self._bsplines_constructor, (list, tuple)):
                bs = self._bsplines_constructor[i]
            else:
                bs = self._bsplines_constructor
            mu = self._predict_gam_mean(covars_df_list[i], self.gam_params[i], bs)
            out.append(np.asarray(y_list[i], dtype=float) - mu)
        return out

    def fit(self, ref_data, mov_data):
        """
        Combat GAM fit. GAM regression parameters are fitted using all data.

        ref_data: DataFrame
            Data of the reference site.
        mov_data: DataFrame
            Data of the moving site.

        Notes
        -----
        - We fit a GAM per bundle on pooled data, then compute sigma from pooled residuals.
        - Then we estimate gamma/delta on standardized data, optionally with EB.
        """
        ref_data, mov_data = self.prepare_data(ref_data, mov_data)

        all_data = pd.concat([ref_data, mov_data], axis=0)

        _, y_mov = self.get_design_matrices(mov_data)
        _, y_ref = self.get_design_matrices(ref_data)
        _, y_all = self.get_design_matrices(all_data)

        cov_all_map = {b: self._build_covars_df(all_data.query("bundle == @b")) for b in self.bundle_names}
        cov_mov_map = {b: self._build_covars_df(mov_data.query("bundle == @b")) for b in self.bundle_names}
        cov_ref_map = {b: self._build_covars_df(ref_data.query("bundle == @b")) for b in self.bundle_names}

        n_bundles = len(y_all)

        gam_params = []
        bs_constructors = []
        linear_cols_global = None
        smooth_cols_global = None

        for i in range(n_bundles):
            bname = self.bundle_names[i]
            y = np.asarray(y_all[i], dtype=float)
            cov_bundle = cov_all_map[bname]

            params, linear_cols, smooth_cols, bs = self._fit_gam_params(cov_bundle, y)
            gam_params.append(params)
            bs_constructors.append(bs)

            if linear_cols_global is None:
                linear_cols_global = linear_cols
                smooth_cols_global = smooth_cols
            else:
                if linear_cols != linear_cols_global or smooth_cols != smooth_cols_global:
                    raise AssertionError("CombatGam: inconsistent GAM design columns across bundles.")

        self.gam_params = np.vstack([p.reshape(1, -1) for p in gam_params])
        self.gam_linear_cols = linear_cols_global
        self.gam_smooth_cols = smooth_cols_global
        self.gam_bspline_cfg = {
            "perform_smoothing": len(self.gam_smooth_cols) > 0,
            "smooth_terms": list(self.smooth_terms),
            "smooth_cols": list(self.gam_smooth_cols),
            "df_spline": self.df_spline,
            "spline_degree": self.spline_degree,
            "smooth_term_bounds": tuple(self.smooth_term_bounds),
        }
        if len(self.gam_smooth_cols) > 0:
            self._bsplines_constructor = bs_constructors
        else:
            self._bsplines_constructor = [None] * n_bundles

        sigma = []
        for i in range(n_bundles):
            bname = self.bundle_names[i]
            bs = self._bsplines_constructor[i] if self._bsplines_constructor else None
            mu_all = self._predict_gam_mean(cov_all_map[bname], self.gam_params[i], bs)
            resid = np.asarray(y_all[i], dtype=float) - mu_all
            sigma.append(np.std(resid, ddof=1))
        self.sigma = np.asarray(sigma, dtype=float)

        cov_mov_list = [cov_mov_map[b] for b in self.bundle_names]
        cov_ref_list = [cov_ref_map[b] for b in self.bundle_names]

        z_mov = self.standardize_moving_data(cov_mov_list, y_mov)
        self.gamma_mov = np.array([np.mean(x) for x in z_mov], dtype=float)
        self.delta_mov = np.array([np.std(x, ddof=1) for x in z_mov], dtype=float)

        if getattr(self, "robust", None) == "FLIP":
            self.gamma_mov = np.array([np.median(x) for x in z_mov], dtype=float)
            self.delta_mov = np.array(
                [np.mean(x[x <= np.median(x)]) - np.median(x) for x in z_mov],
                dtype=float
            )

        if self.use_empirical_bayes:
            self.gamma_mov, self.delta_mov = Combat.emperical_bayes_estimate(
                z_mov, self.gamma_mov, self.delta_mov ** 2
            )
        self.gamma_mov *= self.sigma

        z_ref = self.standardize_moving_data(cov_ref_list, y_ref)
        self.gamma_ref = np.array([np.mean(x) for x in z_ref], dtype=float)
        self.delta_ref = np.array([np.std(x, ddof=1) for x in z_ref], dtype=float)
        self.gamma_ref *= self.sigma

        self.set_model_fit_params(ref_data, mov_data)
        return

    def apply(self, data):
        """
        Apply the harmonization fitted model to data.

        data: DataFrame
            Dataframe representing the data to harmonize.

        Returns
        -------
        harm_y: list[array]
            Harmonized data values per bundle.
        """
        if (
            self.gam_params is None
            or self.sigma is None
            or self.gamma_ref is None
            or self.delta_ref is None
            or self.gamma_mov is None
            or self.delta_mov is None
        ):
            raise AssertionError("Model parameters are not fitted.")

        cov_map = {b: self._build_covars_df(data.query("bundle == @b")) for b in self.bundle_names}
        _, Y = self.get_design_matrices(data)

        harm_y = []
        for i in range(len(Y)):
            bname = self.bundle_names[i]
            cov_bundle = cov_map[bname]
            bs = self._bsplines_constructor[i] if self._bsplines_constructor else None
            mu = self._predict_gam_mean(cov_bundle, self.gam_params[i], bs)

            harm_y.append(
                (self.delta_ref[i] / self.delta_mov[i])
                * (np.asarray(Y[i], dtype=float) - mu - self.gamma_mov[i])
                + self.gamma_ref[i]
                + mu
            )
        return harm_y

    def predict(self, ages, bundle, moving_site=True, sex=0.5, handedness=0.5):
        """
        Use the model to predict the GAM mean (covariate effect only).

        ages: array
            Age used to do the prediction.
        bundle: str
            Bundle to use.

        Returns
        -------
        mu: array
            Model-predicted mean for the input covariates.
        """
        if self.gam_params is None:
            raise AssertionError("CombatGam: model is not fitted.")

        idx = list(self.bundle_names).index(bundle)

        d = {"age": np.asarray(ages, dtype=float)}
        if not self.ignore_sex_covariate:
            d["sex"] = np.ones(len(ages), dtype=float) * float(sex)
        if not self.ignore_handedness_covariate:
            d["handedness"] = np.ones(len(ages), dtype=float) * float(handedness)

        cov = pd.DataFrame(d)

        do_smooth = bool(self.gam_bspline_cfg and self.gam_bspline_cfg.get("perform_smoothing", False))
        bs = None
        if do_smooth:
            if isinstance(self._bsplines_constructor, (list, tuple)) and len(self._bsplines_constructor) > idx:
                bs = self._bsplines_constructor[idx]
            if bs is None:
                X_spline = cov[self.gam_smooth_cols].astype(float).to_numpy()
                bs = self._make_bspline(X_spline)

        mu = self._predict_gam_mean(cov, self.gam_params[idx], bs)
        return mu

    def set_model_fit_params(self, ref_data, mov_data):
        """
        Set the model parameter given the input data used for the fit.

        ref_data: DataFrame
            Data of the reference site.
        mov_data: DataFrame
            Data of the moving site.
        """
        super().set_model_fit_params(ref_data, mov_data)
        self.model_params["regul_ref"] = self.regul_ref
        self.model_params["regul_mov"] = self.regul_mov
        self.model_params["name"] = "gam"
        self.model_params["gam"] = {
            "smooth_terms": list(self.smooth_terms),
            "smooth_term_bounds": tuple(self.smooth_term_bounds),
            "df_spline": self.df_spline,
            "spline_degree": self.spline_degree,
            "linear_cols": list(self.gam_linear_cols) if self.gam_linear_cols is not None else None,
            "smooth_cols": list(self.gam_smooth_cols) if self.gam_smooth_cols is not None else None,
            "gam_bspline_cfg": dict(self.gam_bspline_cfg) if self.gam_bspline_cfg is not None else None,
        }

    def save_model(self, model_filename):
        """
        Save the harmonization model to file.

        model_filename: str
            Model filename.

        Notes
        -----
        - The CSV stores bundle-wise params plus a GAM params block.
        - A sidecar pickle stores GAM column config safely.
        """
        if self.gam_params is None:
            raise AssertionError("CombatGam: gam_params missing.")
        if self.sigma is None:
            raise AssertionError("CombatGam: sigma missing.")

        p = self.gam_params.shape[1]

        base = np.hstack(
            [
                np.asarray(self.bundle_names).reshape(-1, 1),
                self.sigma.reshape(-1, 1),
                self.gamma_ref.reshape(-1, 1),
                self.delta_ref.reshape(-1, 1),
                self.gamma_mov.reshape(-1, 1),
                self.delta_mov.reshape(-1, 1),
            ]
        )

        params = np.hstack([base, self.gam_params]).transpose()

        labels = ["bundle_names", "sigma", "ref_gamma", "ref_delta", "mov_gamma", "mov_delta"]
        labels += [f"gam_param_{k}" for k in range(p)]
        labels = np.array(labels).reshape(-1, 1)

        out = np.hstack([labels, params.astype(object)])
        header = str(self.model_params)
        np.savetxt(model_filename, out, delimiter=",", fmt="%s", header=header)

        cfg = {
            "gam_linear_cols": self.gam_linear_cols,
            "gam_smooth_cols": self.gam_smooth_cols,
            "gam_bspline_cfg": self.gam_bspline_cfg,
            "bsplines_constructor": self._bsplines_constructor,
        }
        cfg_filename = model_filename + ".gam_cfg.pkl"
        import pickle
        with open(cfg_filename, "wb") as f:
            pickle.dump(cfg, f)

    def initialize_from_model_params(self, model_filename):
        """
        Initialize the object from a model file.

        model_filename: str
            Model filename.
        """
        super().initialize_from_model_params(model_filename)

        self.regul_ref = self.model_params.get("regul_ref", self.regul_ref)
        self.regul_mov = self.model_params.get("regul_mov", self.regul_mov)
        gam_cfg = self.model_params.get("gam", {})
        self.df_spline = gam_cfg.get("df_spline", self.df_spline)
        self.spline_degree = gam_cfg.get("spline_degree", self.spline_degree)
        self.smooth_terms = list(gam_cfg.get("smooth_terms", self.smooth_terms))
        self.smooth_term_bounds = tuple(gam_cfg.get("smooth_term_bounds", self.smooth_term_bounds))

        params = np.loadtxt(model_filename, delimiter=",", dtype=str, skiprows=1)

        self.bundle_names = params[0, 1:].astype(str)
        self.sigma = params[1, 1:].astype("float64").transpose()
        self.gamma_ref = params[2, 1:].astype("float64").transpose()
        self.delta_ref = params[3, 1:].astype("float64").transpose()
        self.gamma_mov = params[4, 1:].astype("float64").transpose()
        self.delta_mov = params[5, 1:].astype("float64").transpose()

        gam_block = params[6:, 1:].astype("float64").transpose()
        self.gam_params = gam_block

        cfg_filename = model_filename + ".gam_cfg.pkl"
        import pickle
        with open(cfg_filename, "rb") as f:
            cfg = pickle.load(f)

        self.gam_linear_cols = cfg["gam_linear_cols"]
        self.gam_smooth_cols = cfg["gam_smooth_cols"]
        self.gam_bspline_cfg = cfg["gam_bspline_cfg"]

        self._bsplines_constructor = cfg.get("bsplines_constructor")
        if self._bsplines_constructor is None and self.gam_bspline_cfg and self.gam_bspline_cfg.get("perform_smoothing", False):
            self._bsplines_constructor = [None] * len(self.bundle_names)
