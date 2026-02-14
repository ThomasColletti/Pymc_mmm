# mmm_inference.py
from __future__ import annotations

import numpy as np
import pandas as pd
import arviz as az

from pymc_marketing.mmm import MMM
from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation
from pymc_extras.prior import Prior


CHANNEL_COLS = ["search_spend", "social_spend", "ctv_spend", "tv_spend"]
CONTROL_COLS = ["prime_day_promo", "mothers_day_promo", "t"]
DATE_COL = "week"
TARGET_COL = "sales"


def build_mmm_from_training_config(prior_sigma: np.ndarray) -> MMM:
    """Rebuild the MMM object exactly like training (structure must match)."""
    model_config = {
        "intercept": Prior("Normal", mu=500, sigma=150),
        "channel_coef": Prior("HalfNormal", sigma=prior_sigma),
        "control_coef": Prior("Normal", mu=0, sigma=0.05),
        "seasonality": Prior("Laplace", mu=0, b=0.2),
        "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=6)),
    }

    return MMM(
        model_config=model_config,
        sampler_config={"progressbar": False},
        date_column=DATE_COL,
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        channel_columns=CHANNEL_COLS,
        control_columns=CONTROL_COLS,
        yearly_seasonality=2,
    )


def _compute_prior_sigma_from_training_df(train_df: pd.DataFrame) -> np.ndarray:
    """Matches your training logic: sigma proportional to spend share."""
    total_spend_per_channel = train_df[CHANNEL_COLS].sum(axis=0)
    spend_share = total_spend_per_channel / total_spend_per_channel.sum()
    n_channels = len(CHANNEL_COLS)
    prior_sigma = n_channels * spend_share.to_numpy()
    return prior_sigma


def _clean_and_format_new_df(
    df_new: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensures new df has same required columns and types.
    Also builds 't' aligned to training timeline (continues the trend index).
    """
    df = df_new.copy()

    # Required columns
    required = [DATE_COL] + CHANNEL_COLS + ["prime_day_promo", "mothers_day_promo"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"New data is missing required columns: {missing}")

    # Parse/sort dates
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Numeric coercion + fill
    for c in CHANNEL_COLS + ["prime_day_promo", "mothers_day_promo"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Build trend feature t
    # Training: t = 0..(len(train_df)-1)
    # Inference: continue from last training t + 1, aligned by week ordering
    train_df = train_df.copy()
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    train_df = train_df.sort_values(DATE_COL).reset_index(drop=True)
    last_t = len(train_df) - 1
    df["t"] = np.arange(last_t + 1, last_t + 1 + len(df), dtype=float)

    return df


def predict_sales(
    model_nc_path: str,
    new_df: pd.DataFrame,
    training_csv_path: str,
    hdi_prob: float = 0.9,
) -> pd.DataFrame:
    """
    Loads posterior from .nc and returns posterior predictive sales summary for new data.

    Returns a dataframe:
      week, sales_pred_mean, sales_pred_hdi_low, sales_pred_hdi_high
    """
    # 1) Load posterior
    idata = az.from_netcdf(model_nc_path)

    # 2) Load training data (needed to compute prior_sigma and align trend t)
    train_df = pd.read_csv(training_csv_path)
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    train_df = train_df.sort_values(DATE_COL).reset_index(drop=True)

    prior_sigma = _compute_prior_sigma_from_training_df(train_df)

    # 3) Build MMM object with same config as training
    mmm = build_mmm_from_training_config(prior_sigma=prior_sigma)

    # 4) Clean/format new data into X_new
    df = _clean_and_format_new_df(new_df, train_df=train_df)

    X_new = df[[DATE_COL] + CHANNEL_COLS + ["prime_day_promo", "mothers_day_promo", "t"]]

    # 5) Posterior predictive
    # pymc-marketing MMM supports passing idata + new data for prediction
    # Depending on version, method name may be `sample_posterior_predictive` or `predict`.
    try:
        ppc_idata = mmm.sample_posterior_predictive(
            X=X_new,
            idata=idata,
        )
        # common variable name is "sales" but can vary by version
        # we'll try a few options robustly
        if hasattr(ppc_idata, "posterior_predictive") and TARGET_COL in ppc_idata.posterior_predictive:
            y_pp = ppc_idata.posterior_predictive[TARGET_COL]
        else:
            # fallback: grab the first var
            y_pp = next(iter(ppc_idata.posterior_predictive.data_vars.values()))
    except AttributeError:
        # Fallback if your version uses `predict` returning samples or mean
        ppc_idata = mmm.predict(X=X_new, idata=idata)
        # try to interpret
        if isinstance(ppc_idata, az.InferenceData) and TARGET_COL in ppc_idata.posterior_predictive:
            y_pp = ppc_idata.posterior_predictive[TARGET_COL]
        else:
            raise RuntimeError(
                "Could not run posterior predictive. Your pymc-marketing MMM version "
                "may use different method names. Tell me your pymc-marketing version."
            )

    # y_pp dims typically: (chain, draw, time)
    y_samples = y_pp.stack(sample=("chain", "draw")).transpose("sample", ...).values  # (S, T)

    mean = y_samples.mean(axis=0)
    hdi = az.hdi(y_samples, hdi_prob=hdi_prob)  # (T, 2)
    low, high = hdi[:, 0], hdi[:, 1]

    out = pd.DataFrame(
        {
            "week": df[DATE_COL].values,
            "sales_pred_mean": mean,
            f"sales_pred_hdi_low_{int(hdi_prob*100)}": low,
            f"sales_pred_hdi_high_{int(hdi_prob*100)}": high,
        }
    )

    return out


### testing
