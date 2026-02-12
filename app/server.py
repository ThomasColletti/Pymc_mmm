"""
FastAPI server for a fitted PyMC-Marketing MMM saved as NetCDF (.nc).

Deployment pattern (recommended for Bayesian models):
1) Recreate the MMM model structure in code (same columns, transforms, seasonality, priors).
2) Load the fitted posterior from a .nc file.
3) Attach it to `mmm.fit_result`.
4) Serve predictions / budget optimization without refitting.
"""

from __future__ import annotations

import os
from typing import List, Optional, Dict, Any
import arviz as az
from fastapi import FastAPI, HTTPException

DATE_COLUMN = "date"
CHANNEL_COLUMNS = ["search_spend", "social_spend", "ctv_spend", "tv_spend"]
CONTROL_COLUMNS = ["prime_day_promo", "mothers_day_promo"]

POSTERIOR_PATH = os.getenv("MMM_POSTERIOR_PATH", "artifacts/mmm_posterior.nc")

def build_mmm() -> Any:
    from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

    # --- Transforms (must match training) ---
    adstock = GeometricAdstock(l_max=8, normalize=True)
    saturation = LogisticSaturation()

    # --- Priors / model config ---
    # IMPORTANT: In your environment, `Prior` class import might differ.
    # Since you trained successfully, paste your *working* my_model_config here.
    #
    # If your version expects dict priors, you can use something like:
    my_model_config = {
        "intercept": {"dist": "Normal", "kwargs": {"mu": 500, "sigma": 150}},
        "sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 6}},
        "control_coef": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 0.05}},
        "seasonality": {"dist": "Laplace", "kwargs": {"mu": 0, "b": 0.2}},
    }

    mmm = MMM(
        date_column=DATE_COLUMN,
        channel_columns=CHANNEL_COLUMNS,
        control_columns=CONTROL_COLUMNS,
        yearly_seasonality=2,
        adstock=adstock,
        saturation=saturation,
        model_config=my_model_config,
    )
    return mmm


def load_posterior(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Posterior file not found at: {path}. "
            f"Set MMM_POSTERIOR_PATH or place file at that location."
        )
    idata = az.from_netcdf(path)
    return idata


# Initialize at import time (simple deployment)
mmm = build_mmm()
fit_result = load_posterior(POSTERIOR_PATH)
# attach posterior so mmm methods can use it
mmm.fit_result = fit_result

class ScenarioRow(BaseModel):
    date: str  # ISO date string, e.g. "2026-03-02"
    search_spend: float = 0.0
    social_spend: float = 0.0
    ctv_spend: float = 0.0
    tv_spend: float = 0.0

    price_index: float = 1.0
    prime_day_promo: int = 0
    mothers_day_promo: int = 0


class PredictRequest(BaseModel):
    rows: List[ScenarioRow]
    draws: int = Field(default=500, ge=50, le=5000)
    hdi_prob: float = Field(default=0.9, ge=0.5, le=0.99)
    combined: bool = True  # combine chain/draw dimensions if available


class PredictResponse(BaseModel):
    mean: List[float]
    hdi_lower: List[float]
    hdi_upper: List[float]
    variable_name: str



@app.get("/health")
def health():
    return {
        "status": "ok",
        "posterior_path": POSTERIOR_PATH,
        "channels": CHANNEL_COLUMNS,
        "controls": CONTROL_COLUMNS,
        "date_column": DATE_COLUMN,
    }



@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = _to_dataframe(req.rows)

    try:
        # Different pymc-marketing versions accept different signatures.
        # Common patterns:
        #  - mmm.sample_posterior_predictive(df, extend_idata=False, combined=True)
        #  - mmm.sample_posterior_predictive(X=df, extend_idata=False, combined=True)
        try:
            ppc = mmm.sample_posterior_predictive(
                df, extend_idata=False, combined=req.combined
            )
        except TypeError:
            ppc = mmm.sample_posterior_predictive(
                X=df, extend_idata=False, combined=req.combined
            )

        y_name, y_draws = _extract_ppc_y(ppc)

        # thin draws if needed
        if y_draws.shape[0] > req.draws:
            idx = np.linspace(0, y_draws.shape[0] - 1, req.draws).astype(int)
            y_draws = y_draws[idx]

        mean = y_draws.mean(axis=0)
        hdi = az.hdi(y_draws, hdi_prob=req.hdi_prob)  # (time, 2)

        return PredictResponse(
            mean=mean.tolist(),
            hdi_lower=hdi[:, 0].tolist(),
            hdi_upper=hdi[:, 1].tolist(),
            variable_name=y_name,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))