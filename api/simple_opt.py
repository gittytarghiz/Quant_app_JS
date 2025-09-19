from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

from data_management.monolith_loader import get_downloaded_series
from .utils import df_to_records, series_to_records, compute_metrics


router = APIRouter(prefix="/opt", tags=["opt"])


class BaseReq(BaseModel):
    tickers: list[str]
    start: str
    end: str
    dtype: str = Field(default="close")
    interval: str = Field(default="1d")
    leverage: float = 1.0


def _load_rets(tickers: list[str], start: str, end: str, dtype: str, interval: str) -> pd.DataFrame:
    prices = get_downloaded_series(tickers, start, end, dtype=dtype, interval=interval).dropna()
    rets = prices.pct_change().dropna()
    if rets.empty:
        raise ValueError("No returns available for given inputs.")
    return rets


def _pack(weights: np.ndarray, tickers: list[str], rets: pd.DataFrame) -> dict[str, Any]:
    w = np.asarray(weights, dtype=float).ravel()
    w = w / max(w.sum(), 1e-12)
    pnl = (rets @ w).astype(float)
    weights_df = pd.DataFrame([w], columns=[t.upper() for t in tickers], index=[rets.index[0]])
    weights_df.index.name = "date"
    return {
        "weights": df_to_records(weights_df),
        "pnl": series_to_records(pnl, value_name="pnl"),
        "details": {"metrics": compute_metrics(pnl, weights_df)},
    }


@router.post("/eqw-simple")
def eqw_simple(req: BaseReq) -> dict[str, Any]:
    rets = _load_rets([t.upper() for t in req.tickers], req.start, req.end, req.dtype, req.interval)
    n = rets.shape[1]
    w = float(req.leverage) * np.full(n, 1.0 / n)
    return _pack(w, list(rets.columns), rets)


@router.post("/minvar-simple")
def minvar_simple(req: BaseReq) -> dict[str, Any]:
    rets = _load_rets([t.upper() for t in req.tickers], req.start, req.end, req.dtype, req.interval)
    v = np.var(rets.values, axis=0).astype(float)
    v = np.where(v <= 0, 1e-6, v)
    w = float(req.leverage) * (1.0 / v)
    return _pack(w, list(rets.columns), rets)


@router.post("/rp-simple")
def risk_parity_simple(req: BaseReq) -> dict[str, Any]:
    rets = _load_rets([t.upper() for t in req.tickers], req.start, req.end, req.dtype, req.interval)
    s = np.std(rets.values, axis=0).astype(float)
    s = np.where(s <= 0, 1e-6, s)
    w = float(req.leverage) * (1.0 / s)
    return _pack(w, list(rets.columns), rets)


@router.post("/mvo-simple")
def mvo_simple(req: BaseReq, risk_aversion: float = 1.0) -> dict[str, Any]:
    rets = _load_rets([t.upper() for t in req.tickers], req.start, req.end, req.dtype, req.interval)
    mu = np.mean(rets.values, axis=0).astype(float)
    S = np.cov(rets.values, rowvar=False).astype(float)
    # regularize
    d = np.diag_indices_from(S)
    S[d] += 1e-6
    try:
        inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(S)
    w = inv @ mu
    w = float(req.leverage) * w
    return _pack(w, list(rets.columns), rets)

