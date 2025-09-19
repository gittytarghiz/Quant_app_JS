from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field
import numpy as np

from portfolio_optimization.walkforward_mvo_target import (
    walkforward_mvo_target_return as _wf_mvo_tr,
)
from .utils import compute_metrics


router = APIRouter(prefix="/opt", tags=["opt"])


class FrontierRequest(BaseModel):
    tickers: list[str]
    start: str
    end: str
    dtype: str = Field(default="close")
    interval: str = Field(default="1d")
    rebalance: str = Field(default="monthly")
    costs: Optional[dict[str, float]] = None
    min_weight: float = 0.0
    max_weight: float = 1.0
    min_obs: int = 60
    leverage: float = 1.0

    # sweep config
    n_points: int = Field(default=15, ge=3, le=60)
    target_min: Optional[float] = None
    target_max: Optional[float] = None
    cov_shrinkage: float = Field(default=0.0, ge=0.0, le=1.0)
    cov_estimator: Optional[str] = Field(default=None, description="sample|diag|lw")


@router.post("/frontier")
def frontier(req: FrontierRequest) -> dict[str, Any]:
    # Establish target range using simple whole-period mu if not provided
    import pandas as pd
    from data_management.monolith_loader import get_downloaded_series
    prices = get_downloaded_series(req.tickers, req.start, req.end, dtype=req.dtype, interval=req.interval).dropna()
    rets = prices.pct_change().dropna()
    if rets.empty:
        raise ValueError("No returns available.")
    mu = rets.mean().values
    tmin = req.target_min if req.target_min is not None else float(mu.min()) * 0.5
    tmax = req.target_max if req.target_max is not None else float(mu.max()) * 1.5
    if tmax <= tmin:
        tmax = tmin + abs(tmin) + 1e-6
    targets = [tmin + (tmax - tmin) * i / (max(req.n_points - 1, 1)) for i in range(req.n_points)]

    points: list[dict[str, Any]] = []
    best_idx = None
    best_sharpe = -1e9
    for i, tgt in enumerate(targets):
        res = _wf_mvo_tr(
            tickers=req.tickers,
            start=req.start,
            end=req.end,
            dtype=req.dtype,
            interval=req.interval,
            rebalance=req.rebalance,
            costs=req.costs,
            min_weight=req.min_weight,
            max_weight=req.max_weight,
            min_obs=req.min_obs,
            leverage=req.leverage,
            target_return=float(tgt),
            cov_shrinkage=req.cov_shrinkage,
            cov_estimator=req.cov_estimator,
            de_maxiter=20,
            de_popsize=15,
            de_tol=0.02,
            de_seed=42,
            de_workers=1,
            de_polish=False,
        )
        m = compute_metrics(res.get("pnl"), res.get("weights"))
        pt = {
            "idx": i,
            "target": float(tgt),
            "metrics": m,
        }
        points.append(pt)
        sh = float(m.get("sharpe") or 0.0)
        if sh > best_sharpe:
            best_sharpe = sh
            best_idx = i

    return {"targets": targets, "points": points, "best_idx": best_idx}

