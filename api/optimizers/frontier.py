from fastapi import APIRouter
from typing import Any, Optional
from pydantic import Field
import numpy as np

from api.core import OptimizationRequest
from api.core.utils import format_pnl, format_weights, normalize_details
from portfolio_optimization.walkforward_mvo_target import walkforward_mvo_target_return
from data_management.monolith_loader import get_downloaded_series

router = APIRouter(prefix="/opt", tags=["opt"])


class FrontierRequest(OptimizationRequest):
    n_points: int = Field(15, ge=3, le=60)
    target_min: Optional[float] = None
    target_max: Optional[float] = None
    cov_shrinkage: float = Field(0.0, ge=0.0, le=1.0)
    cov_estimator: Optional[str] = Field(None, description="sample|diag|lw")


@router.post("/frontier")
async def frontier(req: FrontierRequest) -> dict[str, Any]:
    """Efficient Frontier generator — return targets with PnL, Weights, and Details per point"""
    rets = get_downloaded_series(req.tickers, req.start, req.end, dtype=req.dtype, interval=req.interval).pct_change().dropna()
    if rets.empty:
        raise ValueError("No returns available")

    mu = rets.mean().values
    tmin = req.target_min or float(mu.min()) * 0.5
    tmax = req.target_max or float(mu.max()) * 1.5
    if tmax <= tmin:
        tmax = tmin + abs(tmin) + 1e-6

    targets = np.linspace(tmin, tmax, req.n_points)
    results = []

    for t in targets:
        res = walkforward_mvo_target_return(
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
            target_return=float(t),
            cov_shrinkage=req.cov_shrinkage,
            cov_estimator=req.cov_estimator,
        )
        results.append({
            "target": float(t),
            "pnl": format_pnl(res.get("pnl")),
            "weights": format_weights(res.get("weights")),
            "details": normalize_details(res.get("details")),
        })

    return {"frontier": results}
