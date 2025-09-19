from fastapi import APIRouter
from typing import Any, Optional
from pydantic import Field
from api.core import OptimizationRequest, OptimizationResponse
from api.core.utils import format_weights, format_pnl, normalize_details
from portfolio_optimization.walkforward_mvo_target import walkforward_mvo_target_return
import numpy as np
from data_management.monolith_loader import get_downloaded_series

router = APIRouter(prefix="/opt", tags=["opt"])

class FrontierRequest(OptimizationRequest):
    n_points: int = Field(default=15, ge=3, le=60)
    target_min: Optional[float] = None
    target_max: Optional[float] = None
    cov_shrinkage: float = Field(default=0.0, ge=0.0, le=1.0)
    cov_estimator: Optional[str] = Field(default=None, description="sample|diag|lw")

@router.post("/frontier")
async def frontier(req: FrontierRequest) -> dict[str, Any]:
    """Generate Efficient Frontier points"""
    # Get return estimates
    prices = get_downloaded_series(req.tickers, req.start, req.end, dtype=req.dtype, interval=req.interval)
    rets = prices.pct_change().dropna()
    if rets.empty:
        raise ValueError("No returns available")
        
    # Set target range
    mu = rets.mean().values
    tmin = req.target_min if req.target_min is not None else float(mu.min()) * 0.5
    tmax = req.target_max if req.target_max is not None else float(mu.max()) * 1.5
    if tmax <= tmin:
        tmax = tmin + abs(tmin) + 1e-6
        
    # Generate frontier points
    targets = np.linspace(tmin, tmax, req.n_points)
    points = []
    best_idx = None
    best_metric = float("-inf")
    
    for i, target in enumerate(targets):
        result = walkforward_mvo_target_return(
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
            target_return=float(target),
            cov_shrinkage=req.cov_shrinkage,
            cov_estimator=req.cov_estimator
        )
        
        details = normalize_details(result.get("details"))
        point = {
            "idx": i,
            "target": float(target),
            "metrics": details.get("metrics", {})
        }
        points.append(point)
        
        metric = float(details.get("metrics", {}).get("sharpe", float("-inf")))
        if metric > best_metric:
            best_metric = metric
            best_idx = i
            
    return {
        "targets": targets.tolist(),
        "points": points,
        "best_idx": best_idx
    }