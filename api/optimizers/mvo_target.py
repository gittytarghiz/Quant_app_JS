from fastapi import APIRouter
from typing import Any, Optional
from pydantic import Field

from api.core import OptimizationRequest
from api.core.utils import format_pnl, format_weights, normalize_details
from portfolio_optimization.walkforward_mvo_target import walkforward_mvo_target_return

router = APIRouter(prefix="/opt", tags=["opt"])


class MVOTargetRequest(OptimizationRequest):
    target_return: float = Field(default=0.0)
    cov_shrinkage: float = Field(default=0.0, ge=0.0, le=1.0)
    cov_estimator: Optional[str] = Field(default=None, description="sample|diag|lw")


@router.post("/mvo-target")
async def mvo_target(req: MVOTargetRequest) -> dict[str, Any]:
    """Target Return Mean-Variance Optimization — return PnL + Weights + Details"""
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
        interest_rate=req.interest_rate,
        target_return=req.target_return,
        cov_shrinkage=req.cov_shrinkage,
        cov_estimator=req.cov_estimator,
    )

    return {
        "pnl": format_pnl(result.get("pnl")),
        "weights": format_weights(result.get("weights")),
        "details": normalize_details(result.get("details")),
    }
