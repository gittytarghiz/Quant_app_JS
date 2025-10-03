from fastapi import APIRouter
from typing import Any, Optional
from pydantic import Field

from api.core import OptimizationRequest
from api.core.utils import format_pnl, format_weights, normalize_details
from portfolio_optimization.walkforward_mvo import walkforward_mvo

router = APIRouter(prefix="/opt", tags=["opt"])


class MVORequest(OptimizationRequest):
    objective: str = Field(default="sharpe")
    objective_params: Optional[dict[str, float]] = None


@router.post("/mvo")
async def mvo(req: MVORequest) -> dict[str, Any]:
    """Mean-Variance Optimization — return PnL + Weights + Details"""
    result = walkforward_mvo(
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
        objective=req.objective,
        objective_params=req.objective_params,
    )

    return {
        "pnl": format_pnl(result.get("pnl")),
        "weights": format_weights(result.get("weights")),
        "details": normalize_details(result.get("details")),
    }
