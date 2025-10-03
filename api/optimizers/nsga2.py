from fastapi import APIRouter
from typing import Any
from pydantic import Field
from api.core import OptimizationRequest
from api.core.utils import format_pnl, format_weights, normalize_details
from portfolio_optimization.walkforward_nsga2 import walkforward_nsga2

router = APIRouter(prefix="/opt", tags=["opt"])

class NSGA2Request(OptimizationRequest):
    tries: int = Field(default=48, ge=1, description="Number of random candidates per rebalance")
    seed: int = Field(default=42)
    leverage: float = Field(default=1.0)

@router.post("/nsga2")
async def nsga2(req: NSGA2Request) -> dict[str, Any]:
    """NSGA-II Multi-objective Optimization — return PnL + Weights + Details"""
    result = walkforward_nsga2(
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
        tries=req.tries,
        seed=req.seed,
    )

    return {
        "pnl": format_pnl(result.get("pnl")),
        "weights": format_weights(result.get("weights")),
        "details": normalize_details(result.get("details")),
    }
