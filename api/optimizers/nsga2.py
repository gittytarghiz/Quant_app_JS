from fastapi import APIRouter
from typing import Any
from pydantic import Field
from api.core import OptimizationRequest, OptimizationResponse
from api.core.utils import format_weights, format_pnl, normalize_details
from portfolio_optimization.walkforward_nsga2 import walkforward_nsga2

router = APIRouter(prefix="/opt", tags=["opt"])

class NSGA2Request(OptimizationRequest):
    tries: int = Field(default=48, ge=1, description="Number of random candidates per rebalance")
    seed: int = Field(default=42)
    leverage: float = Field(default=1.0)

@router.post("/nsga2", response_model=OptimizationResponse)
async def nsga2(req: NSGA2Request) -> dict[str, Any]:
    """NSGA-II Multi-objective Optimization"""
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
        tries=req.tries,
        seed=req.seed,
    )
    
    return {
        "weights": format_weights(result.get("weights")),
        "pnl": format_pnl(result.get("pnl")),
        "details": normalize_details(result.get("details"))
    }