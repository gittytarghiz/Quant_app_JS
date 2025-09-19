from fastapi import APIRouter
from typing import Any
from api.core import OptimizationRequest, OptimizationResponse
from portfolio_optimization.walkforward_equal_weight import walkforward_equal_weight
from api.core.utils import format_weights, format_pnl, normalize_details

router = APIRouter(prefix="/opt", tags=["opt"])

@router.post("/equal-weight", response_model=OptimizationResponse)
async def equal_weight(req: OptimizationRequest) -> dict[str, Any]:
    """Equal weight portfolio optimization"""
    result = walkforward_equal_weight(
        tickers=req.tickers,
        start=req.start,
        end=req.end,
        dtype=req.dtype,
        interval=req.interval,
        rebalance=req.rebalance,
        costs=req.costs,
        min_weight=req.min_weight,
        max_weight=req.max_weight,
        leverage=req.leverage,
    )
    
    return {
        "weights": format_weights(result.get("weights")),
        "pnl": format_pnl(result.get("pnl")),
        "details": normalize_details(result.get("details"))
    }