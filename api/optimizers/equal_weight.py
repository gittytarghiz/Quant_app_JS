# api/opt_equal_weight.py
from typing import Any
from fastapi import APIRouter

from api.core import OptimizationRequest
from api.core.utils import format_pnl, format_weights, normalize_details
from portfolio_optimization.walkforward_equal_weight import walkforward_equal_weight

router = APIRouter(prefix="/opt", tags=["opt"])


@router.post("/equal-weight")
async def equal_weight(req: OptimizationRequest) -> dict[str, Any]:
    """
    Equal weight portfolio — return PnL + Weights (JSON-safe).
    Only change vs before: include weights formatted as list[dict].
    """
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
        "pnl": format_pnl(result.get("pnl")),
        "weights": format_weights(result.get("weights")),  
        "details": normalize_details(result.get("details")),
    }
