from fastapi import APIRouter
from typing import Any

from api.core import OptimizationRequest
from api.core.utils import format_pnl, format_weights, normalize_details
from portfolio_optimization.walkforward_risk_parity import walkforward_risk_parity

router = APIRouter(prefix="/opt", tags=["opt"])


@router.post("/erc")
async def erc(req: OptimizationRequest) -> dict[str, Any]:
    """Equal Risk Contribution (ERC) Optimization — return PnL + Weights + Details"""
    result = walkforward_risk_parity(
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
    )

    return {
        "pnl": format_pnl(result.get("pnl")),
        "weights": format_weights(result.get("weights")),
        "details": normalize_details(result.get("details")),
    }
