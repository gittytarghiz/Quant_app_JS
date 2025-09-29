from typing import Any, Optional, List
from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.core.utils import format_pnl, format_weights, normalize_details
from portfolio_optimization.walkforward_user_weights import walkforward_user_weights

router = APIRouter(prefix="/opt", tags=["opt"])

class UserWeightsRequest(BaseModel):
    tickers: List[str]
    static_weights: Optional[dict[str, float]] = None
    start: str
    end: str
    dtype: str = "close"
    interval: str = "1d"
    rebalance: str = "monthly"
    costs: Optional[dict[str, float]] = None
    leverage: float = Field(default=1.0, ge=0.0, le=5.0)

@router.post("/user-weights")
async def user_weights(req: UserWeightsRequest) -> dict[str, Any]:
    """User-specified static portfolio weights — return PnL + Weights + Details"""
    result = walkforward_user_weights(
        tickers=[t.upper() for t in req.tickers],
        start=req.start,
        end=req.end,
        dtype=req.dtype,
        interval=req.interval,
        rebalance=req.rebalance,
        costs=req.costs,
        static_weights={k.upper(): float(v) for k, v in (req.static_weights or {}).items()},
        weights_df=None,
        normalize=True,
        leverage=req.leverage,
    )

    return {
        "pnl": format_pnl(result.get("pnl")),
        "weights": format_weights(result.get("weights")),
        "details": normalize_details(result.get("details")),
    }
