from typing import Any, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.core.utils import format_pnl, format_weights, normalize_details
from portfolio_optimization.walkforward_min_variance import walkforward_min_variance

router = APIRouter(prefix="/opt", tags=["opt"])

class MinVarianceRequest(BaseModel):
    tickers: list[str]
    start: str
    end: str
    dtype: str = "close"
    interval: str = "1d"
    rebalance: str = "monthly"
    costs: Optional[dict[str, float]] = None
    min_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    max_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    min_obs: int = Field(default=60, ge=10)
    leverage: float = Field(default=1.0, ge=0.0, le=5.0)
    interest_rate: float = Field(default=0.0, ge=-1.0, le=1.0)  # ✅ added field

@router.post("/min-variance")
def min_variance(req: MinVarianceRequest) -> dict[str, Any]:
    """Minimum Variance Portfolio — return PnL + Weights + Details"""
    result = walkforward_min_variance(
        tickers=req.tickers,
        start=req.start,
        end=req.end,
        dtype=req.dtype,
        interval=req.interval,
        rebalance=req.rebalance,
        interest_rate=req.interest_rate,
        costs=req.costs,
        min_weight=req.min_weight,
        max_weight=req.max_weight,
        min_obs=req.min_obs,
        leverage=req.leverage,
    )

    return {
        "pnl": format_pnl(result.get("pnl")),
        "weights": format_weights(result.get("weights")),
        "details": normalize_details(result.get("details")),
    }
