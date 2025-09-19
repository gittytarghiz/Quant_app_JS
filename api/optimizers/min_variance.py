from typing import Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from portfolio_optimization.walkforward_min_variance import walkforward_min_variance
from ..utils import df_to_records, series_to_records, compute_metrics

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

@router.post("/min-variance")
def min_variance(req: MinVarianceRequest) -> dict[str, Any]:
    """Minimum Variance Portfolio optimization endpoint."""
    try:
        res = walkforward_min_variance(
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
            leverage=req.leverage
        )

        weights = df_to_records(res["weights"])
        pnl = series_to_records(res["pnl"], value_name="pnl")
        details = dict(res["details"])
        details["metrics"] = compute_metrics(res["pnl"], res["weights"])

        return {"weights": weights, "pnl": pnl, "details": details}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))