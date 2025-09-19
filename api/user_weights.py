from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from portfolio_optimization.walkforward_user_weights import (
    walkforward_user_weights as _wf_uw,
)
from .utils import df_to_records, series_to_records, normalize_details, compute_metrics


router = APIRouter(prefix="/opt", tags=["opt"])


class UserWeightsRequest(BaseModel):
    tickers: list[str]
    start: str
    end: str
    dtype: str = Field(default="close")
    interval: str = Field(default="1d")
    rebalance: str = Field(default="monthly")
    costs: Optional[dict[str, float]] = None
    static_weights: Optional[dict[str, float]] = None
    normalize: bool = True
    leverage: float = 1.0


@router.post("/user-weights")
def user_weights(req: UserWeightsRequest) -> dict[str, Any]:
    # Normalize ticker symbols to uppercase for matching
    tickers = [t.upper() for t in req.tickers]
    static = None
    if req.static_weights:
        static = {k.upper(): float(v) for k, v in req.static_weights.items()}

    res = _wf_uw(
        tickers=tickers,
        start=req.start,
        end=req.end,
        dtype=req.dtype,
        interval=req.interval,
        rebalance=req.rebalance,
        costs=req.costs,
        static_weights=static,
        weights_df=None,
        normalize=req.normalize,
        leverage=req.leverage,
    )
    raw_w = res.get("weights")
    raw_p = res.get("pnl")
    details = normalize_details(res.get("details"))
    try:
        details["metrics"] = compute_metrics(raw_p, raw_w)
    except Exception:
        pass
    return {"weights": df_to_records(raw_w), "pnl": series_to_records(raw_p, value_name="pnl"), "details": details}
