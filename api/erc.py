from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from portfolio_optimization.walkforward_risk_parity import (
    walkforward_risk_parity as _wf_rp,
)
from .utils import df_to_records, series_to_records, normalize_details, compute_metrics


router = APIRouter(prefix="/opt", tags=["opt"])


class ERCRequest(BaseModel):
    tickers: list[str]
    start: str
    end: str
    dtype: str = Field(default="close")
    interval: str = Field(default="1d")
    rebalance: str = Field(default="monthly")
    costs: Optional[dict[str, float]] = None
    min_weight: float = 0.0
    max_weight: float = 1.0
    min_obs: int = 60
    leverage: float = 1.0


@router.post("/erc")
def erc(req: ERCRequest) -> dict[str, Any]:
    # Reuse the variance-based risk parity (which is ERC) and relabel
    res = _wf_rp(
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
    )
    raw_w = res.get("weights")
    raw_p = res.get("pnl")
    details = normalize_details(res.get("details"))
    # Relabel for clarity in the response
    try:
        details.setdefault("config", {})["method"] = "erc"
        details["metrics"] = compute_metrics(raw_p, raw_w)
    except Exception:
        pass
    return {"weights": df_to_records(raw_w), "pnl": series_to_records(raw_p, value_name="pnl"), "details": details}

