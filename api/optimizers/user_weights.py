from typing import Any, Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from datetime import datetime
import numpy as np

from portfolio_optimization.walkforward_user_weights import walkforward_user_weights
from ..utils import df_to_records, series_to_records, compute_metrics

router = APIRouter(prefix="/opt", tags=["opt"])

class UserWeight(BaseModel):
    ticker: str
    weight: float = Field(..., ge=0.0, le=1.0)

class UserWeightsRequest(BaseModel):
    tickers: List[str]
    # Either provide a list of weights or a static mapping
    weights: Optional[List[UserWeight]] = None
    static_weights: Optional[dict[str, float]] = None
    start: str
    end: str
    dtype: str = "close"
    interval: str = "1d"
    rebalance: str = "monthly"
    costs: Optional[dict[str, float]] = None
    leverage: float = Field(default=1.0, ge=0.0, le=5.0)

    @validator("weights")
    def validate_weights(cls, v, values):
        if v is None:
            return v
        if "tickers" in values:
            tickers = set(values["tickers"])
            weight_tickers = {w.ticker for w in v}

            if not weight_tickers.issubset(tickers):
                raise ValueError("All weight tickers must be in the tickers list")

            total = sum(w.weight for w in v)
            if not np.isclose(total, 1.0, rtol=1e-5):
                raise ValueError(f"Weights must sum to 1.0 (got {total:.4f})")

        return v

@router.post("/user-weights")
async def user_weights(req: UserWeightsRequest) -> dict[str, Any]:
    """Apply user-specified portfolio weights with periodic rebalancing.

    This endpoint accepts either:
    - `weights`: a list of `{ticker, weight}` entries (sums to 1), or
    - `static_weights`: a mapping of ticker -> weight.
    """
    try:
        tickers = [t.upper() for t in req.tickers]

        static = None
        if req.static_weights:
            static = {k.upper(): float(v) for k, v in req.static_weights.items()}

        # If weights list provided, convert to dict
        if req.weights is not None:
            weights_dict = {w.ticker.upper(): w.weight for w in req.weights}
            for t in tickers:
                if t not in weights_dict:
                    weights_dict[t] = 0.0
            static = weights_dict

        result = walkforward_user_weights(
            tickers=tickers,
            start=req.start,
            end=req.end,
            dtype=req.dtype,
            interval=req.interval,
            rebalance=req.rebalance,
            costs=req.costs,
            static_weights=static,
            weights_df=None,
            normalize=True,
            leverage=req.leverage,
        )

        weights = df_to_records(result["weights"])
        pnl = series_to_records(result["pnl"], value_name="pnl")
        details = dict(result["details"])
        details["metrics"] = compute_metrics(result["pnl"], result["weights"])

        return {
            "weights": weights,
            "pnl": pnl,
            "details": details
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))