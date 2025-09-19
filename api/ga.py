from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Tuple
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
import logging
from datetime import datetime

from portfolio_optimization.walkforward_ga import walkforward_ga_fast as _wf_ga
from .utils import df_to_records, series_to_records, normalize_details, compute_metrics

logger = logging.getLogger("api.ga")


router = APIRouter(prefix="/opt", tags=["opt"])


class GARequest(BaseModel):
    # Basic portfolio parameters
    tickers: List[str]
    start: str
    end: str
    dtype: str = Field(default="close")
    interval: str = Field(default="1d")
    rebalance: str = Field(default="monthly")
    
    # Portfolio constraints
    min_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    max_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    leverage: float = Field(default=1.0, ge=0.0, le=5.0)
    min_obs: int = Field(default=60, ge=20)
    costs: Optional[Dict[str, float]] = None
    
    # Optimization parameters
    objective: str = Field(default="sharpe")
    objective_params: Optional[Dict[str, float]] = None
    
    # Genetic Algorithm parameters
    seed: Optional[int] = None
    de_maxiter: int = Field(default=40, ge=10, le=200)
    de_popsize: int = Field(default=25, ge=10, le=100)
    de_tol: float = Field(default=0.01, gt=0.0, le=0.1)
    de_mutation: Union[Tuple[float, float], float] = Field(default=(0.5, 1.0))
    de_recombination: float = Field(default=0.7, gt=0.0, le=1.0)
    de_strategy: str = Field(default="best1bin")
    de_seed: int = Field(default=42)
    de_workers: Optional[int] = Field(default=1)
    de_polish: bool = Field(default=False)

    @validator('tickers')
    def validate_tickers(cls, v):
        if not v:
            raise ValueError("tickers must be a non-empty list")
        if len(v) > 64:
            raise ValueError("too many tickers; please limit to 64 or fewer")
        return [t.strip().upper() for t in v if t.strip()]

    @validator('rebalance')
    def validate_rebalance(cls, v):
        allowed = {"daily", "weekly", "monthly", "quarterly"}
        if v.lower() not in allowed:
            raise ValueError(f"rebalance must be one of {sorted(allowed)}")
        return v.lower()

    @validator('interval')
    def validate_interval(cls, v):
        allowed = {"1d", "1wk", "1mo"}
        if v.lower() not in allowed:
            raise ValueError(f"interval must be one of {sorted(allowed)}")
        return v.lower()

    @validator('objective')
    def validate_objective(cls, v):
        allowed = {
            "sharpe", "sortino", "calmar", "cvar", "kelly", 
            "diversification", "min_vol", "max_return"
        }
        if v.lower() not in allowed:
            raise ValueError(f"objective must be one of {sorted(allowed)}")
        return v.lower()

    @validator('de_strategy')
    def validate_strategy(cls, v):
        allowed = {
            "best1bin", "best1exp", "rand1exp", "rand1bin",
            "rand2exp", "best2exp", "rand2bin", "best2bin"
        }
        if v not in allowed:
            raise ValueError(f"de_strategy must be one of {sorted(allowed)}")
        return v

    @validator('de_mutation')
    def validate_mutation(cls, v):
        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ValueError("de_mutation tuple must have exactly 2 elements")
            if not (0 < v[0] <= v[1] <= 2):
                raise ValueError("de_mutation range must be (0,2] with first <= second")
        elif isinstance(v, (int, float)):
            if not (0 < v <= 2):
                raise ValueError("de_mutation scalar must be in (0,2]")
        else:
            raise ValueError("de_mutation must be float or (float,float)")
        return v

    @validator('costs')
    def validate_costs(cls, v):
        if not v:
            return None
        allowed = {"bps", "slippage_bps", "spread_bps"}
        validated = {}
        for k, val in v.items():
            if k in allowed and val is not None:
                try:
                    validated[k] = float(val)
                    if validated[k] < 0:
                        raise ValueError(f"Cost {k} cannot be negative")
                except (TypeError, ValueError):
                    continue
        return validated if validated else None


@router.post("/ga")
async def ga(req: GARequest) -> Dict[str, Any]:
    """Genetic Algorithm optimization endpoint"""
    try:
        # Log the request
        logger.info(
            "GA optimization request: tickers=%d objective=%s interval=%s rebalance=%s",
            len(req.tickers), req.objective, req.interval, req.rebalance
        )

        # Use provided seed or fallback to de_seed
        de_seed = int(req.seed) if req.seed is not None else int(req.de_seed)

        # Run optimization with error handling
        try:
            res = _wf_ga(
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
                objective=req.objective,
                objective_params=req.objective_params,
                de_maxiter=req.de_maxiter,
                de_popsize=req.de_popsize,
                de_tol=req.de_tol,
                de_mutation=req.de_mutation,
                de_recombination=req.de_recombination,
                de_strategy=req.de_strategy,
                de_seed=de_seed,
                de_workers=req.de_workers,
                de_polish=req.de_polish,
            )
        except ValueError as e:
            logger.error(f"Optimization error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected optimization error: {str(e)}")
            raise HTTPException(status_code=500, detail="Optimization failed")

        # Process results
        raw_w = res.get("weights")
        raw_p = res.get("pnl")
        details = normalize_details(res.get("details"))

        # Add performance metrics
        try:
            details["metrics"] = compute_metrics(raw_p, raw_w)
        except Exception as e:
            logger.warning(f"Failed to compute metrics: {str(e)}")
            details["metrics"] = {}

        # Format response
        response = {
            "weights": df_to_records(raw_w),
            "pnl": series_to_records(raw_p, value_name="pnl"),
            "details": details
        }

        # Log success
        logger.info(
            "GA optimization completed: weights_shape=%s pnl_length=%d",
            raw_w.shape if raw_w is not None else None,
            len(raw_p) if raw_p is not None else 0
        )

        return response

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
