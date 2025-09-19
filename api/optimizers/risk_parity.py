from fastapi import APIRouter
from typing import Any, Optional
from pydantic import Field
from api.core import OptimizationRequest, OptimizationResponse
from api.core.utils import format_weights, format_pnl, normalize_details
from portfolio_optimization.walkforward_risk_parity import walkforward_risk_parity

router = APIRouter(prefix="/opt", tags=["opt"])

class RiskParityRequest(OptimizationRequest):
    cov_estimator: Optional[str] = Field(default=None, description="sample|diag|lw")

@router.post("/risk-parity", response_model=OptimizationResponse)
async def risk_parity(req: RiskParityRequest) -> dict[str, Any]:
    """Risk Parity Optimization"""
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
        cov_estimator=req.cov_estimator
    )
    
    return {
        "weights": format_weights(result.get("weights")),
        "pnl": format_pnl(result.get("pnl")),
        "details": normalize_details(result.get("details"))
    }