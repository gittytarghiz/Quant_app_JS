from fastapi import APIRouter
from typing import Any, Optional, Union, Tuple
from pydantic import Field
from api.core import OptimizationRequest, OptimizationResponse
from api.core.utils import format_weights, format_pnl, normalize_details
from portfolio_optimization.walkforward_ga import walkforward_ga_fast

router = APIRouter(prefix="/opt", tags=["opt"])

class GARequest(OptimizationRequest):
    objective: str = Field(default="sharpe")
    objective_params: Optional[dict[str, float]] = None
    seed: Optional[int] = None
    de_maxiter: int = Field(default=40, ge=1)
    de_popsize: int = Field(default=25, ge=1)
    de_tol: float = Field(default=0.01, gt=0)
    de_mutation: Union[Tuple[float, float], float] = Field(default=(0.5, 1.0))
    de_recombination: float = Field(default=0.7, ge=0, le=1)
    de_strategy: str = Field(default="best1bin")
    de_seed: int = Field(default=42)
    de_workers: Optional[int] = Field(default=1)
    de_polish: bool = Field(default=False)

@router.post("/ga", response_model=OptimizationResponse)
async def ga(req: GARequest) -> dict[str, Any]:
    """Genetic Algorithm Optimization"""
    result = walkforward_ga_fast(
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
        de_seed=req.de_seed,
        de_maxiter=req.de_maxiter,
        de_popsize=req.de_popsize,
        de_tol=req.de_tol,
        de_mutation=req.de_mutation,
        de_recombination=req.de_recombination,
        de_strategy=req.de_strategy,
        de_workers=req.de_workers,
        de_polish=req.de_polish
    )
    
    return {
        "weights": format_weights(result.get("weights")),
        "pnl": format_pnl(result.get("pnl")),
        "details": normalize_details(result.get("details"))
    }