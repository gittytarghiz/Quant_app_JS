from fastapi import APIRouter
from typing import Any
from pydantic import Field
from api.core import OptimizationRequest, OptimizationResponse
from api.core.utils import format_weights, format_pnl, normalize_details
from portfolio_optimization.walkforward_pso import walkforward_pso

router = APIRouter(prefix="/opt", tags=["opt"])

class PSORequest(OptimizationRequest):
    pso_particles: int = Field(default=60, ge=1)
    pso_iters: int = Field(default=80, ge=1)
    pso_c1: float = Field(default=1.5, gt=0)
    pso_c2: float = Field(default=1.5, gt=0)
    pso_w: float = Field(default=0.7, ge=0, le=1)
    pso_seed: int = Field(default=42)
    objective: str = Field(default="sharpe")
    cvar_alpha: float = Field(default=0.05, gt=0, lt=1)

@router.post("/pso", response_model=OptimizationResponse)
async def pso(req: PSORequest) -> dict[str, Any]:
    """Particle Swarm Optimization"""
    result = walkforward_pso(
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
        pso_particles=req.pso_particles,
        pso_iters=req.pso_iters,
        pso_c1=req.pso_c1,
        pso_c2=req.pso_c2,
        pso_w=req.pso_w,
        pso_seed=req.pso_seed,
        objective=req.objective,
        cvar_alpha=req.cvar_alpha
    )
    
    return {
        "weights": format_weights(result.get("weights")),
        "pnl": format_pnl(result.get("pnl")),
        "details": normalize_details(result.get("details"))
    }