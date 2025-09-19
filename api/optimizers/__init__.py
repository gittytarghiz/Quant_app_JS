from fastapi import APIRouter
from . import (
    equal_weight,
    mvo,
    min_variance,
    risk_parity,
    ga,
    nsga2,
    erc,
    mvo_target,
    frontier,
    user_weights,
    pso
)

# List of all available optimizer routers
routers = [
    equal_weight.router,
    mvo.router,
    min_variance.router,
    risk_parity.router,
    ga.router,
    nsga2.router,
    erc.router,
    mvo_target.router,
    frontier.router,
    user_weights.router,
    pso.router
]

# Combined router for all optimizers
router = APIRouter()
for r in routers:
    router.include_router(r)