from .optimizers import router as optimizer_router
from .data import router as data_router

# List of all available routers
routers = [
    optimizer_router,
    data_router
]
