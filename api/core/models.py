from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class OptimizationRequest(BaseModel):
    tickers: list[str]
    start: str
    end: str
    dtype: str = Field(default="close")
    interval: str = Field(default="1d")
    rebalance: str = Field(default="monthly")
    costs: Optional[Dict[str, float]] = None
    min_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    max_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    interest_rate: float = 0.04   # NEW FIELD

    min_obs: int = Field(default=60, ge=0)
    leverage: float = Field(default=1.0, ge=0.0, le=5.0)

class OptimizationResponse(BaseModel):
    weights: list[Dict[str, Any]]
    pnl: list[Dict[str, Any]]
    details: Dict[str, Any]