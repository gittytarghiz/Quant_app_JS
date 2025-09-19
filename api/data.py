from fastapi import APIRouter
from typing import List
from pydantic import BaseModel, Field
from data_management.monolith_loader import get_downloaded_series
from api.core.utils import format_weights

router = APIRouter(prefix="/data", tags=["data"])

class DataRequest(BaseModel):
    tickers: List[str]
    start: str
    end: str
    dtype: str = Field(default="close")
    interval: str = Field(default="1d")


@router.post("/prices")
async def get_prices(req: DataRequest):
    """Get historical price data for given tickers"""
    if not req.tickers:
        raise ValueError("tickers must be a non-empty list")
        
    df = get_downloaded_series(
        tickers=req.tickers,
        start=req.start,
        end=req.end,
        dtype=req.dtype,
        interval=req.interval
    )
    
    return {
        "records": format_weights(df),
        "columns": ["date"] + list(df.columns) if not df.empty else ["date"],
        "meta": {
            "tickers": req.tickers,
            "start": req.start,
            "end": req.end,
            "dtype": req.dtype,
            "interval": req.interval,
            "count": len(df) if not df.empty else 0,
            "status": "done"
        }
    }
