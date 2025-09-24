import logging
import logging.config

# Override default logging config (especially for pyswarms) to use console only
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
})

import os
import sys
from pathlib import Path

# Ensure the project root is in the Python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.core import setup_error_handlers
from api import routers

# Initialize FastAPI app
app = FastAPI(title="Portfolio Optimizers API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://quant-app-js.vercel.app",          # production frontend
        "https://quant-app-38wwfyc47-targhis-projects.vercel.app"  # preview frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Setup error handlers
setup_error_handlers(app)

# Include all routers
for router in routers:
    app.include_router(router)


@app.get("/")
async def index():
    """Root endpoint to verify API is working."""
    return {
        "status": "ok",
        "python_version": sys.version,
        "python_path": sys.path,
        "docs_url": "/docs",
        "message": "API is running. See /docs for usage details."
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
