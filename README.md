opt_cpp — C++17 pybind11 optimizers
===================================

This module provides a minimal‑memory C++ backend for portfolio optimization.

Delivered bindings
- NSGA‑II multi‑objective optimizer: `nsga2_optimize(returns, objectives, params, lo, hi, popsize, ngen, seed, crossover_prob, mutation_prob, w_prev=None)`
- Projection utility: `prop_box_sum1(w, lo, hi, iters=6)`

Design goals
- C++17 + pybind11 module (`opt_cpp`)
- Zero‑copy NumPy views; no STL copies of inputs
- Reuse preallocated buffers; O(T·N + popsize·N) working memory
- Optional OpenMP parallel evaluation
- Deterministic for fixed seed

Build
1. Ensure a C++17 compiler and Python dev headers are available.
2. From repo root:

   mkdir -p build && cd build
   cmake -S ../cpp -B . -DOPTCPP_USE_OPENMP=ON
   cmake --build . --config Release

3. Add the build dir to PYTHONPATH or install the built module

   export PYTHONPATH=$PWD:$PYTHONPATH
   python -c "import opt_cpp; print(opt_cpp)"

Usage

Python

   import numpy as np, opt_cpp
   R = np.random.normal(0, 0.01, size=(2000, 30)).astype(np.float64)
   W, S = opt_cpp.nsga2_optimize(R, ["sharpe","min_vol","cvar"], {"alpha":0.05, "ann":252.0},
                                 lo=0.0, hi=0.3, popsize=64, ngen=20, seed=42,
                                 crossover_prob=0.9, mutation_prob=0.1)
   print(W.shape, S.shape)  # Pareto front sizes

Notes
- Objectives match Python implementations used in this repo (annualized Sharpe/Sortino/Calmar, CVaR, Kelly, diversification ratio, target‑min‑vol, etc.).
- For `return_to_turnover`, if `w_prev` is None the L1 norm of `w` is used in the denominator.
- Diversification ratio uses per‑asset standard deviations from the input returns (no full covariance allocation).
- Compile with `-DOPTCPP_FLOAT32=ON` to switch to float32.


Docker
------

Local build and run of the full app (API + Next.js UI):

1) Build and start services

   docker compose up --build

   - API: http://localhost:8000/docs
   - UI:  http://localhost:3000

2) Data cache persistence

   Parquet data downloaded by the API is persisted on the host in `data_management/data/` via a bind mount.

3) Environment

   - Frontend proxies `/opt/*` and `/data/*` to the API using `API_URL` (defaults to `http://backend:8000` in compose).
   - You can toggle API debug logs by setting `DM_DEBUG=true` in `docker-compose.yml`.


Git Init & Push
----------------

This repo now includes a comprehensive `.gitignore` to exclude `venv/`, `node_modules/`, build artifacts, and cached parquet data.

Initialize and push to your remote:

1. Initialize the repository

   git init
   git add .
   git commit -m "Initial commit"

2. Add your remote and push

   git remote add origin <YOUR_REMOTE_URL>
   git branch -M main
   git push -u origin main

# Quant_app_JS
# Quant_app_JS
# Quant_app_Note.js
# Quant_app_JS
# Quant_app_JS
