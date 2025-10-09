"use client";

import { useState } from "react";
import { OptimizerNav } from "../../../lib/components/OptimizerNav";
import { Analytics } from "../../../lib/analytics";
import AssetPicker from "../../../lib/components/AssetPicker";

type OptResponse = {
  weights: Array<Record<string, string | number | null>>;
  pnl: Array<{ date: string; pnl: number | null }>;
  details: Record<string, any>;
};

const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function GAPage() {
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  const [start, setStart] = useState("2020-01-01");
  const [end, setEnd] = useState("2025-01-01");

  const [objective, setObjective] = useState("sharpe");
  const [rf, setRf] = useState(0);
  const [mar, setMar] = useState(0);
  const [alpha, setAlpha] = useState(0.05);

  const [lev, setLev] = useState(1);
  const [intRate, setIntRate] = useState(4.0);
  const [minW, setMinW] = useState(0);
  const [maxW, setMaxW] = useState(1);
  const [interval, setInterval] = useState("1d");
  const [rebalance, setRebalance] = useState("monthly");
  const [minObs, setMinObs] = useState(60);
  const [seed, setSeed] = useState<number | undefined>(42);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [data, setData] = useState<OptResponse | null>(null);

  async function run() {
    if (selectedTickers.length === 0) {
      setErr("Please select at least one asset.");
      return;
    }

    setLoading(true);
    setErr(null);
    try {
      const objective_params: Record<string, number> = {};
      if (objective === "sharpe" && rf) objective_params.rf = Number(rf);
      if (objective === "sortino" && mar) objective_params.mar = Number(mar);
      if (objective === "cvar" && alpha) objective_params.alpha = Number(alpha);

      const payload: any = {
        tickers: selectedTickers,
        start,
        end,
        dtype: "close",
        interval,
        rebalance,
        min_weight: Number(minW),
        max_weight: Number(maxW),
        min_obs: Number(minObs),
        leverage: Number(lev),
        interest_rate: intRate / 100.0,
        objective,
        ...(Object.keys(objective_params).length ? { objective_params } : {}),
        ...(seed !== undefined ? { seed: Number(seed) } : {}),
      };

      const url = `${API}/opt/ga`;
      console.log("POST", url, payload);
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        let msg = `HTTP ${res.status} ${res.statusText}`;
        try {
          const j = await res.json();
          if (j?.detail) msg += ` — ${j.detail}`;
        } catch {}
        throw new Error(msg);
      }

      const json = (await res.json()) as OptResponse;
      console.log("OK /opt/ga", (json?.pnl || []).length);
      setData(json);
    } catch (e: any) {
      setErr(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="p-6 text-gray-100 bg-gray-950 min-h-screen">
      <OptimizerNav />
      <h2 className="text-xl font-semibold mb-4 text-teal-400">
        Genetic Algorithm Optimizer
      </h2>

      <div className="flex flex-col gap-6 mb-6">
        {/* Asset Picker */}
        <AssetPicker onChange={setSelectedTickers} />

        {/* Parameters */}
        <div className="flex flex-wrap gap-x-6 gap-y-4 items-end">
          {/* Dates */}
          <label className="flex flex-col text-sm text-gray-300">
            <span className="mb-1">Start</span>
            <input
              type="date"
              value={start}
              onChange={(e) => setStart(e.target.value)}
              className="bg-gray-900 text-white p-2 rounded-md w-28"
            />
          </label>
          <label className="flex flex-col text-sm text-gray-300">
            <span className="mb-1">End</span>
            <input
              type="date"
              value={end}
              onChange={(e) => setEnd(e.target.value)}
              className="bg-gray-900 text-white p-2 rounded-md w-28"
            />
          </label>

          {/* Objective */}
          <label className="flex flex-col text-sm text-gray-300">
            <span className="mb-1">Objective</span>
            <select
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              className="bg-gray-900 text-white p-2 rounded-md w-40"
            >
              <option value="sharpe">Sharpe</option>
              <option value="sortino">Sortino</option>
              <option value="calmar">Calmar</option>
              <option value="cvar">CVaR</option>
              <option value="return_to_turnover">Return/Turnover</option>
              <option value="return_to_drawdown">Return/Drawdown</option>
              <option value="kelly">Kelly</option>
              <option value="diversification">Diversification</option>
            </select>
          </label>

          {objective === "sharpe" && (
            <label className="flex flex-col text-sm text-gray-300">
              <span className="mb-1">rf</span>
              <input
                type="number"
                step={0.0001}
                value={rf}
                onChange={(e) => setRf(Number(e.target.value || 0))}
                className="bg-gray-900 text-white p-2 rounded-md w-24"
              />
            </label>
          )}
          {objective === "sortino" && (
            <label className="flex flex-col text-sm text-gray-300">
              <span className="mb-1">MAR</span>
              <input
                type="number"
                step={0.0001}
                value={mar}
                onChange={(e) => setMar(Number(e.target.value || 0))}
                className="bg-gray-900 text-white p-2 rounded-md w-24"
              />
            </label>
          )}
          {objective === "cvar" && (
            <label className="flex flex-col text-sm text-gray-300">
              <span className="mb-1">Alpha</span>
              <input
                type="number"
                step={0.01}
                min={0.01}
                max={0.5}
                value={alpha}
                onChange={(e) => setAlpha(Number(e.target.value || 0.05))}
                className="bg-gray-900 text-white p-2 rounded-md w-24"
              />
            </label>
          )}

          {/* Numeric parameters */}
          {[
            { label: "Leverage", value: lev, set: setLev, step: 0.1 },
            { label: "Interest %", value: intRate, set: setIntRate, step: 0.1 },
            { label: "Min W", value: minW, set: setMinW, step: 0.01 },
            { label: "Max W", value: maxW, set: setMaxW, step: 0.01 },
            { label: "Min Obs", value: minObs, set: setMinObs, step: 1 },
            { label: "Seed", value: seed ?? 0, set: setSeed, step: 1 },
          ].map((p, i) => (
            <label key={i} className="flex flex-col text-sm text-gray-300">
              <span className="mb-1">{p.label}</span>
              <input
                type="number"
                step={p.step}
                value={p.value}
                onChange={(e) => p.set(Number(e.target.value))}
                className="bg-gray-900 text-white p-2 rounded-md w-24"
              />
            </label>
          ))}

          {/* Interval & Rebalance */}
          <label className="flex flex-col text-sm text-gray-300">
            <span className="mb-1">Interval</span>
            <select
              value={interval}
              onChange={(e) => setInterval(e.target.value)}
              className="bg-gray-900 text-white p-2 rounded-md w-32"
            >
              <option value="1d">1d</option>
              <option value="weekly">weekly</option>
              <option value="monthly">monthly</option>
            </select>
          </label>
          <label className="flex flex-col text-sm text-gray-300">
            <span className="mb-1">Rebalance</span>
            <select
              value={rebalance}
              onChange={(e) => setRebalance(e.target.value)}
              className="bg-gray-900 text-white p-2 rounded-md w-32"
            >
              <option value="daily">daily</option>
              <option value="weekly">weekly</option>
              <option value="monthly">monthly</option>
              <option value="quarterly">quarterly</option>
            </select>
          </label>

          <button
            onClick={run}
            disabled={loading}
            className={`px-5 py-2 rounded-md font-medium text-white whitespace-nowrap ${
              loading
                ? "bg-teal-800 cursor-wait"
                : "bg-teal-600 hover:bg-teal-500"
            }`}
          >
            {loading ? "Running..." : "Run Optimization"}
          </button>
        </div>
      </div>

      {err && <p className="text-red-400 mb-2">Error: {err}</p>}

      {data && (
        <div className="mt-6">
          <Analytics data={data} />
        </div>
      )}
    </div>
  );
}
