"use client";

import { useState } from "react";
import { OptimizerNav } from "../../../lib/components/OptimizerNav";
import { Analytics } from "../../../lib/analytics";
import AssetPicker from "../../../lib/components/AssetPicker";

type MvoResponse = {
  weights: Array<Record<string, string | number | null>>;
  pnl: Array<{ date: string; pnl: number | null }>;
  details: Record<string, any>;
};

const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function MvoPage() {
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  const [start, setStart] = useState("2020-01-01");
  const [end, setEnd] = useState("2025-01-01");
  const [objective, setObjective] = useState("min_vol");
  const [lev, setLev] = useState(1);
  const [intRate, setIntRate] = useState(4.0);
  const [minW, setMinW] = useState(0);
  const [maxW, setMaxW] = useState(1);
  const [minObs, setMinObs] = useState(60);
  const [rebalance, setRebalance] = useState("monthly");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<MvoResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    if (selectedTickers.length === 0) {
      setErr("Please select at least one asset.");
      return;
    }

    setLoading(true);
    setErr(null);
    try {
      const body = {
        tickers: selectedTickers,
        start,
        end,
        objective,
        leverage: Number(lev),
        interest_rate: intRate / 100.0,
        min_weight: Number(minW),
        max_weight: Number(maxW),
        min_obs: Number(minObs),
        rebalance,
        dtype: "close",
        interval: "1d",
      };

      const url = `${API}/opt/mvo`;
      console.log("POST", url, body);
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        let msg = `HTTP ${res.status} ${res.statusText}`;
        try {
          const errJson = await res.json();
          if (errJson?.detail) msg += ` — ${errJson.detail}`;
        } catch {}
        throw new Error(msg);
      }

      const json = await res.json();
      console.log("OK /opt/mvo", (json?.pnl || []).length);
      setData(json);
    } catch (e: any) {
      console.error("MVO failed", e);
      setErr(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="p-6 text-gray-100 bg-gray-950 min-h-screen">
      <OptimizerNav />
      <h2 className="text-xl font-semibold mb-4 text-teal-400">
        Mean-Variance Optimizer
      </h2>

      <div className="flex flex-col gap-6 mb-6">
        {/* Asset Picker */}
        <AssetPicker onChange={setSelectedTickers} />

        {/* Parameter Controls */}
        <div className="flex flex-wrap gap-x-6 gap-y-4 items-end">
          {/* Date and Objective */}
          <label className="flex flex-col text-sm text-gray-300">
            <span className="mb-1">Start</span>
            <input
              type="date"
              value={start}
              onChange={(e) => setStart(e.target.value)}
              className="bg-gray-900 text-white p-2 rounded-md w-32"
            />
          </label>

          <label className="flex flex-col text-sm text-gray-300">
            <span className="mb-1">End</span>
            <input
              type="date"
              value={end}
              onChange={(e) => setEnd(e.target.value)}
              className="bg-gray-900 text-white p-2 rounded-md w-32"
            />
          </label>

          <label className="flex flex-col text-sm text-gray-300">
            <span className="mb-1">Objective</span>
            <select
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              className="bg-gray-900 text-white p-2 rounded-md w-36"
            >
              <option value="min_vol">Min Vol</option>
              <option value="max_return">Max Return</option>
              <option value="mean_var">Mean-Var</option>
            </select>
          </label>

          {/* Numeric Inputs */}
          {[
            {
              label: "Leverage",
              type: "number",
              step: 0.1,
              min: 0,
              value: lev,
              onChange: (v: any) => setLev(Number(v.target.value || 1)),
            },
            {
              label: "Interest %",
              type: "number",
              step: 0.1,
              min: 0,
              value: intRate,
              onChange: (v: any) => setIntRate(Number(v.target.value || 0)),
            },
            {
              label: "Min W",
              type: "number",
              step: 0.01,
              min: 0,
              max: 1,
              value: minW,
              onChange: (v: any) => setMinW(Number(v.target.value || 0)),
            },
            {
              label: "Max W",
              type: "number",
              step: 0.01,
              min: 0,
              max: 1,
              value: maxW,
              onChange: (v: any) => setMaxW(Number(v.target.value || 1)),
            },
            {
              label: "Min Obs",
              type: "number",
              step: 1,
              min: 2,
              value: minObs,
              onChange: (v: any) => setMinObs(Number(v.target.value || 60)),
            },
          ].map((input, idx) => (
            <label key={idx} className="flex flex-col text-sm text-gray-300">
              <span className="mb-1">{input.label}</span>
              <input
                type={input.type}
                step={input.step}
                min={input.min}
                max={input.max}
                value={input.value}
                onChange={input.onChange}
                className="bg-gray-900 text-white p-2 rounded-md w-28"
              />
            </label>
          ))}

          {/* Rebalance Selector */}
          <label className="flex flex-col text-sm text-gray-300">
            <span className="mb-1">Rebalance</span>
            <select
              value={rebalance}
              onChange={(e) => setRebalance(e.target.value)}
              className="bg-gray-900 text-white p-2 rounded-md w-36"
            >
              <option value="daily">Daily</option>
              <option value="weekly">Weekly</option>
              <option value="monthly">Monthly</option>
              <option value="quarterly">Quarterly</option>
            </select>
          </label>

          {/* Run Button */}
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
