"use client";

import { useState } from "react";
import { OptimizerNav } from "../../../lib/components/OptimizerNav";
import { Analytics } from "../../../lib/analytics";
import AssetPicker from "../../../lib/components/AssetPicker";

type Resp = {
  weights: Array<Record<string, string | number | null>>;
  pnl: Array<{ date: string; pnl: number | null }>;
  details: any;
};

const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function PSOPage() {
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  const [start, setStart] = useState("2020-01-01");
  const [end, setEnd] = useState("2025-01-01");
  const [objective, setObjective] = useState("sharpe");
  const [lev, setLev] = useState(1);
  const [minW, setMinW] = useState(0);
  const [maxW, setMaxW] = useState(1);
  const [intRate, setIntRate] = useState(4.0);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<Resp | null>(null);
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
        min_weight: Number(minW),
        max_weight: Number(maxW),
        interest_rate: intRate / 100.0,
      };

      const url = `${API}/opt/pso`;
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
      console.log("OK /opt/pso", (json?.pnl || []).length);
      setData(json);
    } catch (e: any) {
      console.error("PSO failed", e);
      setErr(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="p-6 text-gray-100 bg-gray-950 min-h-screen">
      <OptimizerNav />
      <h2 className="text-xl font-semibold mb-4 text-teal-400">
        Particle Swarm Optimization (PSO)
      </h2>

      <div className="flex flex-col gap-6 mb-6">
        {/* Asset Picker */}
        <AssetPicker onChange={setSelectedTickers} />

        {/* Parameters */}
        <div className="flex flex-wrap gap-x-6 gap-y-4 items-end">
          {[
            {
              label: "Start",
              type: "date",
              value: start,
              onChange: (v: any) => setStart(v.target.value),
            },
            {
              label: "End",
              type: "date",
              value: end,
              onChange: (v: any) => setEnd(v.target.value),
            },
          ].map((input, idx) => (
            <label key={idx} className="flex flex-col text-sm text-gray-300">
              <span className="mb-1">{input.label}</span>
              <input
                type={input.type}
                value={input.value}
                onChange={input.onChange}
                className="bg-gray-900 text-white p-2 rounded-md w-28"
              />
            </label>
          ))}

          {/* Objective */}
          <label className="flex flex-col text-sm text-gray-300">
            <span className="mb-1">Objective</span>
            <select
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              className="bg-gray-900 text-white p-2 rounded-md w-44"
            >
              <option value="sharpe">Sharpe</option>
              <option value="sortino">Sortino</option>
              <option value="calmar">Calmar</option>
              <option value="cvar">CVaR</option>
              <option value="ret_turnover">Return/Turnover</option>
              <option value="ret_drawdown">Return/Drawdown</option>
              <option value="kelly">Kelly</option>
              <option value="diversification">Diversification</option>
            </select>
          </label>

          {/* Numeric params */}
          {[
            { label: "Leverage", value: lev, set: setLev, step: 0.1 },
            { label: "Interest %", value: intRate, set: setIntRate, step: 0.1 },
            { label: "Min W", value: minW, set: setMinW, step: 0.01 },
            { label: "Max W", value: maxW, set: setMaxW, step: 0.01 },
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
