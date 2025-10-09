"use client";

import { useState, useEffect, ChangeEvent } from "react";
import { Analytics } from "../../../lib/analytics";
import { OptimizerNav } from "../../../lib/components/OptimizerNav";
import AssetPicker from "../../../lib/components/AssetPicker";

type UserWeightsResponse = {
  weights: Array<Record<string, string | number | null>>;
  pnl: Array<{ date: string; pnl: number | null }>;
  details: Record<string, any>;
};

type ParamInput = {
  label: string;
  type: string;
  step?: number;
  min?: number;
  max?: number;
  value: any;
  onChange: (v: ChangeEvent<HTMLInputElement>) => void;
};

const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function UserWeightsPage() {
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  const [weights, setWeights] = useState<Record<string, number>>({});
  const [start, setStart] = useState("2020-01-01");
  const [end, setEnd] = useState("2025-01-01");
  const [lev, setLev] = useState(1);
  const [intRate, setIntRate] = useState(4.0);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<UserWeightsResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (selectedTickers.length === 0) return;
    const equal = 1 / selectedTickers.length;
    const newW: Record<string, number> = {};
    selectedTickers.forEach((t) => (newW[t] = weights[t] ?? equal));
    const total = Object.values(newW).reduce((a, b) => a + b, 0);
    for (const k in newW) newW[k] /= total;
    setWeights(newW);
  }, [selectedTickers]);

  const updateWeight = (ticker: string, val: number) => {
    const newW = { ...weights, [ticker]: val };
    const total = Object.values(newW).reduce((a, b) => a + b, 0);
    for (const k in newW) newW[k] /= total;
    setWeights(newW);
  };

  async function run() {
    if (selectedTickers.length === 0) {
      setErr("Please select at least one asset.");
      return;
    }
    setLoading(true);
    setErr(null);
    try {
      const weightsString = Object.entries(weights)
        .map(([k, v]) => `${k}:${v.toFixed(4)}`)
        .join(",");

      const params = new URLSearchParams({
        tickers: selectedTickers.join(","),
        start,
        end,
        weights: weightsString,
        leverage: lev.toString(),
        interest_rate: (intRate / 100.0).toString(),
      });

      const res = await fetch(`${API}/user_weights?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();

      const pnl = Array.isArray(json.pnl)
        ? json.pnl
        : Object.entries(json.pnl).map(([date, value]) => ({
            date,
            pnl: value,
          }));

      setData({ ...json, pnl });
    } catch (e: any) {
      console.error("UserWeights failed", e);
      setErr(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  const inputs: ParamInput[] = [
    {
      label: "Start",
      type: "date",
      value: start,
      onChange: (v) => setStart(v.target.value),
    },
    {
      label: "End",
      type: "date",
      value: end,
      onChange: (v) => setEnd(v.target.value),
    },
    {
      label: "Leverage",
      type: "number",
      step: 0.1,
      min: 0,
      value: lev,
      onChange: (v) => setLev(Number(v.target.value || 1)),
    },
    {
      label: "Interest %",
      type: "number",
      step: 0.1,
      min: 0,
      value: intRate,
      onChange: (v) => setIntRate(Number(v.target.value || 0)),
    },
  ];

  return (
    <div className="p-6 text-gray-100 bg-gray-950 min-h-screen">
      <OptimizerNav />
      <h2 className="text-xl font-semibold mb-4 text-teal-400">
        User Weights Backtest
      </h2>

      <div className="flex flex-col gap-6 mb-6">
        <AssetPicker onChange={setSelectedTickers} />

        {selectedTickers.length > 0 && (
          <div className="userweights-container">
            <h3 className="userweights-title">Asset Weights</h3>
            {selectedTickers.map((t) => (
              <div key={t} className="userweights-row">
                <span className="userweights-label">{t}</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={weights[t] ?? 0}
                  onChange={(e) =>
                    updateWeight(t, parseFloat(e.target.value) || 0)
                  }
                  className="userweights-slider"
                  style={{
                    background: `linear-gradient(90deg, #2dd4bf ${
                      (weights[t] ?? 0) * 100
                    }%, #1e293b ${(weights[t] ?? 0) * 100}%)`,
                  }}
                />
                <span className="userweights-value">
                  {(weights[t] * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Parameters */}
        <div className="flex flex-wrap gap-x-6 gap-y-4 items-end">
          {inputs.map((input, idx) => (
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

          <button
            onClick={run}
            disabled={loading}
            className={`px-5 py-2 rounded-md font-medium text-white whitespace-nowrap ${
              loading
                ? "bg-teal-800 cursor-wait"
                : "bg-teal-600 hover:bg-teal-500"
            }`}
          >
            {loading ? "Running..." : "Run Backtest"}
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
