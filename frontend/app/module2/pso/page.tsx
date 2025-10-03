"use client";

import { useState } from "react";
import { OptimizerNav } from "../../../lib/components/OptimizerNav";
import { Analytics } from "../../../lib/analytics";

type Resp = {
  weights: Array<Record<string, string | number | null>>;
  pnl: Array<{ date: string; pnl: number | null }>;
  details: any;
};

const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function PSOPage() {
  const [tickers, setTickers] = useState("AMZN, AAPL, GC=F,NVDA,JNJ");
  const [start, setStart] = useState("2020-01-01");
  const [end, setEnd] = useState("2025-01-01");
  const [objective, setObjective] = useState("sharpe");
  const [lev, setLev] = useState(1);
  const [minW, setMinW] = useState(0);
  const [maxW, setMaxW] = useState(1);
  const [intRate, setIntRate] = useState(4.0); // NEW: interest rate (%)
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<Resp | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setLoading(true);
    setErr(null);
    try {
      const body = {
        tickers: tickers.split(",").map((s) => s.trim()).filter(Boolean),
        start,
        end,
        objective,
        leverage: Number(lev),
        min_weight: Number(minW),
        max_weight: Number(maxW),
        interest_rate: intRate / 100.0, // NEW: convert % → decimal
      };

      const res = await fetch(`${API}/opt/pso`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        let msg = `HTTP ${res.status} ${res.statusText}`;
        try {
          const err = await res.json();
          if (err?.detail) msg += ` — ${err.detail}`;
        } catch {}
        throw new Error(msg);
      }
      setData(await res.json());
    } catch (e: any) {
      setErr(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <OptimizerNav />
      <h2>PSO Optimizer</h2>
      <div
        style={{
          display: "flex",
          gap: 8,
          flexWrap: "wrap",
          alignItems: "center",
          marginBottom: 16,
        }}
      >
        <label>
          Tickers{" "}
          <input
            value={tickers}
            onChange={(e) => setTickers(e.target.value)}
            style={{ width: 280 }}
          />
        </label>
        <label>
          Start{" "}
          <input
            type="date"
            value={start}
            onChange={(e) => setStart(e.target.value)}
          />
        </label>
        <label>
          End{" "}
          <input
            type="date"
            value={end}
            onChange={(e) => setEnd(e.target.value)}
          />
        </label>
        <label>
          Objective{" "}
          <select value={objective} onChange={(e) => setObjective(e.target.value)}>
            <option value="sharpe">sharpe</option>
            <option value="sortino">sortino</option>
            <option value="calmar">calmar</option>
            <option value="cvar">cvar</option>
            <option value="ret_turnover">ret_turnover</option>
            <option value="ret_drawdown">ret_drawdown</option>
            <option value="kelly">kelly</option>
            <option value="diversification">diversification</option>
          </select>
        </label>
        <label>
          Leverage{" "}
          <input
            type="number"
            step={0.1}
            min={0}
            value={lev}
            onChange={(e) => setLev(Number(e.target.value || 1))}
            style={{ width: 90 }}
          />
        </label>
        <label>
          Interest %{" "}
          <input
            type="number"
            step={0.1}
            min={0}
            value={intRate}
            onChange={(e) => setIntRate(Number(e.target.value || 0))}
            style={{ width: 90 }}
          />
        </label>
        <label>
          Min W{" "}
          <input
            type="number"
            step={0.01}
            min={0}
            max={1}
            value={minW}
            onChange={(e) => setMinW(Number(e.target.value || 0))}
            style={{ width: 90 }}
          />
        </label>
        <label>
          Max W{" "}
          <input
            type="number"
            step={0.01}
            min={0}
            max={1}
            value={maxW}
            onChange={(e) => setMaxW(Number(e.target.value || 1))}
            style={{ width: 90 }}
          />
        </label>
        <button onClick={run} disabled={loading}>
          Run API
        </button>
      </div>
      {loading && <p>Loading…</p>}
      {err && <p style={{ color: "crimson" }}>Error: {err}</p>}
      {data && (
        <div style={{ marginTop: 16 }}>
          <Analytics data={data} />
        </div>
      )}
    </div>
  );
}
