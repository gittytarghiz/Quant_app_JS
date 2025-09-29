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

export default function MVOTargetPage() {
  const [tickers, setTickers] = useState("AMZN, AAPL");
  const [start, setStart] = useState("2020-01-01");
  const [end, setEnd] = useState("2025-01-01");
  const [loading, setLoading] = useState(false);
  const [lev, setLev] = useState(1);
  const [minW, setMinW] = useState(0);
  const [maxW, setMaxW] = useState(1);
  const [target, setTarget] = useState(0.001);
  const [covShrink, setCovShrink] = useState(0);
  const [data, setData] = useState<Resp | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [covEst, setCovEst] = useState("sample");

  async function run() {
    setLoading(true);
    setErr(null);
    try {
      const res = await fetch(`${API}/opt/mvo-target-return`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers: tickers.split(",").map((s) => s.trim()).filter(Boolean),
          start,
          end,
          target_return: Number(target),
          cov_shrinkage: Number(covShrink),
          cov_estimator: covEst,
          leverage: Number(lev),
          min_weight: Number(minW),
          max_weight: Number(maxW),
        }),
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
      <h2>MVO — Target Return Min-Vol</h2>
      <div
        style={{
          display: "flex",
          gap: 8,
          flexWrap: "wrap",
          alignItems: "center",
        }}
      >
        <label>
          Tickers{" "}
          <input
            value={tickers}
            onChange={(e) => setTickers(e.target.value)}
            style={{ width: 420 }}
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
          Target return{" "}
          <input
            type="number"
            step={0.0001}
            value={target}
            onChange={(e) => setTarget(Number(e.target.value || 0.0))}
            style={{ width: 120 }}
          />
        </label>
        <label>
          Cov shrink{" "}
          <input
            title="Diagonal shrinkage 0..1"
            type="number"
            step={0.05}
            min={0}
            max={1}
            value={covShrink}
            onChange={(e) => setCovShrink(Number(e.target.value || 0))}
            style={{ width: 110 }}
          />
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
        <label>
          Cov Estimator
          <select value={covEst} onChange={(e) => setCovEst(e.target.value)}>
            <option value="sample">sample</option>
            <option value="diag">diag</option>
            <option value="lw">lw</option>
          </select>
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
