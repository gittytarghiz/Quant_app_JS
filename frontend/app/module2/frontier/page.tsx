"use client";

import { useState } from "react";
import { OptimizerNav } from "../../../lib/components/OptimizerNav";
import { Analytics } from "../../../lib/analytics";

type Resp = {
  weights: Array<Record<string, string | number | null>>;
  pnl: Array<{ date: string; pnl: number | null }>;
  details: {
    targets: number[];
    points: Array<{ idx: number; target: number; metrics: Record<string, any> }>;
    best_idx: number | null;
  };
};

const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function FrontierPage() {
  const [tickers, setTickers] = useState("AMZN, AAPL");
  const [start, setStart] = useState("2020-01-01");
  const [end, setEnd] = useState("2025-01-01");
  const [n, setN] = useState(15);
  const [covEst, setCovEst] = useState("sample");
  const [covShrink, setCovShrink] = useState(0);
  const [minW, setMinW] = useState(0);
  const [maxW, setMaxW] = useState(1);
  const [lev, setLev] = useState(1);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [data, setData] = useState<Resp | null>(null);

  async function run() {
    setLoading(true);
    setErr(null);
    try {
      const body = {
        tickers: tickers.split(",").map((s) => s.trim()).filter(Boolean),
        start,
        end,
        n_points: Number(n),
        cov_estimator: covEst,
        cov_shrinkage: Number(covShrink),
        min_weight: Number(minW),
        max_weight: Number(maxW),
        leverage: Number(lev),
      };
      const url = `${API}/opt/frontier`;
      const res = await fetch(url, {
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
      const json = await res.json();
      // Wrap into Analytics contract
      setData({
        weights: json.weights ?? [],
        pnl: json.pnl ?? [],
        details: {
          targets: json.targets ?? [],
          points: json.points ?? [],
          best_idx: json.best_idx ?? null,
        },
      });
    } catch (e: any) {
      setErr(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <OptimizerNav />
      <h2>Efficient Frontier (Target Return Sweep)</h2>

      <div
        style={{
          display: "flex",
          gap: 8,
          flexWrap: "wrap",
          alignItems: "center",
          marginBottom: 16,
        }}
      >
        {/* inputs same as before */}
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
          <input type="date" value={start} onChange={(e) => setStart(e.target.value)} />
        </label>
        <label>
          End{" "}
          <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
        </label>
        <label>
          N points{" "}
          <input
            type="number"
            min={3}
            max={60}
            value={n}
            onChange={(e) => setN(Number(e.target.value || 15))}
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
        <label>
          Cov shrink{" "}
          <input
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
        <button onClick={run} disabled={loading}>
          Build Frontier
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
