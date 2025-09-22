"use client";

import { useState, useMemo } from "react";
import { LineChart } from "../../../components/LineChart";
import { StatsGrid } from "../../../components/StatsGrid";
import { WeightsTable } from "../../../components/WeightsTable";
import { deriveFromPnl } from "../../../lib/analytics";
import { OptimizerNav } from "../../../components/OptimizerNav";

type Resp = {
  weights: Array<Record<string, string | number | null>>;
  pnl: Array<{ date: string; pnl: number | null }>;
  details: any;
};

function parseWeights(input: string): Record<string, number> {
  const out: Record<string, number> = {};
  input.split(",").forEach(part => {
    const [k, v] = part.split(":").map(s => s.trim());
    if (k && v && !isNaN(Number(v))) out[k.toUpperCase()] = Number(v);
  });
  return out;
}

const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function UserWeightsPage() {
  const [tickers, setTickers] = useState("AMZN, AAPL");
  const [start, setStart] = useState("2020-01-01");
  const [end, setEnd] = useState("2025-01-01");
  const [weights, setWeights] = useState(
    "AAPL:0.125, MSFT:0.125, GOOGL:0.125, META:0.125, JNJ:0.125, GC=F:0.125, AMZN:0.125, NVDA:0.125"
  );
  const [loading, setLoading] = useState(false);
  const [lev, setLev] = useState(1);
  const [data, setData] = useState<Resp | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setLoading(true);
    setErr(null);
    try {
      const res = await fetch(`${API}/opt/user-weights`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers: tickers
            .split(",")
            .map(s => s.trim().toUpperCase())
            .filter(Boolean),
          start,
          end,
          static_weights: parseWeights(weights),
          normalize: true,
          leverage: Number(lev),
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

  const { equityChart, stats } = useMemo(
    () => deriveFromPnl(data?.pnl),
    [data]
  );

  return (
    <div>
      <OptimizerNav />
      <h2>User Weights Tester</h2>
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
            onChange={e => setTickers(e.target.value)}
            style={{ width: 420 }}
          />
        </label>
        <label>
          Start{" "}
          <input
            type="date"
            value={start}
            onChange={e => setStart(e.target.value)}
          />
        </label>
        <label>
          End{" "}
          <input
            type="date"
            value={end}
            onChange={e => setEnd(e.target.value)}
          />
        </label>
      </div>
      <div style={{ marginTop: 8 }}>
        <label>
          Static Weights{" "}
          <input
            value={weights}
            onChange={e => setWeights(e.target.value)}
            style={{ width: 600 }}
          />
        </label>
        <div style={{ fontSize: 12, color: "#666" }}>
          Format: TICKER:weight comma-separated (e.g., AAPL:0.25, MSFT:0.75)
        </div>
      </div>
      <div style={{ marginTop: 8 }}>
        <label>
          Leverage{" "}
          <input
            type="number"
            step={0.1}
            min={0}
            value={lev}
            onChange={e => setLev(Number(e.target.value || 1))}
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
          <h3>PNL</h3>
          <LineChart data={data.pnl as any} />
          {equityChart.length > 0 && (
            <>
              <h3 style={{ marginTop: 16 }}>Equity Curve</h3>
              <LineChart data={equityChart} />
            </>
          )}
          <StatsGrid
            stats={stats as any}
            metrics={(data as any)?.details?.metrics || null}
          />
          <h3 style={{ marginTop: 16 }}>Weights (first 20 rows)</h3>
          <WeightsTable rows={data.weights || []} />
        </div>
      )}
    </div>
  );
}
