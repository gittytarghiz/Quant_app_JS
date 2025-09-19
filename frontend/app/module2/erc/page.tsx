"use client";

import { useState, useMemo } from "react";
import { LineChart } from "../../../components/LineChart";
import { StatsGrid } from "../../../components/StatsGrid";
import { WeightsTable } from "../../../components/WeightsTable";
import { deriveFromPnl } from "../../../lib/analytics";
import { OptimizerNav } from "../../../components/OptimizerNav";

type Resp = { weights: Array<Record<string, string | number | null>>; pnl: Array<{ date: string; pnl: number | null }>; details: any };

export default function ERCPage() {
  const [tickers, setTickers] = useState("AMZN, AAPL");
  const [start, setStart] = useState("2020-01-01");
  const [end, setEnd] = useState("2025-01-01");
  const [loading, setLoading] = useState(false);
  const [lev, setLev] = useState(1);
  const [minW, setMinW] = useState(0);
  const [maxW, setMaxW] = useState(1);
  const [data, setData] = useState<Resp | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setLoading(true); setErr(null);
    try {
      const res = await fetch(`/opt/erc`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tickers: tickers.split(',').map(s => s.trim()).filter(Boolean), start, end, leverage: Number(lev), min_weight: Number(minW), max_weight: Number(maxW) })
      });
      if (!res.ok) {
        let msg = `HTTP ${res.status} ${res.statusText}`;
        try { const err = await res.json(); if (err?.detail) msg += ` — ${err.detail}`; } catch {}
        throw new Error(msg);
      }
      setData(await res.json());
    } catch (e: any) { setErr(e.message || String(e)); } finally { setLoading(false); }
  }

  const { equityChart, stats } = useMemo(() => deriveFromPnl(data?.pnl), [data]);

  return (
    <div>
      <OptimizerNav />
      <h2>ERC (Equal Risk Contribution)</h2>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        <label>Tickers <input value={tickers} onChange={e => setTickers(e.target.value)} style={{ width: 420 }} /></label>
        <label>Start <input type="date" value={start} onChange={e => setStart(e.target.value)} /></label>
        <label>End <input type="date" value={end} onChange={e => setEnd(e.target.value)} /></label>
        <label>Leverage <input type="number" step={0.1} min={0} value={lev} onChange={e => setLev(Number(e.target.value||1))} style={{ width: 90 }} /></label>
        <label>Min W <input type="number" step={0.01} min={0} max={1} value={minW} onChange={e => setMinW(Number(e.target.value||0))} style={{ width: 90 }} /></label>
        <label>Max W <input type="number" step={0.01} min={0} max={1} value={maxW} onChange={e => setMaxW(Number(e.target.value||1))} style={{ width: 90 }} /></label>
        <button onClick={run} disabled={loading}>Run API</button>
      </div>
      {loading && <p>Loading…</p>}
      {err && <p style={{ color: 'crimson' }}>Error: {err}</p>}

      {data && (
        <div style={{ marginTop: 16 }}>
          <h3>PNL</h3>
          <LineChart data={data.pnl as any} />
          {equityChart.length > 0 && (<>
            <h3 style={{ marginTop: 16 }}>Equity Curve</h3>
            <LineChart data={equityChart} />
          </>)}
          <StatsGrid stats={stats as any} metrics={(data as any)?.details?.metrics || null} />
          <h3 style={{ marginTop: 16 }}>Weights (first 20 rows)</h3>
          <WeightsTable rows={data.weights || []} />
        </div>
      )}
    </div>
  );
}
