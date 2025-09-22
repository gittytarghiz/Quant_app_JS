"use client";

import { useState, useMemo } from "react";
import { LineChart } from "../../../components/LineChart";
import { StatsGrid } from "../../../components/StatsGrid";
import { WeightsTable } from "../../../components/WeightsTable";
import { deriveFromPnl } from "../../../lib/analytics";
import { OptimizerNav } from "../../../components/OptimizerNav";

type Resp = { weights: Array<Record<string, string | number | null>>; pnl: Array<{ date: string; pnl: number | null }>; details: any };

const API = (process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000");

export default function GAPage() {
  const [tickers, setTickers] = useState("AMZN, AAPL");
  const [start, setStart] = useState("2020-01-01");
  const [end, setEnd] = useState("2025-01-01");
  const [objective, setObjective] = useState("sharpe");
  const [loading, setLoading] = useState(false);
  const [lev, setLev] = useState(1);
  const [minW, setMinW] = useState(0);
  const [maxW, setMaxW] = useState(1);
  const [interval, setInterval] = useState("1d");
  const [rebalance, setRebalance] = useState("monthly");
  const [minObs, setMinObs] = useState(60);
  // objective params
  const [rf, setRf] = useState(0);
  const [mar, setMar] = useState(0);
  const [alpha, setAlpha] = useState(0.05);
  // Seed only (hide other advanced knobs)
  const [seed, setSeed] = useState<number | undefined>(42);
  const [data, setData] = useState<Resp | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setLoading(true); setErr(null);
    try {
      const objective_params: Record<string, number> = {};
      if (objective === 'sharpe' && rf) objective_params.rf = Number(rf);
      if (objective === 'sortino' && mar) objective_params.mar = Number(mar);
      if (objective === 'cvar' && alpha) objective_params.alpha = Number(alpha);
      const payload: any = {
        tickers: tickers.split(',').map(s => s.trim()).filter(Boolean),
        start, end, objective,
        interval, rebalance,
        min_weight: Number(minW), max_weight: Number(maxW), min_obs: Number(minObs),
        leverage: Number(lev),
        objective_params,
      };
      if (seed !== undefined && seed !== null) payload.seed = Number(seed);
      const res = await fetch(`${API}/opt/ga`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
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
      <h2>GA (DE) Tester</h2>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        <label>Tickers <input value={tickers} onChange={e => setTickers(e.target.value)} style={{ width: 420 }} /></label>
        <label>Start <input type="date" value={start} onChange={e => setStart(e.target.value)} /></label>
        <label>End <input type="date" value={end} onChange={e => setEnd(e.target.value)} /></label>
        <label>Objective
          <select value={objective} onChange={e => setObjective(e.target.value)}>
            <option value="sharpe">sharpe</option>
            <option value="sortino">sortino</option>
            <option value="calmar">calmar</option>
            <option value="cvar">cvar</option>
            <option value="return_to_turnover">return_to_turnover</option>
            <option value="return_to_drawdown">return_to_drawdown</option>
            <option value="kelly">kelly</option>
            <option value="diversification">diversification</option>
          </select>
        </label>
        {objective === 'sharpe' && (
          <label>rf <input type="number" step={0.0001} value={rf} onChange={e => setRf(Number(e.target.value||0))} style={{ width: 100 }} /></label>
        )}
        {objective === 'sortino' && (
          <label>MAR <input type="number" step={0.0001} value={mar} onChange={e => setMar(Number(e.target.value||0))} style={{ width: 100 }} /></label>
        )}
        {objective === 'cvar' && (
          <label>alpha <input type="number" step={0.01} min={0.01} max={0.5} value={alpha} onChange={e => setAlpha(Number(e.target.value||0.05))} style={{ width: 90 }} /></label>
        )}
        <label>Leverage <input type="number" step={0.1} min={0} value={lev} onChange={e => setLev(Number(e.target.value||1))} style={{ width: 90 }} /></label>
        <label>Min W <input type="number" step={0.01} min={0} max={1} value={minW} onChange={e => setMinW(Number(e.target.value||0))} style={{ width: 90 }} /></label>
        <label>Max W <input type="number" step={0.01} min={0} max={1} value={maxW} onChange={e => setMaxW(Number(e.target.value||1))} style={{ width: 90 }} /></label>
        <label>Interval
          <select value={interval} onChange={e => setInterval(e.target.value)}>
            <option value="1d">1d</option>
            <option value="weekly">weekly</option>
            <option value="monthly">monthly</option>
          </select>
        </label>
        <label>Rebalance
          <select value={rebalance} onChange={e => setRebalance(e.target.value)}>
            <option value="daily">daily</option>
            <option value="weekly">weekly</option>
            <option value="monthly">monthly</option>
            <option value="quarterly">quarterly</option>
          </select>
        </label>
        <label>Min obs <input type="number" min={20} value={minObs} onChange={e => setMinObs(Number(e.target.value||60))} style={{ width: 90 }} /></label>
        <label>Seed <input type="number" value={seed ?? 0} onChange={e => setSeed(Number(e.target.value))} style={{ width: 90 }} /></label>
        <button onClick={run} disabled={loading}>Run API</button>
      </div>
      {/* Advanced knobs removed for simplicity */}
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
