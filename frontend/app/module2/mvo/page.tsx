"use client";

import { useState, useMemo } from "react";
import { LineChart } from "../../../components/LineChart";
import { StatsGrid } from "../../../components/StatsGrid";
import { WeightsTable } from "../../../components/WeightsTable";
import { deriveFromPnl } from "../../../lib/analytics";
import { OptimizerNav } from "../../../components/OptimizerNav";

type MvoResponse = {
  weights: Array<Record<string, string | number | null>>;
  pnl: Array<{ date: string; pnl: number | null }>;
  details: Record<string, any>;
};

const API = process.env.NEXT_PUBLIC_API_BASE_URL || ""; // same-origin if empty

export default function MvoPage() {
  const [tickers, setTickers] = useState("AMZN, AAPL");
  const [start, setStart] = useState("2020-01-01");
  const [end, setEnd] = useState("2025-01-01");
  const [objective, setObjective] = useState("min_vol");
  const [l2Lambda, setL2Lambda] = useState(0.001);
  const [covShrink, setCovShrink] = useState(0);
  const [covEst, setCovEst] = useState("sample");
  const [loading, setLoading] = useState(false);
  const [lev, setLev] = useState(1);
  const [minW, setMinW] = useState(0);
  const [maxW, setMaxW] = useState(1);
  const [data, setData] = useState<MvoResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setLoading(true); setErr(null);
    try {
      const body: any = {
        tickers: tickers.split(',').map(s => s.trim()).filter(Boolean),
        start, end, objective, leverage: Number(lev),
        min_weight: Number(minW), max_weight: Number(maxW)
      };
      // Provide objective_params for supported objectives
      const op: Record<string, number> = {};
      if (objective === 'min_vol_l2') op.l2_lambda = Number(l2Lambda);
      if (covShrink && covShrink > 0) op.cov_shrinkage = Number(covShrink);
      if (covEst && covEst !== 'sample') op.cov_estimator = covEst as any;
      if (Object.keys(op).length) body.objective_params = op;

      const url = `${API}/opt/mvo`;
      console.log('POST', url, body);
      const res = await fetch(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (!res.ok) {
        let msg = `HTTP ${res.status} ${res.statusText}`;
        try { const err = await res.json(); if (err?.detail) msg += ` — ${err.detail}`; } catch {}
        throw new Error(msg);
      }
      const json = await res.json();
      console.log('OK /opt/mvo', json?.details?.metrics, (json?.pnl||[]).length);
      setData(json);
    } catch (e: any) { console.error('MVO failed', e); setErr(e.message || String(e)); } finally { setLoading(false); }
  }

  const { equityChart, stats } = useMemo(() => deriveFromPnl(data?.pnl), [data]);

  return (
    <div>
      <OptimizerNav />
      <h2>MVO Tester</h2>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        <label>Tickers <input value={tickers} onChange={e => setTickers(e.target.value)} style={{ width: 280 }} /></label>
        <label>Start <input type="date" value={start} onChange={e => setStart(e.target.value)} /></label>
        <label>End <input type="date" value={end} onChange={e => setEnd(e.target.value)} /></label>
        <label>Objective
          <select value={objective} onChange={e => setObjective(e.target.value)}>
            <option value="min_vol">min_vol</option>
            <option value="max_return">max_return</option>
            <option value="min_vol_l2">min_vol_l2</option>
            <option value="sharpe">sharpe</option>
            <option value="sortino">sortino</option>
            <option value="calmar">calmar</option>
            <option value="cvar">cvar</option>
            <option value="kelly">kelly</option>
            <option value="diversification">diversification</option>
            <option value="return_to_turnover">return_to_turnover</option>
            <option value="return_to_drawdown">return_to_drawdown</option>
          </select>
        </label>
        {objective === 'min_vol_l2' && (
          <label>L2 lambda <input type="number" step={0.0001} min={0} value={l2Lambda} onChange={e => setL2Lambda(Number(e.target.value||0.001))} style={{ width: 110 }} /></label>
        )}
        <label>Cov shrink <input title="Diagonal shrinkage 0..1" type="number" step={0.05} min={0} max={1} value={covShrink}
          onChange={e => setCovShrink(Number(e.target.value||0))} style={{ width: 110 }} /></label>
        <label>Cov Estimator
          <select value={covEst} onChange={e => setCovEst(e.target.value)}>
            <option value="sample">sample</option>
            <option value="diag">diag</option>
            <option value="lw">lw</option>
          </select>
        </label>
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
