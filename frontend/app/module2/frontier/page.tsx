"use client";

import { useMemo, useState } from "react";
import { OptimizerNav } from "../../../components/OptimizerNav";

type Point = { idx: number; target: number; metrics: Record<string, any> };
type Resp = { targets: number[]; points: Point[]; best_idx: number | null };

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
    setLoading(true); setErr(null);
    try {
      const res = await fetch(`/opt/frontier`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tickers: tickers.split(',').map(s => s.trim()).filter(Boolean),
          start, end, n_points: Number(n), cov_estimator: covEst, cov_shrinkage: Number(covShrink),
          min_weight: Number(minW), max_weight: Number(maxW), leverage: Number(lev)
        })
      });
      if (!res.ok) {
        let msg = `HTTP ${res.status} ${res.statusText}`;
        try { const err = await res.json(); if (err?.detail) msg += ` — ${err.detail}`; } catch {}
        throw new Error(msg);
      }
      setData(await res.json());
    } catch (e: any) { setErr(e.message || String(e)); } finally { setLoading(false); }
  }

  const rows = useMemo(() => {
    const pts = data?.points || [];
    const mapped = pts.map(p => ({
      idx: p.idx,
      target: p.target,
      ann_return: p.metrics?.ann_return ?? null,
      ann_vol: p.metrics?.ann_vol ?? null,
      sharpe: p.metrics?.sharpe ?? null,
      cagr: p.metrics?.cagr ?? null,
    }));
    // Sort rows by annualized volatility
    const sortedRows = mapped.sort((a, b) => (a.ann_vol ?? 0) - (b.ann_vol ?? 0));
    // Determine best row based on highest sharpe
    const bestRow = sortedRows.reduce((best, row) => {
      return (row.sharpe !== null && row.sharpe > (best.sharpe ?? -Infinity)) ? row : best;
    }, sortedRows[0] || {});
    // Attach a flag to indicate best row
    return sortedRows.map(r => ({ ...r, best: r.idx === bestRow.idx }));
  }, [data]);

  return (
    <div>
      <OptimizerNav />
      <h2>Efficient Frontier (Target Return Sweep)</h2>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        <label>Tickers <input value={tickers} onChange={e => setTickers(e.target.value)} style={{ width: 420 }} /></label>
        <label>Start <input type="date" value={start} onChange={e => setStart(e.target.value)} /></label>
        <label>End <input type="date" value={end} onChange={e => setEnd(e.target.value)} /></label>
        <label>N points <input type="number" min={3} max={60} value={n} onChange={e => setN(Number(e.target.value||15))} style={{ width: 90 }} /></label>
        <label>Cov Estimator
          <select value={covEst} onChange={e => setCovEst(e.target.value)}>
            <option value="sample">sample</option>
            <option value="diag">diag</option>
            <option value="lw">lw</option>
          </select>
        </label>
        <label>Cov shrink <input type="number" step={0.05} min={0} max={1} value={covShrink} onChange={e => setCovShrink(Number(e.target.value||0))} style={{ width: 110 }} /></label>
        <label>Leverage <input type="number" step={0.1} min={0} value={lev} onChange={e => setLev(Number(e.target.value||1))} style={{ width: 90 }} /></label>
        <label>Min W <input type="number" step={0.01} min={0} max={1} value={minW} onChange={e => setMinW(Number(e.target.value||0))} style={{ width: 90 }} /></label>
        <label>Max W <input type="number" step={0.01} min={0} max={1} value={maxW} onChange={e => setMaxW(Number(e.target.value||1))} style={{ width: 90 }} /></label>
        <button onClick={run} disabled={loading}>Build Frontier</button>
      </div>
      {loading && <p>Loading…</p>}
      {err && <p style={{ color: 'crimson' }}>Error: {err}</p>}

      {rows.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <h3>Frontier Points (sorted by risk)</h3>
          <table className="table">
            <thead><tr><th>#</th><th>Target</th><th>Ann Vol</th><th>Ann Return</th><th>Sharpe</th><th>CAGR</th></tr></thead>
            <tbody>
              {rows.map(r => (
                <tr key={r.idx} style={{ fontWeight: r.best ? 600 : 400 }}>
                  <td>{r.idx}</td>
                  <td>{r.target?.toFixed(6)}</td>
                  <td>{r.ann_vol != null ? r.ann_vol.toFixed(4) : '-'}</td>
                  <td>{r.ann_return != null ? r.ann_return.toFixed(4) : '-'}</td>
                  <td>{r.sharpe != null ? r.sharpe.toFixed(3) : '-'}</td>
                  <td>{r.cagr != null ? r.cagr.toFixed(4) : '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted">Bold row = highest Sharpe across the sampled frontier.</p>
        </div>
      )}
    </div>
  );
}
