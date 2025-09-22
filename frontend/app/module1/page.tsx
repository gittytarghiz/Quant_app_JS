"use client";

import { useState } from "react";
import { WeightsTable } from "../../components/WeightsTable";

type Resp = { records: Array<Record<string, string | number | null>>; columns: string[]; meta: any };

export default function Module1Page() {
  const [tickers, setTickers] = useState("AMZN, AAPL");
  const [start, setStart] = useState("2013-01-01");
  const [end, setEnd] = useState("2025-01-01");
  const [dtype, setDtype] = useState("close");
  const [interval, setInterval] = useState("1d");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<Resp | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const API = (process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000").replace(/\/$/, "");

  async function fetchData() {
    setLoading(true); setErr(null);
    try {
      const ctrl = new AbortController();
      const tid = setTimeout(() => ctrl.abort(), 20000);
      const url = `${API}/data/prices`;
      const payload = {
        tickers: tickers.split(',').map(s => s.trim()).filter(Boolean),
        start, end, dtype, interval
      };
      console.log("POST", url, payload);
      const res = await fetch(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        signal: ctrl.signal,
        body: JSON.stringify(payload)
      });
      clearTimeout(tid);
      if (!res.ok) {
        let msg = `HTTP ${res.status} ${res.statusText}`;
        try { const err = await res.json(); if (err?.detail) msg += ` — ${err.detail}`; } catch {}
        throw new Error(msg);
      }
      const json = await res.json();
      console.log("OK /data/prices", json?.meta, json?.columns?.length, json?.records?.length);
      setData(json);
    } catch (e: any) {
      const msg = e?.name === 'AbortError' ? 'Request timed out' : (e.message || String(e));
      console.error("Fetch failed:", msg, e);
      setErr(msg);
    } finally {
      setLoading(false);
    }
  }

  function downloadCsv() {
    const rows = data?.records || [];
    if (!rows.length) return;
    const cols = Object.keys(rows[0]);
    const header = cols.join(",");
    const lines = rows.map(r => cols.map(c => String(r[c] ?? "")).join(","));
    const csv = [header, ...lines].join("\n");
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `prices_${dtype}_${interval}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div>
      <h2>Module 1 — Data Downloader</h2>
      <div className="row">
        <label>Tickers <input value={tickers} onChange={e => setTickers(e.target.value)} style={{ width: 360 }} /></label>
        <label>Start <input type="date" value={start} onChange={e => setStart(e.target.value)} /></label>
        <label>End <input type="date" value={end} onChange={e => setEnd(e.target.value)} /></label>
        <label>Type
          <select value={dtype} onChange={e => setDtype(e.target.value)}>
            <option value="open">open</option>
            <option value="high">high</option>
            <option value="low">low</option>
            <option value="close">close</option>
            <option value="volume">volume</option>
          </select>
        </label>
        <label>Interval
          <select value={interval} onChange={e => setInterval(e.target.value)}>
            <option value="1d">1d</option>
            <option value="1wk">1wk</option>
            <option value="1mo">1mo</option>
          </select>
        </label>
        <button onClick={fetchData} disabled={loading}>Fetch</button>
        <button onClick={downloadCsv} disabled={loading || !(data?.records?.length)}>Download CSV</button>
      </div>
      {loading && <p>Loading…</p>}
      {err && <p style={{ color: 'crimson' }}>Error: {err}</p>}
      {!!data?.records?.length && (
        <div className="spaced">
          <div style={{ color: '#9aa2b1', marginBottom: 8 }}>
            Rows: {data.meta?.count} — Columns: {Array.isArray(data.columns) ? data.columns.join(', ') : ''}
          </div>
          <WeightsTable rows={data.records} />
        </div>
      )}
    </div>
  );
}
