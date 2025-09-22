"use client";

import { useState } from "react";

type Resp = { records: Array<Record<string, any>>; columns: string[]; meta: any };

export default function DataTestPage() {
  const [tickers, setTickers] = useState("AMZN, AAPL");
  const [start, setStart] = useState("2013-01-01");
  const [end, setEnd] = useState("2024-01-01");
  const [dtype, setDtype] = useState("close");
  const [interval, setInterval] = useState("1d");
  const [out, setOut] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const API = (process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000").replace(/\/$/, "");

  async function ping() {
    setLoading(true); setOut("");
    try {
      const url = `${API}/data/prices?tickers=${encodeURIComponent(tickers)}&start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}&dtype=${encodeURIComponent(dtype)}&interval=${encodeURIComponent(interval)}`;
      const r = await fetch(url, { method: 'GET' });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j: Resp = await r.json();
      setOut(JSON.stringify({ meta: j.meta, first: j.records?.slice(0, 5) }, null, 2));
    } catch (e: any) {
      setOut(String(e?.message || e));
    } finally { setLoading(false); }
  }

  return (
    <div style={{ padding: 16 }}>
      <h2>Data Fetch Test</h2>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        <label>Tickers <input value={tickers} onChange={e => setTickers(e.target.value)} style={{ width: 260 }} /></label>
        <label>Start <input type="date" value={start} onChange={e => setStart(e.target.value)} /></label>
        <label>End <input type="date" value={end} onChange={e => setEnd(e.target.value)} /></label>
        <label>Type <select value={dtype} onChange={e => setDtype(e.target.value)}>
          <option value="open">open</option>
          <option value="high">high</option>
          <option value="low">low</option>
          <option value="close">close</option>
          <option value="volume">volume</option>
        </select></label>
        <label>Interval <select value={interval} onChange={e => setInterval(e.target.value)}>
          <option value="1d">1d</option>
          <option value="1wk">1wk</option>
          <option value="1mo">1mo</option>
        </select></label>
        <button onClick={ping} disabled={loading}>Fetch</button>
      </div>
      <pre style={{ marginTop: 12, background: '#0b0f14', color: '#c9d1d9', padding: 12, borderRadius: 6, maxHeight: 420, overflow: 'auto' }}>{out}</pre>
      <p style={{ color: '#8899aa' }}>API: {API}/data/prices</p>
    </div>
  );
}

