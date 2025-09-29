"use client";
import { useState } from "react";
import { LineChart } from "../../lib/components/LineChart";

export default function Module1Page() {
  const [tickers, setTickers] = useState("AMZN, AAPL");
  const [start, setStart] = useState("2013-01-01");
  const [end, setEnd] = useState("2025-01-01");
  const [dtype, setDtype] = useState("close");
  const [interval, setInterval] = useState("1d");
  const [loading, setLoading] = useState(false);
  const [pnls, setPnls] = useState<Record<string, Array<{ date: string; pnl: number | null }>>>({});
  const [raw, setRaw] = useState<any[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const API = (process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000").replace(/\/$/, "");

  async function fetchData() {
    setLoading(true); setErr(null);
    try {
      const res = await fetch(`${API}/data/prices`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers: tickers.split(",").map(s => s.trim()).filter(Boolean),
          start, end, dtype, interval
        })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      const rows = json.records || [];
      const cols = (json.columns || []).filter((c: string) => c !== "date");
      setRaw(rows);

      // build pnl series for each ticker
      const out: Record<string, Array<{ date: string; pnl: number | null }>> = {};
      for (const col of cols) {
        const series: Array<{ date: string; pnl: number | null }> = [];
        let prev: number | null = null;
        for (const r of rows) {
          const price = Number(r[col]);
          const change = prev != null && price ? price / prev - 1 : null;
          series.push({ date: String(r.date), pnl: change });
          prev = price;
        }
        out[col] = series;
      }
      setPnls(out);
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  }

  function downloadCsv() {
    if (!raw.length) return;
    const cols = Object.keys(raw[0]);
    const header = cols.join(",");
    const lines = raw.map(r => cols.map(c => String(r[c] ?? "")).join(","));
    const csv = [header, ...lines].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `prices_${dtype}_${interval}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold">Module 1 — Data Downloader</h2>

      <div className="flex flex-wrap gap-2 items-center">
        <input value={tickers} onChange={e => setTickers(e.target.value)} className="px-2 py-1 rounded bg-gray-800" />
        <input type="date" value={start} onChange={e => setStart(e.target.value)} className="px-2 py-1 rounded bg-gray-800" />
        <input type="date" value={end} onChange={e => setEnd(e.target.value)} className="px-2 py-1 rounded bg-gray-800" />
        <select value={dtype} onChange={e => setDtype(e.target.value)} className="px-2 py-1 rounded bg-gray-800">
          {["open","high","low","close","volume"].map(x => <option key={x} value={x}>{x}</option>)}
        </select>
        <select value={interval} onChange={e => setInterval(e.target.value)} className="px-2 py-1 rounded bg-gray-800">
          {["1d","1wk","1mo"].map(x => <option key={x} value={x}>{x}</option>)}
        </select>
        <button onClick={fetchData} disabled={loading} className="bg-blue-600 px-3 py-1 rounded">
          {loading ? "Loading…" : "Fetch"}
        </button>
        <button onClick={downloadCsv} disabled={!raw.length} className="bg-green-600 px-3 py-1 rounded">
          Download CSV
        </button>
      </div>

      {err && <p className="text-red-400">Error: {err}</p>}

      {Object.entries(pnls).map(([ticker, pnl]) => (
        <div key={ticker}>
          <h3 className="text-sm text-gray-300 mb-1">{ticker}</h3>
          <LineChart pnl={pnl} />
        </div>
      ))}
    </div>
  );
}
