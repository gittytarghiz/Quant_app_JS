"use client";

import { useState, useMemo } from "react";
import { OptimizerNav } from "../../../lib/components/OptimizerNav";
import AssetPicker from "../../../lib/components/AssetPicker";

type Resp = {
  pnl: Array<{ date: string; pnl: number | null }>;
  weights: Array<Record<string, string | number | null>>;
  details: Record<string, any>;
};
type Key =
  | "eqw"
  | "mvo"
  | "minvar"
  | "risk_parity"
  | "pso"
  | "ga"
  | "nsgaii";
type RunResult =
  | { ok: true; ms: number; data: Resp }
  | { ok: false; ms: number; error: string }
  | { ok: null };

const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

const METHODS: { key: Key; label: string; path: string }[] = [
  { key: "eqw", label: "Equal Weight", path: "/opt/equal-weight" },
  { key: "mvo", label: "MVO", path: "/opt/mvo" },
  { key: "minvar", label: "Minimum Variance", path: "/opt/min-variance" },
  { key: "risk_parity", label: "Risk Parity", path: "/opt/risk-parity" },
  { key: "pso", label: "Particle Swarm", path: "/opt/pso" },
  { key: "ga", label: "Genetic Algorithm", path: "/opt/ga" },
  { key: "nsgaii", label: "NSGA-II", path: "/opt/nsga2" },
];

// ✅ Embedded metrics calculator
function StatsGrid({ pnl }: { pnl: Resp["pnl"] }) {
  const metrics = useMemo(() => {
    const ret: number[] = [];
    const dates: string[] = [];
    for (const p of pnl)
      if (p.pnl != null && isFinite(p.pnl)) {
        ret.push(p.pnl);
        dates.push(p.date);
      }
    if (ret.length < 2) return null;

    const mean = (a: number[]) => a.reduce((s, x) => s + x, 0) / a.length;
    const stdev = (a: number[]) => {
      if (a.length < 2) return 0;
      const m = mean(a);
      let v = 0;
      for (const x of a) v += (x - m) ** 2;
      return Math.sqrt(v / (a.length - 1));
    };
    const ms = dates.map((d) => new Date(d).getTime());
    const gaps = ms.slice(1).map((t, i) => Math.max(1, (t - ms[i]) / 86400000));
    const md = gaps.sort((a, b) => a - b)[Math.floor(gaps.length / 2)];
    const AF = md <= 2 ? 252 : md <= 10 ? 52 : md <= 40 ? 12 : 1;

    const mu = mean(ret);
    const sd = stdev(ret);

    let eq = 1, peak = 1, maxDD = 0;
    for (const r of ret) {
      eq *= 1 + r;
      if (eq > peak) peak = eq;
      else maxDD = Math.min(maxDD, eq / peak - 1);
    }
    const first = new Date(dates[0]).getTime();
    const last = new Date(dates.at(-1) || 0).getTime();
    const years = Math.max((last - first) / (365.25 * 86400000), ret.length / AF);
    const total = eq - 1;
    const cagr = years > 0 ? Math.pow(1 + total, 1 / years) - 1 : 0;

    const sharpe = sd ? (mu / sd) * Math.sqrt(AF) : 0;
    const calmar = Math.abs(maxDD) ? cagr / Math.abs(maxDD) : 0;

    return {
      "CAGR": cagr,
      "Ann. Vol": sd * Math.sqrt(AF),
      "Sharpe": sharpe,
      "Max Drawdown": maxDD,
      "Calmar": calmar,
    };
  }, [pnl]);

  if (!metrics) return <div className="text-sm text-gray-400">No data</div>;
  const percent = ["CAGR", "Ann. Vol", "Max Drawdown"];
  const fmt = (k: string, v: number) =>
    percent.includes(k) ? `${(v * 100).toFixed(2)}%` : v.toFixed(2);

  return (
    <div className="grid grid-cols-2 gap-2 text-sm mt-1">
      {Object.entries(metrics).map(([k, v]) => (
        <div
          key={k}
          className={`p-2 rounded-md bg-gray-900/60 border border-gray-800`}
        >
          <div className="text-gray-400">{k}</div>
          <div className={`font-medium ${Number(v) < 0 ? "text-red-400" : "text-teal-400"}`}>
            {fmt(k, v as number)}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function RunAllPage() {
  const [tickers, setTickers] = useState<string[]>([]);
  const [start, setStart] = useState("2020-01-01");
  const [end, setEnd] = useState("2025-01-01");
  const [lev, setLev] = useState(1);
  const [minW, setMinW] = useState(0);
  const [maxW, setMaxW] = useState(1);
  const [intRate, setIntRate] = useState(4.0);
  const [objective, setObjective] = useState("sharpe");
  const [tries, setTries] = useState(64);
  const [seed, setSeed] = useState(42);
  const [results, setResults] = useState<Record<Key, RunResult>>({
    eqw: { ok: null }, mvo: { ok: null }, minvar: { ok: null },
    risk_parity: { ok: null }, pso: { ok: null }, ga: { ok: null }, nsgaii: { ok: null },
  });
  const [err, setErr] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const common = useMemo(() => ({
    tickers, start, end,
    leverage: Number(lev),
    min_weight: Number(minW),
    max_weight: Number(maxW),
    dtype: "close", interval: "1d", rebalance: "monthly",
    interest_rate: intRate / 100.0,
  }), [tickers, start, end, lev, minW, maxW, intRate]);

  async function runOne(key: Key, path: string) {
    const t0 = performance.now();
    const body =
      key === "ga" || key === "pso"
        ? { ...common, objective }
        : key === "nsgaii"
        ? { ...common, tries, seed }
        : { ...common };
    try {
      const res = await fetch(`${API}${path}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const ms = Math.round(performance.now() - t0);
      if (!res.ok) {
        let msg = `HTTP ${res.status} ${res.statusText}`;
        try { const j = await res.json(); if (j?.detail) msg += ` — ${j.detail}`; } catch {}
        setResults((p) => ({ ...p, [key]: { ok: false, ms, error: msg } }));
      } else {
        const data: Resp = await res.json();
        setResults((p) => ({ ...p, [key]: { ok: true, ms, data } }));
      }
    } catch (e: any) {
      const ms = Math.round(performance.now() - t0);
      setResults((p) => ({
        ...p,
        [key]: { ok: false, ms, error: String(e?.message ?? e) },
      }));
    }
  }

  async function onRunAll() {
    if (!tickers.length) { setErr("Select at least one asset."); return; }
    setErr(null); setRunning(true);
    setResults({
      eqw:{ok:null}, mvo:{ok:null}, minvar:{ok:null},
      risk_parity:{ok:null}, pso:{ok:null}, ga:{ok:null}, nsgaii:{ok:null},
    });
    await Promise.all(METHODS.map(({ key, path }) => runOne(key, path)));
    setRunning(false);
  }

  return (
    <div className="p-6 text-gray-100 bg-gray-950 min-h-screen">
{(OptimizerNav as any)({ active: "run-all" })}
      <h2 className="text-xl font-semibold mb-4 text-teal-400">
        Run All Optimizers
      </h2>

      <div className="flex flex-col gap-6 mb-6">
        <AssetPicker onChange={setTickers} />
        <div className="flex flex-wrap gap-x-6 gap-y-4 items-end">
          {[
            { label:"Start",type:"date",v:start,on:(e:any)=>setStart(e.target.value)},
            { label:"End",type:"date",v:end,on:(e:any)=>setEnd(e.target.value)},
            { label:"Lev",type:"number",step:0.1,v:lev,on:(e:any)=>setLev(Number(e.target.value||1))},
            { label:"Interest %",type:"number",step:0.1,v:intRate,on:(e:any)=>setIntRate(Number(e.target.value||0))},
            { label:"Min W",type:"number",step:0.01,v:minW,on:(e:any)=>setMinW(Number(e.target.value||0))},
            { label:"Max W",type:"number",step:0.01,v:maxW,on:(e:any)=>setMaxW(Number(e.target.value||1))},
          ].map((p,i)=>(
            <label key={i} className="flex flex-col text-sm text-gray-300">
              <span className="mb-1">{p.label}</span>
              <input
                className="bg-gray-900 text-white p-2 rounded-md w-28"
                type={p.type} step={p.step} value={p.v} onChange={p.on}/>
            </label>
          ))}
          <button
            onClick={onRunAll}
            disabled={running}
            className={`px-5 py-2 rounded-md font-medium text-white ${
              running ? "bg-teal-800 cursor-wait" : "bg-teal-600 hover:bg-teal-500"
            }`}>
            {running ? "Running..." : "Run All"}
          </button>
        </div>
      </div>

      {err && <p className="text-red-400 mb-3">Error: {err}</p>}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {METHODS.map(({ key, label }) => {
          const r = results[key];
          return (
            <div key={key} className="border border-gray-800 rounded-lg p-3 bg-gray-900/40">
              <div className="flex justify-between items-center mb-2">
                <b>{label}</b>
                <span className="text-xs text-gray-400">
                  {r && (r as any).ms ? `${(r as any).ms} ms` : ""}
                </span>
              </div>
              {r.ok === false && (
                <div className="text-red-400 break-words">Failed: {r.error}</div>
              )}
              {r.ok === true && <StatsGrid pnl={r.data.pnl} />}
              {r.ok === null && <div className="text-gray-500 text-sm">Idle</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
