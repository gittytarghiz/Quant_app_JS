"use client";

import { useMemo } from "react";

type P = { pnl: Array<{ date: string; pnl: number | null }> };

function mean(a: number[]) { return a.reduce((s, x) => s + x, 0) / a.length; }
function stdev(a: number[]) {
  if (a.length < 2) return 0;
  const m = mean(a); let v = 0;
  for (const x of a) v += (x - m) * (x - m);
  return Math.sqrt(v / (a.length - 1));
}
function periodsPerYear(dates: string[]) {
  if (dates.length < 2) return 252;
  const ms = dates.map(d => new Date(d).getTime());
  const gaps = ms.slice(1).map((t, i) => Math.max(1, (t - ms[i]) / 86400000));
  const md = gaps.sort((a,b)=>a-b)[Math.floor(gaps.length/2)];
  if (md <= 2) return 252;
  if (md <= 10) return 52;
  if (md <= 40) return 12;
  return 1;
}
function percentile(arr: number[], p: number) {
  if (!arr.length) return 0;
  const a = [...arr].sort((x,y)=>x-y);
  const idx = Math.min(a.length - 1, Math.max(0, Math.floor(p * (a.length - 1))));
  return a[idx];
}
function cvar(arr: number[], p: number) {
  if (!arr.length) return 0;
  const a = [...arr].sort((x,y)=>x-y);
  const cut = percentile(a, p);
  const tail = a.filter(x => x <= cut);
  return tail.length ? mean(tail) : cut;
}
function skewness(arr: number[]) {
  const n = arr.length; if (n < 3) return 0;
  const m = mean(arr), sd = stdev(arr); if (!sd) return 0;
  let s3 = 0; for (const x of arr) s3 += Math.pow(x - m, 3);
  return (n / ((n - 1) * (n - 2))) * (s3 / Math.pow(sd, 3));
}
function excessKurtosis(arr: number[]) {
  const n = arr.length; if (n < 4) return 0;
  const m = mean(arr), sd = stdev(arr); if (!sd) return 0;
  let s4 = 0; for (const x of arr) s4 += Math.pow(x - m, 4);
  // Fisher’s excess kurtosis with small-sample correction
  const g2 = (n*(n+1)/((n-1)*(n-2)*(n-3))) * (s4/Math.pow(sd,4)) - (3*(n-1)*(n-1))/((n-2)*(n-3));
  return g2;
}

export function StatsGrid({ pnl }: P) {
  const metrics = useMemo(() => {
    const ret: number[] = [];
    const dates: string[] = [];
    for (const p of pnl) if (p.pnl != null && isFinite(p.pnl)) { ret.push(p.pnl); dates.push(p.date); }
    if (ret.length < 2) return null;

    const AF = periodsPerYear(dates);
    const mu = mean(ret);
    const sd = stdev(ret);
    const neg = ret.filter(r => r < 0);
    const ddv = neg.length ? stdev(neg) : 0;

    // equity, drawdown, longest DD duration
    let eq = 1, peak = 1, maxDD = 0;
    let ddStartIdx: number | null = null;
    let maxDDLenDays = 0;
    for (let i = 0; i < ret.length; i++) {
      eq *= 1 + ret[i];
      if (eq > peak) {
        peak = eq;
        ddStartIdx = null; // new peak resets DD
      } else {
        if (ddStartIdx === null) ddStartIdx = i; // start of a DD
        const dd = eq / peak - 1;
        if (dd < maxDD) maxDD = dd;
        // track longest DD span by days
        const startIdx = ddStartIdx ?? i;
        const start = new Date(dates[startIdx]).getTime();
        const now = new Date(dates[i]).getTime();
        const days = Math.max(0, Math.round((now - start)/86400000));
        if (days > maxDDLenDays) maxDDLenDays = days;
      }
    }
    const total = eq - 1;

    const first = new Date(dates[0]).getTime();
    const last  = new Date(dates[dates.length - 1]).getTime();
    const years = Math.max((last - first) / (365.25 * 86400000), ret.length / AF);
    const cagr = years > 0 ? Math.pow(1 + total, 1 / years) - 1 : 0;

    // additional stats
    const wins = ret.filter(r => r > 0);
    const losses = ret.filter(r => r < 0);
    const winRate = ret.length ? wins.length / ret.length : 0;
    const avgWin = wins.length ? mean(wins) : 0;
    const avgLoss = losses.length ? mean(losses) : 0; // negative
    const profitFactor = (wins.reduce((s,x)=>s+x,0)) / Math.abs(losses.reduce((s,x)=>s+x,0) || 1);

    const var95 = percentile(ret, 0.05);   // historical VaR at 95% (a negative number)
    const cvar95 = cvar(ret, 0.05);        // historical CVaR at 95% (mean of worst 5%)

    const calmar = Math.abs(maxDD) > 0 ? cagr / Math.abs(maxDD) : 0;
    const skew = skewness(ret);
    const kurt = excessKurtosis(ret);

    return {
      Sharpe: sd ? (mu / sd) * Math.sqrt(AF) : 0,
      Sortino: ddv ? (mu / ddv) * Math.sqrt(AF) : 0,
      "Ann. Vol": sd * Math.sqrt(AF),
      CAGR: cagr,
      "Total Return": total,
      "Max Drawdown": maxDD,           // negative
      "Calmar": calmar,
      "Win Rate": winRate,             // %
      "Profit Factor": isFinite(profitFactor) ? profitFactor : 0,
      "Avg Win": avgWin,               // %
      "Avg Loss": avgLoss,             // %
      "VaR(95%)": var95,               // %
      "CVaR(95%)": cvar95,             // %
      "Skew": skew,
      "Kurtosis": kurt,
      "Max DD Dur (d)": maxDDLenDays,  // days
    };
  }, [pnl]);

  if (!metrics) return <div className="text-sm">No data</div>;

  const percentKeys = new Set([
    "Ann. Vol","CAGR","Total Return","Max Drawdown",
    "Win Rate","Avg Win","Avg Loss","VaR(95%)","CVaR(95%)",
  ]);

  const fmt = (k: string, v: number) => {
    if (!isFinite(v)) return "-";
    if (k === "Max DD Dur (d)") return `${Math.round(v)}`; // plain integer
    if (percentKeys.has(k)) return `${(v * 100).toFixed(2)}%`;
    // ratios & moments
    if (k === "Sharpe" || k === "Sortino" || k === "Calmar" || k === "Profit Factor" || k === "Skew")
      return v.toFixed(2);
    if (k === "Kurtosis") return v.toFixed(2);
    return v.toFixed(2);
  };

  return (
  <div className="stats-grid">
  {Object.entries(metrics).map(([k, v]) => {
    const isNeg =
      (typeof v === "number" && !percentKeys.has(k) && (v as number) < 0) ||
      (k === "Max Drawdown" && (v as number) < 0);

    return (
      <div key={k} className={`stat-card ${isNeg ? "neg" : "pos"}`}>
        <span className="stat-label">{k}</span>
        <span className="stat-value">{fmt(k, v as number)}</span>
      </div>
    );
  })}
</div>



);


}
