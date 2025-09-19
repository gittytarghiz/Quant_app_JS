export type PnlPoint = { date: string; pnl: number | null };

export type Derived = {
  equityChart: { date: string; pnl: number }[];
  stats: {
    ann: number;
    sharpe: number;
    sortino: number;
    annVol: number;
    maxDrawdown: number; // negative
    totalReturn: number;
    cagr: number;
  } | null;
};

export function deriveFromPnl(pnlIn: PnlPoint[] | undefined | null): Derived {
  const pnl = (pnlIn || []).filter((p): p is { date: string; pnl: number } => typeof p?.pnl === 'number');
  if (!pnl.length) return { equityChart: [], stats: null };

  const times = pnl.map(p => new Date(p.date).getTime()).sort((a, b) => a - b);
  const deltas = times.slice(1).map((t, i) => (t - times[i]) / (24 * 3600 * 1000));
  const med = deltas.length ? [...deltas].sort((a, b) => a - b)[Math.floor(deltas.length / 2)] : 1;
  const ann = med <= 2 ? 252 : med <= 10 ? 52 : med <= 40 ? 12 : 252;

  let eq = 1.0;
  const equity: number[] = [];
  for (const p of pnl) { eq *= 1 + p.pnl; equity.push(eq); }
  const equityChart = pnl.map((p, i) => ({ date: p.date, pnl: equity[i] }));

  const rets = pnl.map(p => p.pnl);
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const variance = rets.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(1, rets.length - 1);
  const vol = Math.sqrt(Math.max(variance, 0));
  const annVol = vol * Math.sqrt(ann);
  const sharpe = annVol > 0 ? (mean * ann) / annVol : 0;
  const downs = rets.filter(r => r < 0);
  const ds = downs.length ? Math.sqrt(downs.reduce((a, b) => a + b * b, 0) / downs.length) : 0;
  const sortino = ds > 0 ? (mean * ann) / (ds * Math.sqrt(ann)) : 0;
  let peak = equity[0] || 1;
  let mdd = 0;
  for (const v of equity) { if (v > peak) peak = v; mdd = Math.min(mdd, v / peak - 1); }
  const totalReturn = equity[equity.length - 1] - 1;
  const T = rets.length;
  const cagr = Math.pow(Math.max(equity[equity.length - 1], 1e-9), ann / Math.max(T, 1)) - 1;

  return { equityChart, stats: { ann, sharpe, sortino, annVol, maxDrawdown: mdd, totalReturn, cagr } };
}

