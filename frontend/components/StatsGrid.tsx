"use client";

import { useMemo } from "react";

type Stats = null | Record<string, number>;
type Metrics = null | Record<string, any>;
type Fmt = '2f' | '3f' | '2%' | '0f';

export function StatsGrid({ stats, metrics }: { stats: Stats; metrics?: Metrics }) {
  const get = (k: string): number | undefined => {
    // Prefer server metrics (snake_case), then fallback to client stats (camelCase)
    if (metrics && typeof metrics[k] === 'number') return metrics[k];
    const map: Record<string, string> = {
      ann_vol: 'annVol',
      total_return: 'totalReturn',
      max_drawdown: 'maxDrawdown',
    };
    const ks = map[k] || k;
    const v = stats ? (stats as any)[ks] : undefined;
    return typeof v === 'number' ? v : undefined;
  };

  const items: Array<{ label: string; value?: number; fmt?: Fmt }> = [
    { label: 'Sharpe', value: get('sharpe'), fmt: '3f' as Fmt },
    { label: 'Sortino', value: get('sortino'), fmt: '3f' as Fmt },
    { label: 'Ann. Vol', value: get('ann_vol'), fmt: '2%' as Fmt },
    { label: 'Ann. Return', value: get('ann_return'), fmt: '2%' as Fmt },
    { label: 'CAGR', value: get('cagr'), fmt: '2%' as Fmt },
    { label: 'Total Return', value: get('total_return'), fmt: '2%' as Fmt },
    { label: 'Max Drawdown', value: get('max_drawdown'), fmt: '2%' as Fmt },
    { label: 'Calmar', value: get('calmar'), fmt: '3f' as Fmt },
    { label: 'Avg Turnover', value: get('avg_turnover'), fmt: '2%' as Fmt },
    { label: 'Total Turnover', value: get('total_turnover'), fmt: '2%' as Fmt },
    { label: 'VaR 95%', value: get('var_95'), fmt: '2%' as Fmt },
    { label: 'CVaR 95%', value: get('cvar_95'), fmt: '2%' as Fmt },
    { label: 'Max DD Days', value: get('max_drawdown_days'), fmt: '0f' as Fmt },
    { label: 'Curr DD Days', value: get('current_drawdown_days'), fmt: '0f' as Fmt },
  ].filter(it => typeof it.value === 'number' && isFinite(it.value as number));

  if (items.length === 0) return null;
  return (
    <div className="spaced">
      <h3>Stats</h3>
      <div className="stats-grid">
        {items.map((s, i) => <Stat key={i} label={s.label} value={s.value} fmt={s.fmt} />)}
      </div>
    </div>
  );
}

function Stat({ label, value, fmt = "2f" }: { label: string; value?: number; fmt?: "2f" | "3f" | "2%" | "0f" }) {
  const text = useMemo(() => {
    if (typeof value !== 'number' || !isFinite(value)) return "-";
    if (fmt === "2%") return `${(value * 100).toFixed(2)}%`;
    if (fmt === "3f") return value.toFixed(3);
    if (fmt === "0f") return Math.round(value).toString();
    return value.toFixed(2);
  }, [value, fmt]);
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{text}</div>
    </div>
  );
}
