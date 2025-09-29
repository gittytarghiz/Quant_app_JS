"use client";

import {
  LineChart as LC,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

type P = { pnl: Array<{ date: string; pnl: number | null }> };

export function LineChart({ pnl }: P) {
  // build equity curve
  let eq = 1;
  const data = pnl.map(p => {
    if (p.pnl != null) eq *= 1 + p.pnl;
    return { date: p.date, equity: eq };
  });

  return (
    <div className="card" style={{ height: 340 }}>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-gray-200">Equity Curve</h3>
        <span className="text-xs text-gray-400">Cumulative return over time</span>
      </div>
      <ResponsiveContainer width="100%" height="90%">
        <LC data={data}>
          <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis
            dataKey="date"
            tick={{ fill: "#ccc", fontSize: 11 }} // brighter than muted gray
            tickLine={false}
          />
          <YAxis
            domain={["auto", "auto"]}
            tickFormatter={(v) => `${((v - 1) * 100).toFixed(0)}%`}
            tick={{ fill: "#ccc", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            formatter={(v: number) => `${((v - 1) * 100).toFixed(2)}%`}
            labelFormatter={(d) => `Date: ${d}`}
            contentStyle={{ background: "#151922", border: "1px solid #222839" }}
          />
          <Line
            type="monotone"
            dataKey="equity"
            stroke="var(--primary)"
            dot={false}
            strokeWidth={2}
          />
        </LC>
      </ResponsiveContainer>
    </div>
  );
}
