"use client";

import { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

type WeightRow = { date?: string; [k: string]: string | number | null | undefined };

const toISO = (d: any) => {
  try {
    const dt = new Date(d);
    if (isNaN(dt.getTime())) return String(d);
    return dt.toISOString().slice(0, 10); // YYYY-MM-DD
  } catch {
    return String(d);
  }
};

export function WeightsChart({ weights }: { weights?: WeightRow[] }) {
  const { data, tickers } = useMemo(() => {
    if (!weights || weights.length === 0) return { data: [], tickers: [] as string[] };

    const keys = Array.from(
      weights.reduce((s, r) => {
        Object.keys(r || {}).forEach((k) => k !== "date" && s.add(k));
        return s;
      }, new Set<string>())
    );

    const cleaned = weights.map((row) => {
      const out: Record<string, number | string> = { date: toISO(row.date) };
      for (const k of keys) {
        const v = row[k];
        const num = typeof v === "number" ? v : v == null ? 0 : Number(v);
        out[k] = Number.isFinite(num) ? num : 0;
      }
      return out;
    });

    const active = keys.filter((k) => cleaned.some((r) => (r[k] as number) !== 0));

    return { data: cleaned, tickers: active };
  }, [weights]);

  if (!data.length || !tickers.length) {
    return (
      <div className="w-full min-h-[100px] bg-gray-900 rounded-xl p-4 text-xs text-gray-400 flex items-center">
        No weight data to display.
      </div>
    );
  }

  return (
    <div
      className="w-full bg-gray-900 rounded-xl p-4 shadow-md"
      style={{ height: "400px" }} // ✅ force height so ResponsiveContainer works
    >
      <h3 className="text-sm text-gray-300 mb-2">Portfolio Weights Over Time</h3>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} stackOffset="expand">
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#9CA3AF" }} />
          <YAxis
            tickFormatter={(v) => `${Math.round(Number(v) * 100)}%`}
            tick={{ fontSize: 10, fill: "#9CA3AF" }}
          />
          <Tooltip
            formatter={(value: any) => `${(Number(value) * 100).toFixed(2)}%`}
            labelFormatter={(label) => `Date: ${label}`}
          />
          <Legend />
          {tickers.map((t, i) => {
            const hue = (i * 57) % 360;
            const color = `hsl(${hue},70%,60%)`;
            return (
              <Area
                key={t}
                type="monotone"
                dataKey={t}
                stackId="1"
                stroke={color}
                fill={color}
                isAnimationActive={false}
              />
            );
          })}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
