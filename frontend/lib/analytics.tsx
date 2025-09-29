"use client";

import { StatsGrid } from "./components/StatsGrid";
import { LineChart } from "./components/LineChart";
import { WeightsChart } from "./components/WeightsChart";

type EqwResponse = {
  pnl: Array<{ date: string; pnl: number | null }>;
  weights?: Array<Record<string, string | number | null>>;
  details?: Record<string, any>;
};

export function Analytics({ data }: { data: EqwResponse }) {
  return (
    <div className="analytics-wrapper">
      <LineChart pnl={data.pnl} />
      <StatsGrid pnl={data.pnl} />
      {data.weights?.length ? <WeightsChart weights={data.weights} /> : null}
    </div>
  );
}
