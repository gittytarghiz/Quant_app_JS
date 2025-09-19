"use client";

import { useEffect, useRef } from "react";

type Point = { date: string; pnl: number | null };

export function LineChart({ data, height = 160 }: { data: Point[]; height?: number }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const dpr = (typeof devicePixelRatio === 'number' && devicePixelRatio > 0) ? devicePixelRatio : 1;
    const W = c.clientWidth * dpr;
    const H = height * dpr;
    c.width = W; c.height = H;
    ctx.clearRect(0, 0, W, H);

    const pts = data.filter(d => typeof d.pnl === 'number') as {date:string; pnl:number}[];
    if (pts.length === 0) return;

    const xs = pts.map(p => new Date(p.date).getTime());
    const ys = pts.map(p => p.pnl);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const y0 = Math.min(...ys), y1 = Math.max(...ys);
    const pad = 8 * dpr;
    const bottomPad = pad + 18 * dpr; // extra space for date labels
    const sx = (x: number) => pad + (x - x0) / Math.max(1, (x1 - x0)) * (W - 2 * pad);
    const sy = (y: number) => H - bottomPad - (y - y0) / Math.max(1e-9, (y1 - y0)) * (H - (pad + bottomPad));

    // zero line
    const z = 0 >= y0 && 0 <= y1 ? sy(0) : null;
    if (z !== null) {
      ctx.strokeStyle = '#ddd';
      ctx.beginPath();
      ctx.moveTo(pad, z);
      ctx.lineTo(W - pad, z);
      ctx.stroke();
    }

    // line
    ctx.strokeStyle = '#1e88e5';
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    ctx.moveTo(sx(xs[0]), sy(ys[0]));
    for (let i = 1; i < xs.length; i++) {
      ctx.lineTo(sx(xs[i]), sy(ys[i]));
    }
    ctx.stroke();

    // x-axis date ticks and labels
    const msPerDay = 24 * 3600 * 1000;
    const spanDays = (x1 - x0) / msPerDay;
    const fmt = (t: number) => {
      const d = new Date(t);
      if (spanDays >= 365 * 2) return String(d.getFullYear());
      if (spanDays >= 180) return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
      return `${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
    };
    const ticks = 5;
    ctx.fillStyle = '#9aa2b1';
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.font = `${12 * dpr}px ui-sans-serif, system-ui, -apple-system, Segoe UI`;
    ctx.textAlign = 'center';
    const labelY = H - (pad * 0.5);
    for (let i = 0; i < ticks; i++) {
      const t = x0 + (i / (ticks - 1)) * (x1 - x0);
      const xx = sx(t);
      // grid line
      ctx.beginPath();
      ctx.moveTo(xx, H - bottomPad);
      ctx.lineTo(xx, pad);
      ctx.stroke();
      // label (convert coords to CSS pixel space)
      ctx.fillText(fmt(t), xx, labelY);
    }
  }, [data, height]);
  return <canvas ref={ref} style={{ width: '100%', height }} />;
}
