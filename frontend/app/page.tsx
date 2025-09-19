import Link from "next/link";

export default function Home() {
  return (
    <div>
      <div className="hero">
        <h2 style={{ marginBottom: 4 }}>Welcome to Quant App</h2>
        <div className="muted">Backtesting and portfolio construction made simple.</div>
        <div style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <Link href="/module1" className="btn btn-primary">Open Module 1 — Data</Link>
          <Link href="/module2" className="btn">Open Module 2 — Optimization</Link>
        </div>
      </div>

      <div className="grid">
        <div className="card">
          <h3>Module 1 — Data</h3>
          <p className="muted">Download historical prices for any universe across common intervals (daily, weekly, monthly). Export as CSV.</p>
          <Link href="/module1" className="btn">Go to Data</Link>
        </div>
        <div className="card">
          <h3>Module 2 — Optimization</h3>
          <p className="muted">Run equal-weight, MVO, risk parity and evolutionary methods. Inspect weights, PnL, and metrics.</p>
          <Link href="/module2" className="btn">Go to Optimization</Link>
        </div>
      </div>

      <p style={{ marginTop: 24 }} className="muted">
        Tip: set <code>NEXT_PUBLIC_API_BASE_URL</code> to your FastAPI server (e.g. http://localhost:8000).
      </p>
    </div>
  );
}
