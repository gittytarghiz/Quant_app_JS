import Link from "next/link";
import { OptimizerNav } from "../../lib/components/OptimizerNav";

export default function Module2Home() {
  return (
    <div>
      <OptimizerNav />
      <div className="hero">
        <h2 style={{ marginBottom: 4 }}>Module 2 — Portfolio Optimization</h2>
        <div className="muted">Build, backtest, and compare strategies. Choose an optimizer, tune constraints, and review performance.</div>
        <div style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <Link href="/module2/mvo" className="btn btn-primary">Run MVO</Link>
          <Link href="/module2/risk-parity" className="btn">Risk Parity</Link>
          <Link href="/module2/frontier" className="btn">Efficient Frontier</Link>
          <Link href="/module2/ga" className="btn">Genetic (DE)</Link>
        </div>
      </div>

      <div className="grid">
        <div className="card">
          <h3>Equal Weight</h3>
          <p className="muted">Baseline portfolio with box constraints and turnover costs.</p>
          <Link href="/module2/eqw" className="btn">Open</Link>
        </div>
        <div className="card">
          <h3>MVO</h3>
          <p className="muted">Mean-variance with Sharpe/Sortino/Calmar/CVaR, Kelly, diversification. Supports L2 and covariance shrinkage.</p>
          <Link href="/module2/mvo" className="btn">Open</Link>
        </div>
        
        
        <div className="card">
          <h3>Min-Variance</h3>
          <p className="muted">Classic minimum variance portfolio using expanding windows.</p>
          <Link href="/module2/minvar" className="btn">Open</Link>
        </div>
        <div className="card">
          <h3>Risk Parity</h3>
          <p className="muted">Equalize risk contributions (variance-based). Choose covariance estimator.</p>
          <Link href="/module2/risk-parity" className="btn">Open</Link>
        </div>
        <div className="card">
          <h3>ERC <span style={{ color: 'var(--accent)' }}>NEW</span></h3>
          <p className="muted">Equal Risk Contribution alias of risk parity with consistent outputs.</p>
          <Link href="/module2/erc" className="btn">Open</Link>
        </div>
        <div className="card">
          <h3>Genetic Algorithm (DE)</h3>
          <p className="muted">Differential Evolution with rich objective set and advanced controls.</p>
          <Link href="/module2/ga" className="btn">Open</Link>
        </div>
        <div className="card">
          <h3>Particle Swarm</h3>
          <p className="muted">PSO with Sharpe/Sortino/Calmar/CVaR and growth/diversification objectives.</p>
          <Link href="/module2/pso" className="btn">Open</Link>
        </div>
        <div className="card">
          <h3>NSGA‑II</h3>
          <p className="muted">Multi‑objective optimizer; explore tradeoffs across multiple metrics.</p>
          <Link href="/module2/nsga2" className="btn">Open</Link>
        </div>
        <div className="card">
          <h3>User Weights</h3>
          <p className="muted">Provide your own weights over time and evaluate performance.</p>
          <Link href="/module2/user-weights" className="btn">Open</Link>
        </div>
      </div>

      <p style={{ marginTop: 16 }} className="muted">Tip: set <code>NEXT_PUBLIC_API_BASE</code> to your FastAPI server (e.g. http://localhost:8000) if not using same‑origin.</p>
    </div>
  );
}
