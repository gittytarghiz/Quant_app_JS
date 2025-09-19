import "./globals.css";
import Link from "next/link";

export const metadata = { title: "Quant App" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="app-shell">
          <aside className="sidebar">
            <div className="brand">Quant App</div>
            <nav className="nav">
              <div className="nav-section">Modules</div>
              <ul className="nav-list">
                <li><Link href="/">Home</Link></li>
                <li><Link href="/module1">Module 1 — Data</Link></li>
                <li><Link href="/module2">Module 2 — Optimization</Link></li>
              </ul>
            </nav>
          </aside>
          <main className="content">
            <div className="container">
              {children}
            </div>
          </main>
        </div>
      </body>
    </html>
  );
}
