"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

type Item = { href: string; label: string };

const ITEMS: Item[] = [
  { href: "/module2/eqw", label: "Equal Weight" },
  { href: "/module2/mvo", label: "MVO" },
  { href: "/module2/mvo-target-return", label: "MVO Target" },
  { href: "/module2/frontier", label: "Frontier" },
  { href: "/module2/minvar", label: "Min-Var" },
  { href: "/module2/risk-parity", label: "Risk Parity" },
  { href: "/module2/erc", label: "ERC" },
  { href: "/module2/ga", label: "GA" },
  { href: "/module2/pso", label: "PSO" },
  { href: "/module2/nsga2", label: "NSGA-II" },
  { href: "/module2/user-weights", label: "User Weights" },
];

export function OptimizerNav() {
  const pathname = usePathname();
  return (
    <div className="tabs">
      <div className="tabs-inner">
        {ITEMS.map((it) => {
          const active = pathname?.startsWith(it.href);
          return (
            <Link key={it.href} href={it.href} className={`tab${active ? " active" : ""}`}>{it.label}</Link>
          );
        })}
      </div>
    </div>
  );
}

export default OptimizerNav;

