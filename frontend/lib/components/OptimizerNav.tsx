"use client";

import Link from "next/link";

export function OptimizerNav() {
  const links = [
    { name: "Equal Weight", href: "/module2/eqw" },
    { name: "MVO", href: "/module2/mvo" },
    { name: "Minimum Variance", href: "/module2/minvar" },
    { name: "Risk Parity", href: "/module2/risk-parity" },
    { name: "Particle Swarm", href: "/module2/pso" },
    { name: "Genetic Algorithm", href: "/module2/ga" },
    { name: "NSGA-II", href: "/module2/nsga2" },
    { name: "User Weights", href: "/module2/user-weights" },
  ];

  return (
    <div className="flex flex-wrap gap-3 border-b border-gray-700 pb-2 mb-4">
      {links.map((link) => (
        <Link key={link.href} href={link.href}>
          <button className="px-3 py-1 rounded bg-gray-800 text-sm text-gray-300 hover:bg-gray-700 hover:text-white transition">
            {link.name}
          </button>
        </Link>
      ))}
    </div>
  );
}
