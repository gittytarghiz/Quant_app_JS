"use client";

export function WeightsTable({ rows }: { rows: Array<Record<string, string | number | null>> }) {
  if (!rows || rows.length === 0) return null;
  const cols = Object.keys(rows[0]).filter(k => k !== 'date');
  return (
    <div style={{ overflowX: 'auto' }}>
      <table>
        <thead>
          <tr>
            <th>date</th>
            {cols.map(c => (
              <th key={c} style={{ textAlign: 'right' }}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 20).map((r, i) => (
            <tr key={i}>
              <td>{String(r.date)}</td>
              {cols.map(c => (
                <td key={c} style={{ textAlign: 'right' }}>{(r[c] as number | null)?.toFixed?.(4) ?? ''}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
