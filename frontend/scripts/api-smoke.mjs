#!/usr/bin/env node
// Simple Node smoke test for API and Next.js rewrites.
// Usage:
//   node scripts/api-smoke.mjs --api http://localhost:8000
//   node scripts/api-smoke.mjs --via-next http://localhost:3000

function parseArgs() {
  const args = process.argv.slice(2);
  const out = { api: null, viaNext: null };
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--api') out.api = args[++i];
    if (args[i] === '--via-next') out.viaNext = args[++i];
  }
  return out;
}

async function hit(url, body) {
  const res = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  let json; try { json = JSON.parse(text); } catch { json = { raw: text }; }
  return { status: res.status, json };
}

async function main() {
  const { api, viaNext } = parseArgs();
  const payload = {
    tickers: ['AAPL','MSFT'], start: '2024-01-01', end: '2024-03-01', dtype: 'close', interval: '1d'
  };
  if (api) {
    const url = api.replace(/\/$/, '') + '/data/prices';
    console.log('POST', url);
    const r = await hit(url, payload);
    console.log('Status', r.status, 'Count', r.json?.meta?.count, 'Cols', r.json?.columns?.length);
    if (r.status !== 200) process.exit(1);
  }
  if (viaNext) {
    const url = viaNext.replace(/\/$/, '') + '/data/prices';
    console.log('POST (via Next)', url);
    const r = await hit(url, payload);
    console.log('Status', r.status, 'Count', r.json?.meta?.count, 'Cols', r.json?.columns?.length);
    if (r.status !== 200) process.exit(1);
  }
}

main().catch(e => { console.error('Smoke failed', e); process.exit(1); });

