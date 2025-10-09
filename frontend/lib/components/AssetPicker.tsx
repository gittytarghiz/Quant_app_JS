"use client";
import { useEffect, useState, useMemo, useRef } from "react";

export default function AssetPicker({
  onChange,
}: {
  onChange?: (tickers: string[]) => void;
}) {
  type MapT = Record<string, string[]>;

  const [assets, setAssets] = useState<MapT>({});
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [visibleClasses, setVisibleClasses] = useState<Set<string>>(new Set());
  const [open, setOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Load once
  useEffect(() => {
    (async () => {
      const res = await fetch("/asset_classification.json");
      const data = (await res.json()) as MapT;
      setAssets(data);
      setVisibleClasses(new Set(Object.keys(data))); // all visible by default
    })();
  }, []);

  useEffect(() => {
    onChange?.(selected);
  }, [selected]);

  const allTickers = useMemo(() => {
    const visible = Array.from(visibleClasses);
    return visible.length
      ? visible.flatMap((cls) => assets[cls] || []).map((x) => x.toUpperCase())
      : [];
  }, [assets, visibleClasses]);

  const filtered = useMemo(() => {
    const q = query.trim().toUpperCase();
    if (!q) return [];
    return allTickers.filter((t) => t.includes(q)).slice(0, 50);
  }, [query, allTickers]);

  const selectTicker = (t: string) => {
    t = t.toUpperCase();
    if (!selected.includes(t)) setSelected([...selected, t]);
    setQuery("");
    setOpen(false);
  };

  const deselectTicker = (t: string) =>
    setSelected((prev) => prev.filter((x) => x !== t));

  const toggleClassVisibility = (cls: string) => {
    setVisibleClasses((prev) => {
      const copy = new Set(prev);
      if (copy.has(cls)) copy.delete(cls);
      else copy.add(cls);
      return copy;
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      const t = query.trim().toUpperCase();
      if (!t) return;
      if (allTickers.includes(t)) selectTicker(t);
      else if (filtered.length > 0) selectTicker(filtered[0]);
    }
  };

  return (
    <div
      style={{
        position: "relative",
        background: "#0f0f0f",
        border: "1px solid #222",
        borderRadius: 8,
        padding: 8,
        color: "#eee",
      }}
    >
      {/* Asset class visibility toggles */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 6,
          marginBottom: 8,
        }}
      >
        {Object.keys(assets).map((cls) => {
          const visible = visibleClasses.has(cls);
          return (
            <button
              key={cls}
              onClick={() => toggleClassVisibility(cls)}
              style={{
                background: visible ? "#1d4ed8" : "#1e293b",
                color: "#e2e8f0",
                border: "none",
                borderRadius: 6,
                padding: "4px 8px",
                cursor: "pointer",
                fontSize: 13,
                opacity: visible ? 1 : 0.6,
              }}
            >
              {cls}
            </button>
          );
        })}
      </div>

      {/* Selected tickers + input */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 6,
          alignItems: "center",
        }}
        onClick={() => inputRef.current?.focus()}
      >
        {selected.map((t) => (
          <span
            key={t}
            style={{
              background: "#064e3b",
              color: "#d1fae5",
              padding: "2px 6px",
              borderRadius: 6,
              display: "flex",
              alignItems: "center",
              gap: 4,
            }}
          >
            {t}
            <button
              onClick={() => deselectTicker(t)}
              style={{
                background: "none",
                border: "none",
                color: "#94a3b8",
                cursor: "pointer",
                fontSize: 12,
              }}
            >
              ×
            </button>
          </span>
        ))}
        <input
          ref={inputRef}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 100)}
          onKeyDown={handleKeyDown}
          placeholder="Type to search..."
          style={{
            flex: 1,
            background: "transparent",
            border: "none",
            color: "#fff",
            outline: "none",
            minWidth: 120,
          }}
        />
      </div>

      {/* Dropdown */}
      {open && query && (
        <div
          style={{
            position: "absolute",
            top: "100%",
            left: 0,
            right: 0,
            background: "#111",
            border: "1px solid #333",
            borderRadius: 6,
            maxHeight: 200,
            overflowY: "auto",
            zIndex: 10,
          }}
        >
          {filtered.length === 0 ? (
            <div style={{ padding: 8, color: "#666" }}>No results</div>
          ) : (
            filtered.map((t) => (
              <div
                key={t}
                onMouseDown={(e) => {
                  e.preventDefault();
                  selectTicker(t);
                }}
                style={{
                  padding: "6px 10px",
                  cursor: "pointer",
                  color: "#eee",
                }}
              >
                {t}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
