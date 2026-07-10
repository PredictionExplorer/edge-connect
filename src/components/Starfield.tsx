'use client';

import { memo, useMemo } from 'react';

/** Deterministic decorative starfield (seeded, so SSR and client agree). */
export const Starfield = memo(function Starfield() {
  const stars = useMemo(() => {
    let seed = 0x51a7ca7 >>> 0;
    const rand = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    return Array.from({ length: 140 }, (_, i) => ({
      x: rand() * 100,
      y: rand() * 100,
      r: 0.14 + rand() * 0.5,
      o: 0.25 + rand() * 0.65,
      layer: i % 2,
    }));
  }, []);

  return (
    <svg
      aria-hidden
      className="pointer-events-none fixed inset-0 h-full w-full"
      preserveAspectRatio="xMidYMid slice"
      viewBox="0 0 100 100"
    >
      <g className="star-layer-a">
        {stars
          .filter((s) => s.layer === 0)
          .map((s, i) => (
            <circle key={i} cx={s.x} cy={s.y} r={s.r} fill="#f4eede" opacity={s.o * 0.5} />
          ))}
      </g>
      <g className="star-layer-b">
        {stars
          .filter((s) => s.layer === 1)
          .map((s, i) => (
            <circle key={i} cx={s.x} cy={s.y} r={s.r} fill="#e8c48b" opacity={s.o * 0.4} />
          ))}
      </g>
    </svg>
  );
});
