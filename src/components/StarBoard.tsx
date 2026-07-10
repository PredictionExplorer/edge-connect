'use client';

import { memo, useMemo, useState } from 'react';
import type { Board } from '@/lib/star/board';
import { EMPTY } from '@/lib/star/scoring';
import { PLAYER_COLORS } from './theme';

const S = 100; // unit-coordinate scale

export interface StarBoardProps {
  board: Board;
  stones: ArrayLike<number>;
  /** Node controller from the scorer, for the territory overlay. */
  nodeOwner?: ArrayLike<number> | null;
  /** 1 = stone belongs to an alive star. Dead stones render dimmed. */
  aliveStone?: ArrayLike<number> | null;
  showTerritory?: boolean;
  lastMove?: number;
  currentTurnMoves?: number[];
  toMove?: 0 | 1;
  interactive?: boolean;
  onPlace?: (node: number) => void;
  onHover?: (node: number) => void;
  className?: string;
}

// Must match the arm base angle in board.ts (corner up, '*' arm lower right).
function pentagonPath(radius: number, rotate = 54): string {
  const pts: string[] = [];
  for (let i = 0; i < 5; i++) {
    const a = ((rotate + 72 * i) * Math.PI) / 180;
    pts.push(`${(radius * Math.cos(a)).toFixed(2)},${(radius * Math.sin(a)).toFixed(2)}`);
  }
  return `M${pts.join('L')}Z`;
}

export const StarBoard = memo(function StarBoard({
  board,
  stones,
  nodeOwner,
  aliveStone,
  showTerritory = false,
  lastMove = -1,
  currentTurnMoves = [],
  toMove = 0,
  interactive = false,
  onPlace,
  onHover,
  className,
}: StarBoardProps) {
  const [hovered, setHovered] = useState(-1);

  const stoneR = Math.min(board.minEdge * 0.46 * S, 9.5);

  const { meshPath, bridgeChordPath, bridgeStarPoints, periRingPath } = useMemo(() => {
    const { adjOff, adj, xs, ys, ringOf, sectorOf, bridge } = board;
    const mesh: string[] = [];
    for (let u = 0; u < board.n; u++) {
      for (let e = adjOff[u]; e < adjOff[u + 1]; e++) {
        const v = adj[e];
        if (v < u) continue;
        // Bridge chords (non-consecutive ring-1 pairs) are drawn separately.
        if (
          ringOf[u] === 1 &&
          ringOf[v] === 1 &&
          (sectorOf[u] + 1) % 5 !== sectorOf[v] &&
          (sectorOf[v] + 1) % 5 !== sectorOf[u]
        ) {
          continue;
        }
        mesh.push(
          `M${(xs[u] * S).toFixed(2)} ${(ys[u] * S).toFixed(2)}L${(xs[v] * S).toFixed(2)} ${(ys[v] * S).toFixed(2)}`,
        );
      }
    }
    const chords: string[] = [];
    for (let i = 0; i < 5; i++) {
      for (let j = i + 1; j < 5; j++) {
        if ((i + 1) % 5 === j || (j + 1) % 5 === i) continue;
        const a = bridge[i];
        const b = bridge[j];
        chords.push(
          `M${(xs[a] * S).toFixed(2)} ${(ys[a] * S).toFixed(2)}L${(xs[b] * S).toFixed(2)} ${(ys[b] * S).toFixed(2)}`,
        );
      }
    }
    // Pentagram polygon *10 -> T10 -> R10 -> S10 -> A10 (every second point).
    const order = [0, 2, 4, 1, 3].map((i) => bridge[i]);
    const starPoints = order
      .map((u) => `${(xs[u] * S).toFixed(2)},${(ys[u] * S).toFixed(2)}`)
      .join(' ');
    // Perimeter outline through all ring-r nodes.
    const peri: string[] = [];
    for (let s = 0; s < 5; s++) {
      for (let y = 0; y < board.rings; y++) {
        const u = board.idx(s, board.rings, y);
        peri.push(`${(xs[u] * S).toFixed(2)} ${(ys[u] * S).toFixed(2)}`);
      }
    }
    return {
      meshPath: mesh.join(''),
      bridgeChordPath: chords.join(''),
      bridgeStarPoints: starPoints,
      periRingPath: `M${peri.join('L')}Z`,
    };
  }, [board]);

  const hover = (u: number) => {
    setHovered(u);
    onHover?.(u);
  };

  const territory = showTerritory && nodeOwner ? nodeOwner : null;

  return (
    <svg
      viewBox="-119 -119 238 238"
      className={className}
      role="application"
      aria-label={`*Star board with ${board.rings} rings`}
      onMouseLeave={() => hover(-1)}
    >
      <defs>
        <radialGradient id="plate" cx="38%" cy="30%" r="90%">
          <stop offset="0%" stopColor="#1b2140" />
          <stop offset="55%" stopColor="#12162b" />
          <stop offset="100%" stopColor="#0b0e1e" />
        </radialGradient>
        {PLAYER_COLORS.map((c, i) => (
          <radialGradient key={i} id={`stone${i}`} cx="35%" cy="30%" r="80%">
            <stop offset="0%" stopColor={c.bright} />
            <stop offset="55%" stopColor={c.base} />
            <stop offset="100%" stopColor={c.deep} />
          </radialGradient>
        ))}
        <filter id="softGlow" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="1.6" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <filter id="stoneShadow" x="-60%" y="-60%" width="220%" height="220%">
          <feDropShadow dx="0" dy="1.1" stdDeviation="1.4" floodColor="#000" floodOpacity="0.55" />
        </filter>
      </defs>

      {/* Plate */}
      <path d={pentagonPath(114)} fill="url(#plate)" stroke="rgba(232,196,139,0.38)" strokeWidth="1.1" strokeLinejoin="round" />
      <path d={pentagonPath(110)} fill="none" stroke="rgba(232,196,139,0.14)" strokeWidth="0.6" strokeLinejoin="round" />

      {/* Mesh */}
      <path d={meshPath} stroke="rgba(232,196,139,0.30)" strokeWidth="0.55" fill="none" strokeLinecap="round" />
      {/* Perimeter emphasis */}
      <path d={periRingPath} stroke="rgba(232,196,139,0.55)" strokeWidth="1.1" fill="none" strokeLinejoin="round" />

      {/* Central star bridge */}
      <polygon
        points={bridgeStarPoints}
        fill="rgba(232,196,139,0.10)"
        stroke="none"
        fillRule="nonzero"
      />
      <path
        d={bridgeChordPath}
        stroke="rgba(247,220,166,0.75)"
        strokeWidth="0.9"
        fill="none"
        strokeLinecap="round"
        filter="url(#softGlow)"
      />

      {/* Node layers */}
      {Array.from({ length: board.n }, (_, u) => {
        const x = board.xs[u] * S;
        const y = board.ys[u] * S;
        const stone = stones[u];
        const isEmpty = stone === EMPTY;
        const owner = territory ? territory[u] : -1;
        const quark = board.isQuark[u] === 1;
        const peri = board.isPeri[u] === 1;
        const dead = !isEmpty && aliveStone && showTerritory ? aliveStone[u] === 0 : false;

        return (
          <g key={u}>
            {/* peri / quark markers */}
            {peri && (
              <circle
                cx={x}
                cy={y}
                r={stoneR * (quark ? 1.5 : 1.28)}
                fill="none"
                stroke={quark ? 'rgba(247,220,166,0.8)' : 'rgba(232,196,139,0.45)'}
                strokeWidth={quark ? 1.0 : 0.55}
                strokeDasharray={quark ? undefined : '1.6 1.9'}
              />
            )}

            {/* territory tint */}
            {territory && owner !== -1 && (isEmpty || dead) && (
              <circle
                cx={x}
                cy={y}
                r={stoneR * 0.55}
                fill={PLAYER_COLORS[owner as 0 | 1].glow}
                opacity={peri ? 0.95 : 0.5}
              />
            )}

            {/* empty node dot */}
            {isEmpty && (
              <circle cx={x} cy={y} r={stoneR * 0.16} fill="rgba(232,196,139,0.55)" />
            )}

            {/* hover ghost */}
            {isEmpty && hovered === u && interactive && (
              <circle
                cx={x}
                cy={y}
                r={stoneR}
                fill={`url(#stone${toMove})`}
                opacity={0.45}
              />
            )}

            {/* stone */}
            {!isEmpty && (
              <g className="stone-pop" opacity={dead ? 0.35 : 1}>
                <circle
                  cx={x}
                  cy={y}
                  r={stoneR}
                  fill={`url(#stone${stone})`}
                  stroke={PLAYER_COLORS[stone as 0 | 1].deep}
                  strokeWidth="0.5"
                  filter={dead ? undefined : 'url(#stoneShadow)'}
                />
                <ellipse
                  cx={x - stoneR * 0.3}
                  cy={y - stoneR * 0.38}
                  rx={stoneR * 0.34}
                  ry={stoneR * 0.22}
                  fill="rgba(255,255,255,0.55)"
                  transform={`rotate(-24 ${x - stoneR * 0.3} ${y - stoneR * 0.38})`}
                />
              </g>
            )}

            {/* current-turn + last-move markers */}
            {currentTurnMoves.includes(u) && u !== lastMove && (
              <circle cx={x} cy={y} r={stoneR * 0.22} fill="rgba(255,255,255,0.85)" />
            )}
            {u === lastMove && (
              <circle cx={x} cy={y} fill="none" stroke="rgba(255,255,255,0.9)">
                {/* SMIL keeps the pulse alive without JS-driven animation. */}
                <animate
                  attributeName="r"
                  values={`${stoneR};${stoneR * 1.5}`}
                  dur="1.6s"
                  repeatCount="indefinite"
                />
                <animate
                  attributeName="stroke-opacity"
                  values="0.9;0"
                  dur="1.6s"
                  repeatCount="indefinite"
                />
                <animate
                  attributeName="stroke-width"
                  values="0.9;0.5"
                  dur="1.6s"
                  repeatCount="indefinite"
                />
              </circle>
            )}

            {/* hit target */}
            {interactive && isEmpty && (
              <circle
                cx={x}
                cy={y}
                r={Math.max(stoneR, 4)}
                fill="transparent"
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => hover(u)}
                onClick={() => onPlace?.(u)}
              />
            )}
          </g>
        );
      })}
    </svg>
  );
});
