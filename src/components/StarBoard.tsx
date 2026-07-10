'use client';

import { memo, useId, useMemo, useRef, useState, type KeyboardEvent } from 'react';
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
  playerNames?: readonly [string, string];
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

type DirectionKey = 'ArrowUp' | 'ArrowDown' | 'ArrowLeft' | 'ArrowRight';

/**
 * Choose the nearest node that lies substantially in the requested visual
 * direction. This makes the board behave like a spatial control rather than
 * exposing implementation-order navigation.
 */
function nodeInDirection(board: Board, from: number, key: DirectionKey): number {
  const direction =
    key === 'ArrowUp'
      ? [0, -1]
      : key === 'ArrowDown'
        ? [0, 1]
        : key === 'ArrowLeft'
          ? [-1, 0]
          : [1, 0];
  let best = from;
  let bestCost = Number.POSITIVE_INFINITY;

  for (let candidate = 0; candidate < board.n; candidate++) {
    if (candidate === from) continue;
    const dx = board.xs[candidate] - board.xs[from];
    const dy = board.ys[candidate] - board.ys[from];
    const distance = Math.hypot(dx, dy);
    const alignment = (dx * direction[0] + dy * direction[1]) / distance;
    if (alignment < 0.35) continue;
    const cost = distance / alignment;
    if (cost < bestCost) {
      best = candidate;
      bestCost = cost;
    }
  }

  return best;
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
  playerNames,
  className,
}: StarBoardProps) {
  const [hovered, setHovered] = useState(-1);
  const [activeNode, setActiveNode] = useState(0);
  const [focusedNode, setFocusedNode] = useState(-1);
  const nodeRefs = useRef(new Map<number, SVGCircleElement>());
  const instructionsId = useId();

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
  const occupiedCount = Array.from({ length: board.n }, (_, node) => stones[node]).filter(
    (stone) => stone !== EMPTY,
  ).length;
  const currentPlayerName = playerNames?.[toMove] || PLAYER_COLORS[toMove].name;

  const focusNode = (node: number) => {
    setActiveNode(node);
    nodeRefs.current.get(node)?.focus();
  };

  const handleNodeKeyDown = (
    event: KeyboardEvent<SVGCircleElement>,
    node: number,
    isEmpty: boolean,
  ) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (isEmpty) onPlace?.(node);
      return;
    }

    let next = node;
    if (
      event.key === 'ArrowUp' ||
      event.key === 'ArrowDown' ||
      event.key === 'ArrowLeft' ||
      event.key === 'ArrowRight'
    ) {
      next = nodeInDirection(board, node, event.key);
    } else if (event.key === 'Home') {
      next = 0;
    } else if (event.key === 'End') {
      next = board.n - 1;
    } else {
      return;
    }

    event.preventDefault();
    focusNode(next);
  };

  return (
    <svg
      viewBox="-119 -119 238 238"
      className={className}
      role={interactive ? 'group' : 'img'}
      aria-label={`*Star board with ${board.rings} rings, ${occupiedCount} of ${board.n} nodes occupied`}
      aria-describedby={instructionsId}
      onMouseLeave={() => hover(-1)}
    >
      <desc id={instructionsId}>
        {interactive
          ? 'Use the arrow keys to move between nodes. Press Enter or Space to place a stone on an empty node.'
          : 'A non-interactive preview of the game board.'}
      </desc>
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
        const nodeKind = quark ? 'quark peri' : peri ? 'peri' : 'interior node';
        const nodeState = isEmpty
          ? `empty ${nodeKind}; ${currentPlayerName} may place here`
          : `${playerNames?.[stone as 0 | 1] || PLAYER_COLORS[stone as 0 | 1].name} stone on ${nodeKind}${
              dead ? ', not currently part of a living star' : ''
            }${u === lastMove ? ', last move' : ''}${
              currentTurnMoves.includes(u) && u !== lastMove ? ', placed this turn' : ''
            }`;

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
              <circle
                className="last-move-pulse"
                cx={x}
                cy={y}
                fill="none"
                stroke="rgba(255,255,255,0.9)"
              >
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

            {interactive && focusedNode === u && (
              <circle
                aria-hidden
                cx={x}
                cy={y}
                r={Math.max(stoneR * 1.45, 5.5)}
                fill="none"
                stroke="rgba(255,255,255,0.95)"
                strokeWidth="1.1"
                pointerEvents="none"
              />
            )}

            {/* hit target */}
            {interactive && (
              <circle
                ref={(element) => {
                  if (element) nodeRefs.current.set(u, element);
                  else nodeRefs.current.delete(u);
                }}
                cx={x}
                cy={y}
                r={Math.max(stoneR, 4)}
                fill="transparent"
                role="button"
                tabIndex={activeNode === u ? 0 : -1}
                aria-label={`Node ${board.labels[u]}, ${nodeState}`}
                aria-disabled={!isEmpty}
                style={{ cursor: isEmpty ? 'pointer' : 'default', outline: 'none' }}
                onMouseEnter={() => hover(u)}
                onFocus={() => {
                  setActiveNode(u);
                  setFocusedNode(u);
                  hover(u);
                }}
                onBlur={() => {
                  setFocusedNode((current) => (current === u ? -1 : current));
                  hover(-1);
                }}
                onKeyDown={(event) => handleNodeKeyDown(event, u, isEmpty)}
                onClick={() => {
                  if (isEmpty) onPlace?.(u);
                }}
              />
            )}
          </g>
        );
      })}
    </svg>
  );
});
