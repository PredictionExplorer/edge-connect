'use client';

import {
  memo,
  useCallback,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
  type PointerEvent,
} from 'react';
import type { Board } from '@/lib/star/board';
import { EMPTY } from '@/lib/star/scoring';
import { PLAYER_COLORS } from './theme';

const S = 100; // unit-coordinate scale

interface GroupPresentation {
  groupOf: Int32Array;
  groupSize: Int32Array;
  connectionPaths: [string, string];
  starConnectionPaths: [string, string];
  groupPaths: string[];
}

function buildGroupPresentation(
  board: Board,
  stones: ArrayLike<number>,
  aliveStone?: ArrayLike<number> | null,
): GroupPresentation {
  const parent = new Int32Array(board.n).fill(-1);
  for (let node = 0; node < board.n; node++) {
    if (stones[node] !== EMPTY) parent[node] = node;
  }

  const find = (node: number): number => {
    let root = node;
    while (parent[root] !== root) {
      parent[root] = parent[parent[root]];
      root = parent[root];
    }
    return root;
  };

  for (let node = 0; node < board.n; node++) {
    const color = stones[node];
    if (color === EMPTY) continue;
    for (let edge = board.adjOff[node]; edge < board.adjOff[node + 1]; edge++) {
      const neighbor = board.adj[edge];
      if (neighbor <= node || stones[neighbor] !== color) continue;
      const nodeRoot = find(node);
      const neighborRoot = find(neighbor);
      if (nodeRoot !== neighborRoot) parent[neighborRoot] = nodeRoot;
    }
  }

  const groupOf = new Int32Array(board.n).fill(-1);
  const groupSize = new Int32Array(board.n);
  for (let node = 0; node < board.n; node++) {
    if (stones[node] === EMPTY) continue;
    const root = find(node);
    groupOf[node] = root;
    groupSize[root]++;
  }

  const connections: [string[], string[]] = [[], []];
  const starConnections: [string[], string[]] = [[], []];
  const groupSegments = Array.from({ length: board.n }, () => [] as string[]);
  for (let node = 0; node < board.n; node++) {
    const color = stones[node];
    if (color !== 0 && color !== 1) continue;
    for (let edge = board.adjOff[node]; edge < board.adjOff[node + 1]; edge++) {
      const neighbor = board.adj[edge];
      if (neighbor <= node || stones[neighbor] !== color) continue;
      const segment =
        `M${(board.xs[node] * S).toFixed(2)} ${(board.ys[node] * S).toFixed(2)}` +
        `L${(board.xs[neighbor] * S).toFixed(2)} ${(board.ys[neighbor] * S).toFixed(2)}`;
      connections[color].push(segment);
      if (aliveStone?.[node] === 1 && aliveStone[neighbor] === 1) {
        starConnections[color].push(segment);
      }
      groupSegments[groupOf[node]].push(segment);
    }
  }

  return {
    groupOf,
    groupSize,
    connectionPaths: [connections[0].join(''), connections[1].join('')],
    starConnectionPaths: [starConnections[0].join(''), starConnections[1].join('')],
    groupPaths: groupSegments.map((segments) => segments.join('')),
  };
}

export interface StarBoardProps {
  board: Board;
  stones: ArrayLike<number>;
  /** Node controller from the scorer, for the territory overlay. */
  nodeOwner?: ArrayLike<number> | null;
  /** 1 = stone currently belongs to a living star. */
  aliveStone?: ArrayLike<number> | null;
  /** 1 = existing stone cannot form a living star in any completion. */
  provablyDeadStone?: ArrayLike<number> | null;
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
  provablyDeadStone,
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
  const svgRef = useRef<SVGSVGElement>(null);
  const nodeRefs = useRef(new Map<number, SVGCircleElement>());
  const instructionsId = useId();

  const stoneR = Math.min(board.minEdge * 0.46 * S, 9.5);
  const groups = useMemo(
    () => buildGroupPresentation(board, stones, aliveStone),
    [aliveStone, board, stones],
  );

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

  const hover = useCallback((u: number) => {
    setHovered(u);
    onHover?.(u);
  }, [onHover]);

  const nodeAtPointer = useCallback(
    (clientX: number, clientY: number): number => {
      const svg = svgRef.current;
      if (!svg) return -1;
      const rect = svg.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return -1;

      const scale = Math.min(rect.width / 238, rect.height / 238);
      const renderedWidth = 238 * scale;
      const renderedHeight = 238 * scale;
      const originX = rect.left + (rect.width - renderedWidth) / 2;
      const originY = rect.top + (rect.height - renderedHeight) / 2;
      const x = (clientX - originX) / scale - 119;
      const y = (clientY - originY) / scale - 119;
      const maxDistance = Math.max(board.minEdge * S * 0.56, stoneR * 1.25);
      let nearest = -1;
      let nearestDistance = maxDistance;

      for (let node = 0; node < board.n; node++) {
        const distance = Math.hypot(board.xs[node] * S - x, board.ys[node] * S - y);
        if (distance <= nearestDistance) {
          nearest = node;
          nearestDistance = distance;
        }
      }
      return nearest;
    },
    [board, stoneR],
  );

  const handlePointerMove = useCallback(
    (event: PointerEvent<SVGSVGElement>) => {
      if (!interactive) return;
      const node = nodeAtPointer(event.clientX, event.clientY);
      if (node !== hovered) hover(node);
    },
    [hover, hovered, interactive, nodeAtPointer],
  );

  const handleBoardClick = useCallback(
    (event: MouseEvent<SVGSVGElement>) => {
      if (!interactive) return;
      const node = nodeAtPointer(event.clientX, event.clientY);
      if (node >= 0 && stones[node] === EMPTY) onPlace?.(node);
    },
    [interactive, nodeAtPointer, onPlace, stones],
  );

  const territory = showTerritory && nodeOwner ? nodeOwner : null;
  const occupiedCount = Array.from({ length: board.n }, (_, node) => stones[node]).filter(
    (stone) => stone !== EMPTY,
  ).length;
  const currentPlayerName = playerNames?.[toMove] || PLAYER_COLORS[toMove].name;
  const highlightedGroup =
    hovered >= 0 && stones[hovered] !== EMPTY ? groups.groupOf[hovered] : -1;
  const highlightedColor =
    highlightedGroup >= 0 ? (stones[hovered] as 0 | 1) : null;

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
      ref={svgRef}
      viewBox="-119 -119 238 238"
      className={className}
      role={interactive ? 'group' : 'img'}
      aria-label={`*Star board with ${board.rings} rings, ${occupiedCount} of ${board.n} nodes occupied`}
      aria-describedby={instructionsId}
      onPointerMove={handlePointerMove}
      onMouseLeave={() => hover(-1)}
      onClick={handleBoardClick}
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
        <filter id="connectionGlow" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="0.8" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
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

      {/* Same-color graph connections. The brighter pass marks living stars. */}
      <g aria-hidden pointerEvents="none">
        {PLAYER_COLORS.map((color, player) => (
          <path
            key={`connections-${player}`}
            data-connection-layer="group"
            data-player={player}
            d={groups.connectionPaths[player as 0 | 1]}
            stroke={color.deep}
            strokeWidth={Math.max(stoneR * 0.48, 1.2)}
            strokeOpacity="0.72"
            fill="none"
            strokeLinecap="round"
          />
        ))}
        {PLAYER_COLORS.map((color, player) => (
          <path
            key={`star-connections-${player}`}
            data-connection-layer="star"
            data-player={player}
            d={groups.starConnectionPaths[player as 0 | 1]}
            stroke={color.base}
            strokeWidth={Math.max(stoneR * 0.24, 0.7)}
            strokeOpacity="0.82"
            fill="none"
            strokeLinecap="round"
            filter="url(#connectionGlow)"
          />
        ))}
        {highlightedGroup >= 0 && highlightedColor !== null && (
          <path
            data-connection-layer="highlight"
            d={groups.groupPaths[highlightedGroup]}
            stroke={PLAYER_COLORS[highlightedColor].bright}
            strokeWidth={Math.max(stoneR * 0.68, 1.6)}
            strokeOpacity="0.95"
            fill="none"
            strokeLinecap="round"
            filter="url(#connectionGlow)"
          />
        )}
      </g>

      {/* Node layers */}
      {Array.from({ length: board.n }, (_, u) => {
        const x = board.xs[u] * S;
        const y = board.ys[u] * S;
        const stone = stones[u];
        const isEmpty = stone === EMPTY;
        const owner = territory ? territory[u] : -1;
        const quark = board.isQuark[u] === 1;
        const peri = board.isPeri[u] === 1;
        const notInStar = !isEmpty && aliveStone ? aliveStone[u] === 0 : false;
        const provablyDead = !isEmpty && provablyDeadStone?.[u] === 1;
        const dimmed = provablyDead || (showTerritory && notInStar);
        const inHighlightedGroup =
          !isEmpty && highlightedGroup >= 0 && groups.groupOf[u] === highlightedGroup;
        const nodeKind = quark ? 'quark peri' : peri ? 'peri' : 'interior node';
        const nodeState = isEmpty
          ? `empty ${nodeKind}; ${currentPlayerName} may place here`
          : `${playerNames?.[stone as 0 | 1] || PLAYER_COLORS[stone as 0 | 1].name} stone on ${nodeKind}${
              u === lastMove ? ', last move' : ''
            }${
              currentTurnMoves.includes(u) && u !== lastMove ? ', placed this turn' : ''
            }${
              provablyDead
                ? ', provably dead; cannot form a living star in any completion'
                : notInStar
                  ? `, connected group of ${groups.groupSize[groups.groupOf[u]]} stone${
                      groups.groupSize[groups.groupOf[u]] === 1 ? '' : 's'
                    }, not currently part of a living star`
                  : `, part of a living star with ${groups.groupSize[groups.groupOf[u]]} stones`
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
            {territory && owner !== -1 && (isEmpty || notInStar) && (
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
              <g
                className="stone-pop"
                opacity={provablyDead ? 0.22 : dimmed ? 0.35 : 1}
                data-group-root={groups.groupOf[u]}
                data-stone-node={u}
              >
                <circle
                  cx={x}
                  cy={y}
                  r={stoneR}
                  fill={`url(#stone${stone})`}
                  stroke={PLAYER_COLORS[stone as 0 | 1].deep}
                  strokeWidth="0.5"
                  filter={dimmed ? undefined : 'url(#stoneShadow)'}
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

            {inHighlightedGroup && (
              <circle
                aria-hidden
                data-group-highlight={u}
                cx={x}
                cy={y}
                r={stoneR * 1.18}
                fill="none"
                stroke={PLAYER_COLORS[stone as 0 | 1].bright}
                strokeWidth="0.8"
                strokeOpacity="0.9"
                pointerEvents="none"
              />
            )}

            {provablyDead && (
              <g
                aria-hidden
                data-provably-dead-stone={u}
                pointerEvents="none"
                stroke="rgba(255,125,104,0.95)"
                strokeWidth={Math.max(stoneR * 0.14, 0.8)}
                strokeLinecap="round"
              >
                <circle
                  cx={x}
                  cy={y}
                  r={stoneR * 0.92}
                  fill="none"
                  strokeDasharray="1.6 1.6"
                  strokeWidth={Math.max(stoneR * 0.1, 0.65)}
                />
                <path
                  d={`M${x - stoneR * 0.42} ${y - stoneR * 0.42}L${x + stoneR * 0.42} ${
                    y + stoneR * 0.42
                  }M${x + stoneR * 0.42} ${y - stoneR * 0.42}L${
                    x - stoneR * 0.42
                  } ${y + stoneR * 0.42}`}
                  fill="none"
                />
              </g>
            )}

            {/* current-turn + last-move markers */}
            {currentTurnMoves.includes(u) && u !== lastMove && (
              <circle cx={x} cy={y} r={stoneR * 0.22} fill="rgba(255,255,255,0.85)" />
            )}
            {u === lastMove && (
              <g aria-hidden pointerEvents="none">
                <circle
                  data-last-move={u}
                  cx={x}
                  cy={y}
                  r={stoneR * 1.18}
                  fill="none"
                  stroke="rgba(255,255,255,0.9)"
                  strokeWidth="0.8"
                />
                <circle
                  className="last-move-pulse"
                  cx={x}
                  cy={y}
                  fill="none"
                  stroke="rgba(255,255,255,0.75)"
                >
                  <animate
                    attributeName="r"
                    values={`${stoneR * 1.18};${stoneR * 1.65}`}
                    dur="0.8s"
                    repeatCount="1"
                    fill="freeze"
                  />
                  <animate
                    attributeName="stroke-opacity"
                    values="0.75;0"
                    dur="0.8s"
                    repeatCount="1"
                    fill="freeze"
                  />
                </circle>
              </g>
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
                onClick={(event) => {
                  event.stopPropagation();
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
