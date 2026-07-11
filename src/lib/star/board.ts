/**
 * Board model for *Star.
 *
 * The board is a pentagonal mesh of nodes arranged in concentric rings.
 * A node is addressed by (sector s, ring x, position y):
 *   - s ∈ 0..4 — the five sectors *, S, T, A, R, proceeding clockwise,
 *     each bounded by rays from the center through the board corners;
 *   - x ∈ 1..rings — ring number, 1 at the center, `rings` on the perimeter;
 *   - y ∈ 0..x-1 — tangential position, clockwise from the sector's radial arm
 *     (y = 0 lies on the arm itself).
 *
 * Ring x holds 5x nodes, so an r-ring board has 5·r(r+1)/2 nodes, of which the
 * 5r on ring r are perimeter nodes ("peries") and the five arm nodes of ring r
 * are corners ("quarks"). This reproduces the published sizes: 6 rings = 105
 * nodes / 30 peries, 8 = 180/40, 10 = 275/50.
 *
 * Adjacency (a full triangulation between consecutive ring pentagons):
 *   - in-ring cycle:  (s,x,y)–(s,x,y+1), wrapping (s,x,x-1)–(s+1,x,0)
 *   - radial:         (s,x,y)–(s,x-1,y)     for y ≤ x-2
 *   - diagonal:       (s,x,y)–(s,x-1,y-1)   for y ≥ 1
 *   - corner cross:   (s,x,x-1)–(s+1,x-1,0)
 *   - bridge:         the five ring-1 nodes are mutually connected (K5) by the
 *     star-shaped bridge at the center. The bridge is not itself playable.
 *
 * Adjacency is stored in CSR form (adjOff/adj) over dense node ids so that the
 * scoring engine can traverse it with zero allocation.
 */

import { STAR_RULES_CONTRACT } from './rules';

export const SECTOR_CHARS = ['*', 'S', 'T', 'A', 'R'] as const;

export const SUPPORTED_RINGS = STAR_RULES_CONTRACT.board.supportedRings;
export type SupportedRings = (typeof SUPPORTED_RINGS)[number];
export const MIN_RINGS = SUPPORTED_RINGS[0];
export const MAX_RINGS = SUPPORTED_RINGS[SUPPORTED_RINGS.length - 1];

export function isSupportedRings(value: unknown): value is SupportedRings {
  return (
    typeof value === 'number' &&
    Number.isInteger(value) &&
    SUPPORTED_RINGS.includes(value as SupportedRings)
  );
}

export interface Board {
  rings: number;
  /** Total number of nodes: 5·rings·(rings+1)/2 */
  n: number;
  /** Number of perimeter nodes: 5·rings */
  periCount: number;
  sectorOf: Int8Array;
  ringOf: Int8Array;
  posOf: Int8Array;
  isPeri: Uint8Array;
  isQuark: Uint8Array;
  /** Official Nxy label per node, e.g. "T43" (ring 10 is written as 0). */
  labels: string[];
  /** CSR adjacency: neighbors of u are adj[adjOff[u] .. adjOff[u+1]-1]. */
  adjOff: Int32Array;
  adj: Int32Array;
  /** Unit layout coordinates (board circumradius 1, screen y-down). */
  xs: Float64Array;
  ys: Float64Array;
  /** The five ring-1 node ids (clockwise), joined by the central bridge. */
  bridge: number[];
  /** Length of the shortest edge in layout units (for sizing stones). */
  minEdge: number;
  idx(s: number, x: number, y: number): number;
  labelToId: Map<string, number>;
}

function ringStart(x: number): number {
  return (5 * x * (x - 1)) / 2;
}

function ringChar(x: number): string {
  return x === 10 ? '0' : String(x);
}

const boardCache = new Map<number, Board>();

export function getBoard(rings: number): Board {
  if (!isSupportedRings(rings)) {
    throw new Error(
      `rings must be one of ${SUPPORTED_RINGS.join(', ')}, got ${String(rings)}`,
    );
  }
  const cached = boardCache.get(rings);
  if (cached) return cached;
  const board = buildBoard(rings);
  boardCache.set(rings, board);
  return board;
}

function buildBoard(rings: number): Board {
  const n = ringStart(rings + 1);
  const idx = (s: number, x: number, y: number): number => {
    const ss = ((s % 5) + 5) % 5;
    return ringStart(x) + ss * x + y;
  };

  const sectorOf = new Int8Array(n);
  const ringOf = new Int8Array(n);
  const posOf = new Int8Array(n);
  const isPeri = new Uint8Array(n);
  const isQuark = new Uint8Array(n);
  const labels = new Array<string>(n);
  const labelToId = new Map<string, number>();

  for (let x = 1; x <= rings; x++) {
    for (let s = 0; s < 5; s++) {
      for (let y = 0; y < x; y++) {
        const u = idx(s, x, y);
        sectorOf[u] = s;
        ringOf[u] = x;
        posOf[u] = y;
        if (x === rings) {
          isPeri[u] = 1;
          if (y === 0) isQuark[u] = 1;
        }
        const label = `${SECTOR_CHARS[s]}${ringChar(x)}${y}`;
        labels[u] = label;
        labelToId.set(label, u);
      }
    }
  }

  // --- Edges ---------------------------------------------------------------
  const edgeKeys = new Set<number>();
  const edges: number[] = [];
  const addEdge = (a: number, b: number) => {
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    const key = lo * 512 + hi; // n ≤ 275 < 512
    if (edgeKeys.has(key)) return;
    edgeKeys.add(key);
    edges.push(lo, hi);
  };

  for (let x = 1; x <= rings; x++) {
    for (let s = 0; s < 5; s++) {
      for (let y = 0; y < x; y++) {
        const u = idx(s, x, y);
        // in-ring cycle (clockwise successor)
        addEdge(u, y < x - 1 ? idx(s, x, y + 1) : idx(s + 1, x, 0));
        if (x >= 2) {
          if (y <= x - 2) addEdge(u, idx(s, x - 1, y)); // radial
          if (y >= 1) addEdge(u, idx(s, x - 1, y - 1)); // diagonal
          if (y === x - 1) addEdge(u, idx(s + 1, x - 1, 0)); // corner cross
        }
      }
    }
  }
  // Central bridge: K5 over the ring-1 nodes.
  const bridge: number[] = [];
  for (let s = 0; s < 5; s++) bridge.push(idx(s, 1, 0));
  for (let i = 0; i < 5; i++) {
    for (let j = i + 1; j < 5; j++) addEdge(bridge[i], bridge[j]);
  }

  // --- CSR -----------------------------------------------------------------
  const degree = new Int32Array(n);
  const edgeCount = edges.length / 2;
  for (let e = 0; e < edgeCount; e++) {
    degree[edges[2 * e]]++;
    degree[edges[2 * e + 1]]++;
  }
  const adjOff = new Int32Array(n + 1);
  for (let u = 0; u < n; u++) adjOff[u + 1] = adjOff[u] + degree[u];
  const adj = new Int32Array(2 * edgeCount);
  const cursor = adjOff.slice(0, n);
  for (let e = 0; e < edgeCount; e++) {
    const a = edges[2 * e];
    const b = edges[2 * e + 1];
    adj[cursor[a]++] = b;
    adj[cursor[b]++] = a;
  }

  // --- Layout ----------------------------------------------------------------
  // Arm s points at screen angle 54° + 72°·s (y-down ⇒ increasing angle is
  // clockwise): one corner points straight up and the '*' arm sits at the
  // lower right, as in the official notation diagram. Ring-x corners sit at
  // radius x/rings; side nodes are interpolated along the pentagon side
  // toward the next arm's corner.
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  const armX = new Float64Array(6);
  const armY = new Float64Array(6);
  for (let s = 0; s <= 5; s++) {
    const a = ((54 + 72 * s) * Math.PI) / 180;
    armX[s] = Math.cos(a);
    armY[s] = Math.sin(a);
  }
  for (let x = 1; x <= rings; x++) {
    const r = x / rings;
    for (let s = 0; s < 5; s++) {
      for (let y = 0; y < x; y++) {
        const u = idx(s, x, y);
        const t = y / x;
        xs[u] = r * (armX[s] * (1 - t) + armX[s + 1] * t);
        ys[u] = r * (armY[s] * (1 - t) + armY[s + 1] * t);
      }
    }
  }
  let minEdge = Infinity;
  for (let e = 0; e < edgeCount; e++) {
    const a = edges[2 * e];
    const b = edges[2 * e + 1];
    // The bridge K5 chords are not geometric mesh edges; skip them when
    // measuring spacing (ring-1 cycle edges still count).
    if (ringOf[a] === 1 && ringOf[b] === 1 && (sectorOf[a] + 1) % 5 !== sectorOf[b] && (sectorOf[b] + 1) % 5 !== sectorOf[a]) {
      continue;
    }
    const d = Math.hypot(xs[a] - xs[b], ys[a] - ys[b]);
    if (d < minEdge) minEdge = d;
  }

  return {
    rings,
    n,
    periCount: 5 * rings,
    sectorOf,
    ringOf,
    posOf,
    isPeri,
    isQuark,
    labels,
    adjOff,
    adj,
    xs,
    ys,
    bridge,
    minEdge,
    idx,
    labelToId,
  };
}

/** Parse an official label like "T43" (sector char, ring char, position). */
export function parseLabel(board: Board, label: string): number {
  const id = board.labelToId.get(label);
  if (id === undefined) throw new Error(`unknown node label: ${label}`);
  return id;
}

/** Perimeter node ids in clockwise cycle order. */
export function perimeterCycle(board: Board): number[] {
  const out: number[] = [];
  for (let s = 0; s < 5; s++) {
    for (let y = 0; y < board.rings; y++) out.push(board.idx(s, board.rings, y));
  }
  return out;
}

/** Node ids along each radial arm (center to corner), per sector. */
export function armPaths(board: Board): number[][] {
  const out: number[][] = [];
  for (let s = 0; s < 5; s++) {
    const path: number[] = [];
    for (let x = 1; x <= board.rings; x++) path.push(board.idx(s, x, 0));
    out.push(path);
  }
  return out;
}
