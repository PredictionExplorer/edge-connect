/**
 * Scoring engine for *Star.
 *
 * Rules implemented (Kadon rulebook, cross-checked with Wikipedia):
 *
 *   - A star is a connected group of one color CONTAINING (occupying) two or
 *     more peries (edge cells). This is a static test. Wikipedia phrases it
 *     as "owns at least two peries" (occupied or enclosed), but the two
 *     definitions coincide: to enclose a region containing a peri, a group
 *     must seal that region's stretch of the perimeter cycle on both sides,
 *     i.e. occupy at least two peries itself. (The only exceptions are
 *     degenerate wrap-arounds where a single group encloses nearly the whole
 *     perimeter, which cannot occur in play.) The static reading also matches
 *     the rulebook: "A star is a region of connected cells in one color
 *     containing two or more edge cells."
 *
 *   - Groups that are not stars are removed, per the rulebook clarification:
 *     their cells become territory claimable by surrounding stars.
 *
 *   - Territory: maximal connected regions of non-star cells (empty cells and
 *     dead stones). A region — and every peri in it — is owned by a player
 *     when all alive stars bordering the region are that player's (and there
 *     is at least one). Regions bordered by both colors, or by no star at
 *     all, are contested and score for nobody. (In a decided game there are
 *     no contested regions.)
 *
 *   - Player score = owned peries (occupied by their stars + territory)
 *                  + 1 "quark peri" point for owning >= 3 of the 5 corners
 *                  + award of 2 x (opponent star count - own star count).
 *     Tie-break: most quarks. On a decided board the two totals always sum to
 *     (number of peries + 1) - the classic *Star invariant (51 on the
 *     ten-ring board).
 *
 * Performance: one union-find pass over the CSR adjacency (path halving),
 * one flood fill for regions, one aggregation pass - all on preallocated
 * typed arrays with no hashing. O((N + E) * alpha(N)) total; the largest
 * board is 390 nodes / ~1000 edges, so a full evaluation costs on the order
 * of microseconds and is run after every stone placement.
 */

import type { Board } from './board';

export const EMPTY = -1;

export interface PlayerScore {
  /** Peries owned: occupied by this player's stars + enclosed territory. */
  peries: number;
  /** Corner peries (quarks) owned, 0..5. */
  quarks: number;
  /** Number of alive stars. */
  stars: number;
  /** 1 if this player owns three or more quarks. */
  quarkPeri: 0 | 1;
  /** 2 x (opponent stars - own stars). */
  award: number;
  /** peries + quarkPeri + award. */
  total: number;
}

export interface ScoreResult {
  players: [PlayerScore, PlayerScore];
  /**
   * Controller of each node: 0 or 1, or -1 for none/contested. Stones of
   * alive stars map to their color; dead stones and empty nodes map to the
   * player whose stars solely border their region, if any.
   */
  nodeOwner: Int8Array;
  /** 1 for stones that belong to an alive star, 0 otherwise. */
  aliveStone: Uint8Array;
  /** Number of peries owned by neither player at this position. */
  contestedPeries: number;
  /** Player ahead if scored now (0 | 1), or -1 for a dead tie. */
  leader: 0 | 1 | -1;
}

/** Score a position. `stones[u]` is EMPTY (-1), 0, or 1. */
export function scorePosition(board: Board, stones: ArrayLike<number>): ScoreResult {
  const { n, adjOff, adj, isPeri, isQuark } = board;

  // ---- 1. Union-find over same-color adjacent stones ----------------------
  const parent = new Int32Array(n);
  for (let u = 0; u < n; u++) parent[u] = u;
  const find = (u: number): number => {
    let r = u;
    while (parent[r] !== r) {
      parent[r] = parent[parent[r]];
      r = parent[r];
    }
    return r;
  };
  for (let u = 0; u < n; u++) {
    const c = stones[u];
    if (c === EMPTY) continue;
    for (let e = adjOff[u]; e < adjOff[u + 1]; e++) {
      const v = adj[e];
      if (v > u && stones[v] === c) {
        const ru = find(u);
        const rv = find(v);
        if (ru !== rv) parent[rv] = ru;
      }
    }
  }

  // ---- 2. Stars: groups occupying >= 2 peries (static) ---------------------
  const occPeries = new Int32Array(n); // indexed by group root
  for (let u = 0; u < n; u++) {
    if (stones[u] !== EMPTY && isPeri[u]) occPeries[find(u)]++;
  }
  const alive = new Uint8Array(n); // per node: stone of an alive star
  for (let u = 0; u < n; u++) {
    if (stones[u] !== EMPTY && occPeries[find(u)] >= 2) alive[u] = 1;
  }

  // ---- 3. Territory regions over empty cells + dead stones ----------------
  // regionColor: -2 no bordering star, 0/1 sole color, -1 mixed.
  const regionOf = new Int32Array(n).fill(-1);
  const stack = new Int32Array(n);
  const regionColor: number[] = [];
  for (let s = 0; s < n; s++) {
    if (alive[s] || regionOf[s] !== -1) continue;
    const id = regionColor.length;
    let color = -2;
    let top = 0;
    stack[top++] = s;
    regionOf[s] = id;
    while (top > 0) {
      const u = stack[--top];
      for (let e = adjOff[u]; e < adjOff[u + 1]; e++) {
        const v = adj[e];
        if (alive[v]) {
          const cv = stones[v] as number;
          if (color === -2) color = cv;
          else if (color !== cv) color = -1;
        } else if (regionOf[v] === -1) {
          regionOf[v] = id;
          stack[top++] = v;
        }
      }
    }
    regionColor.push(color);
  }

  // ---- 4. Aggregate to players ---------------------------------------------
  const nodeOwner = new Int8Array(n).fill(-1);
  const aliveStone = new Uint8Array(n);
  const peries = [0, 0];
  const quarks = [0, 0];
  const stars = [0, 0];
  let contestedPeries = 0;

  for (let u = 0; u < n; u++) {
    let owner: number;
    if (alive[u]) {
      owner = stones[u] as number;
      aliveStone[u] = 1;
      if (parent[u] === u) stars[owner]++;
    } else {
      owner = regionColor[regionOf[u]];
    }
    if (owner === 0 || owner === 1) {
      nodeOwner[u] = owner;
      if (isPeri[u]) {
        peries[owner]++;
        if (isQuark[u]) quarks[owner]++;
      }
    } else if (isPeri[u]) {
      contestedPeries++;
    }
  }
  // A star's root may be a non-peri stone; roots counted above only when the
  // root node itself is alive — which holds for every alive group since
  // aliveness is a per-group property. (parent[u] === u picks each group
  // exactly once.)

  const players = [0, 1].map((p) => {
    const quarkPeri: 0 | 1 = quarks[p] >= 3 ? 1 : 0;
    const award = 2 * (stars[1 - p] - stars[p]);
    return {
      peries: peries[p],
      quarks: quarks[p],
      stars: stars[p],
      quarkPeri,
      award,
      total: peries[p] + quarkPeri + award,
    };
  }) as [PlayerScore, PlayerScore];

  let leader: 0 | 1 | -1;
  if (players[0].total !== players[1].total) {
    leader = players[0].total > players[1].total ? 0 : 1;
  } else if (players[0].quarks !== players[1].quarks) {
    leader = players[0].quarks > players[1].quarks ? 0 : 1;
  } else {
    leader = -1;
  }

  return { players, nodeOwner, aliveStone, contestedPeries, leader };
}
