/**
 * Naive reference scorer used only in tests.
 *
 * Implements the same scoring semantics as src/lib/star/scoring.ts but with a
 * deliberately different, simple style (string sets, maps, repeated BFS) so
 * the two can cross-validate each other. Kept independent of the engine's
 * union-find / CSR machinery on purpose — do not "optimize" this.
 */

import type { Board } from '../board';
import { EMPTY } from '../scoring';

export interface RefPlayer {
  peries: number;
  quarks: number;
  stars: number;
  quarkPeri: 0 | 1;
  award: number;
  total: number;
}

export interface RefResult {
  players: [RefPlayer, RefPlayer];
  nodeOwner: number[];
  contestedPeries: number;
}

function neighbors(board: Board, u: number): number[] {
  const out: number[] = [];
  for (let e = board.adjOff[u]; e < board.adjOff[u + 1]; e++) out.push(board.adj[e]);
  return out;
}

export function referenceScore(board: Board, stones: ArrayLike<number>): RefResult {
  const n = board.n;

  // Connected same-color groups via BFS.
  const groupOf = new Array<number>(n).fill(-1);
  const groups: { color: number; cells: number[] }[] = [];
  for (let start = 0; start < n; start++) {
    if (stones[start] === EMPTY || groupOf[start] !== -1) continue;
    const color = stones[start] as number;
    const cells: number[] = [];
    const queue = [start];
    groupOf[start] = groups.length;
    while (queue.length) {
      const u = queue.shift()!;
      cells.push(u);
      for (const v of neighbors(board, u)) {
        if (stones[v] === color && groupOf[v] === -1) {
          groupOf[v] = groups.length;
          queue.push(v);
        }
      }
    }
    groups.push({ color, cells });
  }

  const alive = groups.map(() => true);

  interface Region {
    cells: number[];
    periCount: number;
    boundaryGroups: Set<number>;
    boundaryColors: Set<number>;
  }

  const computeRegions = (): { regions: Region[]; regionOf: number[] } => {
    const regionOf = new Array<number>(n).fill(-1);
    const regions: Region[] = [];
    const isTerritory = (u: number) => stones[u] === EMPTY || !alive[groupOf[u]];
    for (let start = 0; start < n; start++) {
      if (!isTerritory(start) || regionOf[start] !== -1) continue;
      const region: Region = {
        cells: [],
        periCount: 0,
        boundaryGroups: new Set(),
        boundaryColors: new Set(),
      };
      const queue = [start];
      regionOf[start] = regions.length;
      while (queue.length) {
        const u = queue.shift()!;
        region.cells.push(u);
        if (board.isPeri[u]) region.periCount++;
        for (const v of neighbors(board, u)) {
          if (isTerritory(v)) {
            if (regionOf[v] === -1) {
              regionOf[v] = regions.length;
              queue.push(v);
            }
          } else {
            region.boundaryGroups.add(groupOf[v]);
            region.boundaryColors.add(stones[v] as number);
          }
        }
      }
      regions.push(region);
    }
    return { regions, regionOf };
  };

  // Fixed point: demote all groups owning < 2 peries, simultaneously per
  // round, no revival. Ownership credit for survival: occupied peries plus
  // peries of regions bounded by that single group alone.
  for (;;) {
    const { regions } = computeRegions();
    const owned = groups.map((g, gi) =>
      alive[gi] ? g.cells.filter((u) => board.isPeri[u]).length : 0,
    );
    for (const region of regions) {
      if (region.boundaryGroups.size === 1) {
        const [gi] = [...region.boundaryGroups];
        owned[gi] += region.periCount;
      }
    }
    let changed = false;
    for (let gi = 0; gi < groups.length; gi++) {
      if (alive[gi] && owned[gi] < 2) {
        alive[gi] = false;
        changed = true;
      }
    }
    if (!changed) break;
  }

  // Player-level ownership.
  const { regions, regionOf } = computeRegions();
  const nodeOwner = new Array<number>(n).fill(-1);
  const peries = [0, 0];
  const quarks = [0, 0];
  const stars = [0, 0];
  let contestedPeries = 0;

  for (let gi = 0; gi < groups.length; gi++) {
    if (!alive[gi]) continue;
    stars[groups[gi].color]++;
    for (const u of groups[gi].cells) {
      nodeOwner[u] = groups[gi].color;
      if (board.isPeri[u]) {
        peries[groups[gi].color]++;
        if (board.isQuark[u]) quarks[groups[gi].color]++;
      }
    }
  }
  for (const region of regions) {
    const colors = [...region.boundaryColors];
    const owner = colors.length === 1 ? colors[0] : -1;
    for (const u of region.cells) {
      nodeOwner[u] = owner;
      if (board.isPeri[u]) {
        if (owner === -1) contestedPeries++;
        else {
          peries[owner]++;
          if (board.isQuark[u]) quarks[owner]++;
        }
      }
    }
  }
  void regionOf;

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
  }) as [RefPlayer, RefPlayer];

  return { players, nodeOwner, contestedPeries };
}
