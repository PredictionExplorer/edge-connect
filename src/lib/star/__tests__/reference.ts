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
  aliveStone: number[];
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

  // Official static star test: survival depends only on perimeter nodes the
  // group directly occupies. Territory is scored after dead groups are
  // removed and must never bootstrap a group's survival.
  const alive = groups.map(
    (group) => group.cells.filter((u) => board.isPeri[u]).length >= 2,
  );

  interface Region {
    cells: number[];
    boundaryColors: Set<number>;
  }

  const computeRegions = (): Region[] => {
    const regionOf = new Array<number>(n).fill(-1);
    const regions: Region[] = [];
    const isTerritory = (u: number) => stones[u] === EMPTY || !alive[groupOf[u]];
    for (let start = 0; start < n; start++) {
      if (!isTerritory(start) || regionOf[start] !== -1) continue;
      const region: Region = {
        cells: [],
        boundaryColors: new Set(),
      };
      const queue = [start];
      regionOf[start] = regions.length;
      while (queue.length) {
        const u = queue.shift()!;
        region.cells.push(u);
        for (const v of neighbors(board, u)) {
          if (isTerritory(v)) {
            if (regionOf[v] === -1) {
              regionOf[v] = regions.length;
              queue.push(v);
            }
          } else {
            region.boundaryColors.add(stones[v] as number);
          }
        }
      }
      regions.push(region);
    }
    return regions;
  };

  // Player-level ownership.
  const regions = computeRegions();
  const nodeOwner = new Array<number>(n).fill(-1);
  const aliveStone = new Array<number>(n).fill(0);
  const peries = [0, 0];
  const quarks = [0, 0];
  const stars = [0, 0];
  let contestedPeries = 0;

  for (let gi = 0; gi < groups.length; gi++) {
    if (!alive[gi]) continue;
    stars[groups[gi].color]++;
    for (const u of groups[gi].cells) {
      aliveStone[u] = 1;
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

  return { players, nodeOwner, aliveStone, contestedPeries };
}
