import { describe, expect, it } from 'vitest';
import { getBoard, SUPPORTED_RINGS, type Board } from '../board';
import {
  applyAction,
  initialState,
  replay,
  type GameAction,
  type GameConfig,
  type GameState,
} from '../game';
import { EMPTY, scorePosition } from '../scoring';
import {
  composeD5Symmetries,
  D5_ROTATIONS,
  D5_SYMMETRIES,
  getD5Maps,
  inverseD5Symmetry,
  transformActions,
  transformNode,
  transformStones,
  type D5Symmetry,
} from '../symmetry';

function neighborSets(board: Board): Set<number>[] {
  return Array.from({ length: board.n }, (_, u) => {
    const neighbors = new Set<number>();
    for (let e = board.adjOff[u]; e < board.adjOff[u + 1]; e++) {
      neighbors.add(board.adj[e]);
    }
    return neighbors;
  });
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function expectStateEquivariant(
  original: GameState,
  transformed: GameState,
  symmetry: D5Symmetry,
): void {
  const board = original.board;
  expect(Array.from(transformed.stones)).toEqual(
    Array.from(transformStones(board, original.stones, symmetry)),
  );
  expect({
    stonesPlaced: transformed.stonesPlaced,
    toMove: transformed.toMove,
    movesLeft: transformed.movesLeft,
    midTurn: transformed.midTurn,
    over: transformed.over,
    canSwap: transformed.canSwap,
    swapped: transformed.swapped,
    turnCount: transformed.turnCount,
  }).toEqual({
    stonesPlaced: original.stonesPlaced,
    toMove: original.toMove,
    movesLeft: original.movesLeft,
    midTurn: original.midTurn,
    over: original.over,
    canSwap: original.canSwap,
    swapped: original.swapped,
    turnCount: original.turnCount,
  });
  expect(transformed.lastMove).toBe(
    original.lastMove === -1
      ? -1
      : transformNode(board, original.lastMove, symmetry),
  );
  expect(transformed.currentTurnMoves).toEqual(
    original.currentTurnMoves.map((node) =>
      transformNode(board, node, symmetry),
    ),
  );
}

describe('D5 board symmetries', () => {
  it('uses the exact sector/ring/position formulas', () => {
    const board = getBoard(8);
    for (let u = 0; u < board.n; u++) {
      const s = board.sectorOf[u];
      const x = board.ringOf[u];
      const y = board.posOf[u];
      for (let turns = 0; turns < 5; turns++) {
        expect(transformNode(board, u, D5_ROTATIONS[turns])).toBe(
          board.idx(s + turns, x, y),
        );
      }

      const reflected =
        y === 0
          ? board.idx(-s, x, 0)
          : board.idx(-s - 1, x, x - y);
      expect(transformNode(board, u, D5_SYMMETRIES[5])).toBe(reflected);
    }
  });

  it('is a graph automorphism on every supported board', () => {
    for (const rings of SUPPORTED_RINGS) {
      const board = getBoard(rings);
      const neighbors = neighborSets(board);
      for (const symmetry of D5_SYMMETRIES) {
        const { forward, inverse } = getD5Maps(board, symmetry);
        const inverseForward = getD5Maps(
          board,
          inverseD5Symmetry(symmetry),
        ).forward;
        expect(new Set(forward).size).toBe(board.n);
        expect(Array.from(inverse)).toEqual(Array.from(inverseForward));

        for (let u = 0; u < board.n; u++) {
          const mapped = forward[u];
          expect(inverse[mapped]).toBe(u);
          expect(board.ringOf[mapped]).toBe(board.ringOf[u]);
          expect(board.isPeri[mapped]).toBe(board.isPeri[u]);
          expect(board.isQuark[mapped]).toBe(board.isQuark[u]);
          for (const v of neighbors[u]) {
            expect(neighbors[mapped].has(forward[v])).toBe(true);
          }
        }
      }
    }
  });

  it('obeys D5 composition and inverse laws', () => {
    const board = getBoard(6);
    for (const after of D5_SYMMETRIES) {
      const inverse = inverseD5Symmetry(after);
      expect(composeD5Symmetries(inverse, after).id).toBe('r0');
      expect(composeD5Symmetries(after, inverse).id).toBe('r0');
      for (const before of D5_SYMMETRIES) {
        const composed = composeD5Symmetries(after, before);
        for (let u = 0; u < board.n; u++) {
          expect(transformNode(board, u, composed)).toBe(
            transformNode(
              board,
              transformNode(board, u, before),
              after,
            ),
          );
        }
      }
    }
  });

  it('round-trips complete stone arrays on every board size', () => {
    for (const rings of SUPPORTED_RINGS) {
      const board = getBoard(rings);
      const stones = new Int8Array(board.n);
      for (let u = 0; u < board.n; u++) {
        stones[u] = (u % 3) - 1;
      }
      for (const symmetry of D5_SYMMETRIES) {
        const transformed = transformStones(board, stones, symmetry);
        const roundTrip = transformStones(
          board,
          transformed,
          inverseD5Symmetry(symmetry),
        );
        expect(Array.from(roundTrip)).toEqual(Array.from(stones));
      }
    }
  });
});

describe('D5 scoring equivariance', () => {
  it('preserves scores and transforms per-node results', () => {
    const rng = mulberry32(0xd5e91a);
    for (const rings of SUPPORTED_RINGS) {
      const board = getBoard(rings);
      for (const density of [0.08, 0.45, 0.9]) {
        const stones = new Int8Array(board.n).fill(EMPTY);
        for (let u = 0; u < board.n; u++) {
          if (rng() < density) stones[u] = rng() < 0.5 ? 0 : 1;
        }
        const original = scorePosition(board, stones);

        for (const symmetry of D5_SYMMETRIES) {
          const transformed = scorePosition(
            board,
            transformStones(board, stones, symmetry),
          );
          expect(transformed.players).toEqual(original.players);
          expect(transformed.contestedPeries).toBe(
            original.contestedPeries,
          );
          expect(transformed.leader).toBe(original.leader);
          expect(Array.from(transformed.nodeOwner)).toEqual(
            Array.from(
              transformStones(board, original.nodeOwner, symmetry),
            ),
          );
          expect(Array.from(transformed.aliveStone)).toEqual(
            Array.from(
              transformStones(board, original.aliveStone, symmetry),
            ),
          );
        }
      }
    }
  });
});

describe('D5 Double *Star transition and replay equivariance', () => {
  it('commutes with every action, intermediate transition, and replay', () => {
    const config: GameConfig = {
      rings: 6,
      mode: 'double',
      pieRule: false,
      playerNames: ['Aurora', 'Vega'],
    };
    const board = getBoard(config.rings);
    const log: GameAction[] = [
      { type: 'place', node: board.idx(0, 6, 0) },
      { type: 'place', node: board.idx(1, 4, 1) },
      { type: 'place', node: board.idx(2, 4, 2) },
      { type: 'place', node: board.idx(3, 6, 0) },
      { type: 'place', node: board.idx(4, 6, 1) },
      { type: 'place', node: board.idx(0, 4, 2) },
      { type: 'place', node: board.idx(2, 5, 3) },
    ];

    for (const symmetry of D5_SYMMETRIES) {
      expect(
        transformActions(
          board,
          [{ type: 'place', node: 0 }, { type: 'swap' }],
          symmetry,
        ),
      ).toEqual([
        { type: 'place', node: transformNode(board, 0, symmetry) },
        { type: 'swap' },
      ]);
      const transformedLog = transformActions(board, log, symmetry);
      expect(
        transformActions(
          board,
          transformedLog,
          inverseD5Symmetry(symmetry),
        ),
      ).toEqual(log);

      let original = initialState(config);
      let transformed = initialState(config);
      expectStateEquivariant(original, transformed, symmetry);
      for (let i = 0; i < log.length; i++) {
        original = applyAction(original, log[i]);
        transformed = applyAction(transformed, transformedLog[i]);
        expectStateEquivariant(original, transformed, symmetry);
      }

      expectStateEquivariant(
        replay(config, log),
        replay(config, transformedLog),
        symmetry,
      );
    }
  });
});
