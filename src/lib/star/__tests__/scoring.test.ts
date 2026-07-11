import { describe, expect, it } from 'vitest';
import {
  getBoard,
  parseLabel,
  SUPPORTED_RINGS,
  type Board,
} from '../board';
import { EMPTY, scorePosition, validateTerminalWinner } from '../scoring';
import { referenceScore } from './reference';

function position(board: Board, blue: string[], red: string[]): Int8Array {
  const stones = new Int8Array(board.n).fill(EMPTY);
  for (const label of blue) stones[parseLabel(board, label)] = 0;
  for (const label of red) stones[parseLabel(board, label)] = 1;
  return stones;
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

describe('scorePosition: hand-built fixtures', () => {
  it('scores bridge-connected stars, separate stars and dead stones (A)', () => {
    const b = getBoard(4);
    const stones = position(
      b,
      // Blue: S arm + A arm joined through the central bridge = ONE star.
      ['S10', 'S20', 'S30', 'S40', 'A10', 'A20', 'A30', 'A40', 'A41'],
      // Red: * arm + T arm joined through the bridge (one star), a separate
      // two-peri star R41-R42 (kept clear of *40, which is ring-adjacent to
      // R43), and a dead lone stone on peri S42.
      ['*10', '*20', '*30', '*40', '*41', 'T10', 'T20', 'T30', 'T40', 'R41', 'R42', 'S42'],
    );
    const r = scorePosition(b, stones);
    expect(r.players[0]).toEqual({
      peries: 3, // S40, A40, A41
      quarks: 2, // S40, A40
      stars: 1,
      quarkPeri: 0,
      award: 2, // 2 x (2 - 1)
      total: 5,
    });
    expect(r.players[1]).toEqual({
      peries: 5, // *40, *41, T40, R41, R42
      quarks: 2, // *40, T40
      stars: 2,
      quarkPeri: 0,
      award: -2,
      total: 3,
    });
    expect(r.contestedPeries).toBe(12);
    expect(r.leader).toBe(0);
    // The lone red stone on S42 is dead and its region touches both colors.
    expect(r.aliveStone[parseLabel(b, 'S42')]).toBe(0);
    expect(r.nodeOwner[parseLabel(b, 'S42')]).toBe(-1);
  });

  it('gives an enclosed dead stone’s peri to the surrounding star (B)', () => {
    const b = getBoard(4);
    // Red walls off the corner peri *43 completely: *42 (ring cycle), S40
    // (cycle wrap), *32 (diagonal), S30 (corner cross). A lone blue stone on
    // *43 is dead; its peri is claimed by the surrounding red star — the
    // situation called out in Wikipedia's scoring example 2.
    const stones = position(b, ['*43', 'T42', 'T43'], ['*42', '*32', 'S30', 'S40']);
    const r = scorePosition(b, stones);
    expect(r.players[1]).toEqual({
      peries: 3, // *42, S40 occupied + *43 enclosed
      quarks: 1, // S40
      stars: 1,
      quarkPeri: 0,
      award: 0,
      total: 3,
    });
    expect(r.players[0]).toEqual({
      peries: 2, // T42, T43
      quarks: 0,
      stars: 1,
      quarkPeri: 0,
      award: 0,
      total: 2,
    });
    const dead = parseLabel(b, '*43');
    expect(r.aliveStone[dead]).toBe(0);
    expect(r.nodeOwner[dead]).toBe(1);
    // Same position with *43 empty instead: red still owns the peri.
    stones[dead] = EMPTY;
    const r2 = scorePosition(b, stones);
    expect(r2.players[1].peries).toBe(3);
    expect(r2.nodeOwner[dead]).toBe(1);
  });

  it('awards the quark peri for three corners and the star-count award (C)', () => {
    const b = getBoard(4);
    const stones = position(
      b,
      // Blue: one six-peri star spanning the A and R sectors.
      ['A40', 'A41', 'A42', 'A43', 'R40', 'R41'],
      // Red: three separate two-stone corner stars.
      ['*40', '*41', 'S40', 'S41', 'T40', 'T41'],
    );
    const r = scorePosition(b, stones);
    expect(r.players[0]).toEqual({
      peries: 6,
      quarks: 2, // A40, R40
      stars: 1,
      quarkPeri: 0,
      award: 4,
      total: 10,
    });
    expect(r.players[1]).toEqual({
      peries: 6,
      quarks: 3, // *40, S40, T40
      stars: 3,
      quarkPeri: 1,
      award: -4,
      total: 3,
    });
    expect(r.leader).toBe(0);
  });

  it('breaks ties by quark count (D)', () => {
    const b = getBoard(4);
    const r = scorePosition(b, position(b, ['*40', '*41'], ['S41', 'S42']));
    expect(r.players[0].total).toBe(2);
    expect(r.players[1].total).toBe(2);
    expect(r.players[0].quarks).toBe(1);
    expect(r.players[1].quarks).toBe(0);
    expect(r.leader).toBe(0);
  });

  it('lets both players bridge simultaneously through the center (E)', () => {
    const b = getBoard(4);
    const r = scorePosition(
      b,
      position(
        b,
        ['*10', '*20', '*30', '*40', 'A10', 'A20', 'A30', 'A40'], // non-adjacent arms
        ['S10', 'S20', 'S30', 'S40', 'T10', 'T20', 'T30', 'T40'],
      ),
    );
    // Each pair of arms is one star thanks to the K5 bridge.
    expect(r.players[0].stars).toBe(1);
    expect(r.players[1].stars).toBe(1);
    expect(r.players[0].award).toBe(0);
    expect(r.players[1].award).toBe(0);
  });

  it('scores the empty board as all-contested', () => {
    const b = getBoard(6);
    const r = scorePosition(b, new Int8Array(b.n).fill(EMPTY));
    expect(r.players[0].total).toBe(0);
    expect(r.players[1].total).toBe(0);
    expect(r.contestedPeries).toBe(30);
    expect(r.leader).toBe(-1);
  });

  it('does not let territory bootstrap a lone perimeter stone', () => {
    const b = getBoard(6);
    const stones = new Int8Array(b.n).fill(EMPTY);
    const lone = b.idx(0, 6, 0);
    stones[lone] = 0;

    const got = scorePosition(b, stones);
    const want = referenceScore(b, stones);
    expect(got.players).toEqual([
      { peries: 0, quarks: 0, stars: 0, quarkPeri: 0, award: 0, total: 0 },
      { peries: 0, quarks: 0, stars: 0, quarkPeri: 0, award: 0, total: 0 },
    ]);
    expect(got.aliveStone[lone]).toBe(0);
    expect(got.nodeOwner[lone]).toBe(-1);
    expect(got.contestedPeries).toBe(b.periCount);
    expect(got.players).toEqual(want.players);
    expect(Array.from(got.aliveStone)).toEqual(want.aliveStone);
    expect(Array.from(got.nodeOwner)).toEqual(want.nodeOwner);
  });

});

describe('scorePosition: cross-validation and invariants', () => {
  it('matches the naive reference scorer on every supported board', () => {
    const rng = mulberry32(0xdecafbad);
    for (const rings of SUPPORTED_RINGS) {
      const b = getBoard(rings);
      for (const density of [0.02, 0.1, 0.25, 0.55, 0.8, 1]) {
        for (let trial = 0; trial < 30; trial++) {
          const stones = new Int8Array(b.n).fill(EMPTY);
          for (let u = 0; u < b.n; u++) {
            if (rng() < density) stones[u] = rng() < 0.5 ? 0 : 1;
          }
          const got = scorePosition(b, stones);
          const want = referenceScore(b, stones);
          expect(got.players).toEqual(want.players);
          expect(got.contestedPeries).toBe(want.contestedPeries);
          expect(Array.from(got.aliveStone)).toEqual(want.aliveStone);
          expect(Array.from(got.nodeOwner)).toEqual(want.nodeOwner);
        }
      }
    }
  });

  it('makes every sampled full supported board decided and decisive', () => {
    const rng = mulberry32(0x5717a5);
    for (const rings of SUPPORTED_RINGS) {
      const b = getBoard(rings);
      for (let trial = 0; trial < 120; trial++) {
        const stones = new Int8Array(b.n);
        // Random full fill, biased per-trial so both clumpy and mixed boards occur.
        const bias = 0.25 + 0.5 * rng();
        for (let u = 0; u < b.n; u++) stones[u] = rng() < bias ? 0 : 1;
        const r = scorePosition(b, stones);
        const sum = r.players[0].total + r.players[1].total;
        const margin = r.players[0].total - r.players[1].total;
        expect(r.contestedPeries).toBe(0);
        expect(r.players[0].peries + r.players[1].peries).toBe(b.periCount);
        expect(r.players[0].quarks + r.players[1].quarks).toBe(5);
        expect(r.players[0].quarkPeri + r.players[1].quarkPeri).toBe(1);
        expect(r.players[0].award + r.players[1].award).toBe(0);
        expect(sum).toBe(5 * rings + 1);
        expect(margin).not.toBe(0);
        expect(Math.abs(margin) % 2).toBe(1);
        expect(r.leader).toBe(margin > 0 ? 0 : 1);
      }
    }
  });

  it('agrees with the simplified and Schmittberger scoring margins', () => {
    const rng = mulberry32(0xace0fba5);
    const b = getBoard(6);
    for (let trial = 0; trial < 200; trial++) {
      const stones = new Int8Array(b.n).fill(EMPTY);
      for (let u = 0; u < b.n; u++) {
        if (rng() < 0.7) stones[u] = rng() < 0.5 ? 0 : 1;
      }
      const r = scorePosition(b, stones);
      const [p0, p1] = r.players;
      // Simplified: sum over stars of (owned peries - 4), plus the quark
      // peri. Identical margin to the conventional system.
      const simplified0 = p0.peries - 4 * p0.stars + p0.quarkPeri;
      const simplified1 = p1.peries - 4 * p1.stars + p1.quarkPeri;
      expect(simplified0 - simplified1).toBe(p0.total - p1.total);
      // Schmittberger alternative: own peries + quark peri minus opponent's,
      // plus (own award - opponent award). Algebraically this equals the
      // conventional score difference, so positive iff conventional winner.
      const schmitt0 =
        p0.peries + p0.quarkPeri - (p1.peries + p1.quarkPeri) + (p0.award - p1.award);
      expect(schmitt0).toBe(p0.total - p1.total);
    }
  });

  it('handles a sustained 1k-position full-board workload', () => {
    const rng = mulberry32(0xbe5eeed);
    const b = getBoard(10);
    const fills: Int8Array[] = [];
    for (let i = 0; i < 50; i++) {
      const stones = new Int8Array(b.n);
      for (let u = 0; u < b.n; u++) stones[u] = rng() < 0.5 ? 0 : 1;
      fills.push(stones);
    }
    let sink = 0;
    for (let i = 0; i < 1_000; i++) {
      sink += scorePosition(b, fills[i % fills.length]).players[0].total;
    }
    expect(Number.isFinite(sink)).toBe(true);
  });

  it('validates a binary terminal winner and all final invariants', () => {
    for (const rings of SUPPORTED_RINGS) {
      const board = getBoard(rings);
      const stones = Int8Array.from(
        { length: board.n },
        (_, node) => (node % 3 === 0 ? 0 : 1),
      );
      const terminal = validateTerminalWinner(board, stones);
      expect(terminal.winner).toBe(terminal.margin > 0 ? 0 : 1);
      expect(terminal.score.leader).toBe(terminal.winner);
      expect(terminal.score.contestedPeries).toBe(0);
      expect(
        terminal.score.players[0].total + terminal.score.players[1].total,
      ).toBe(5 * rings + 1);
      expect(Math.abs(terminal.margin) % 2).toBe(1);
    }
  });

  it('rejects nonfull, unsupported, and synthetic contested terminals', () => {
    const board = getBoard(4);
    expect(() =>
      validateTerminalWinner(board, new Int8Array(board.n).fill(EMPTY)),
    ).toThrow(/full/);

    const unsupported = { ...board, rings: 5 } as Board;
    expect(() =>
      validateTerminalWinner(unsupported, new Int8Array(board.n)),
    ).toThrow(/unsupported/);

    const disconnected = {
      ...board,
      adjOff: new Int32Array(board.n + 1),
      adj: new Int32Array(),
    };
    const mixed = Int8Array.from(
      { length: board.n },
      (_, node) => (node % 2) as 0 | 1,
    );
    expect(() => validateTerminalWinner(disconnected, mixed)).toThrow(
      /zero contested peries/,
    );
  });
});
