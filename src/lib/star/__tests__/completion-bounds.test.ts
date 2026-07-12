import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { getBoard, parseLabel, SUPPORTED_RINGS } from '../board';
import { scoreCompletionBounds } from '../completion-bounds';
import { replay, type GameAction } from '../game';
import { EMPTY, scorePosition, validateTerminalWinner } from '../scoring';

const configFor = (rings: number) => ({
  rings,
  mode: 'double' as const,
  pieRule: false,
  playerNames: ['Zero', 'One'] as [string, string],
});

describe('completion score bounds', () => {
  it('scores both monochrome completions of an empty board', () => {
    for (const rings of SUPPORTED_RINGS) {
      const board = getBoard(rings);
      const bounds = scoreCompletionBounds(
        board,
        new Int8Array(board.n).fill(EMPTY),
      );

      expect(bounds.emptyNodes).toBe(board.n);
      expect(bounds.scenarios[0].winner).toBe(0);
      expect(bounds.scenarios[1].winner).toBe(1);
      expect(bounds.scenarios[0].score.players[0].total).toBe(
        board.periCount - 1,
      );
      expect(bounds.scenarios[1].score.players[1].total).toBe(
        board.periCount - 1,
      );
      expect(bounds.guaranteedWinner).toBeNull();
    }
  });

  it('identifies a winner that survives the opponent-favored completion', () => {
    const board = getBoard(4);
    const zeroDominant = new Int8Array(board.n);
    zeroDominant[0] = EMPTY;
    const oneDominant = new Int8Array(board.n).fill(1);
    oneDominant[0] = EMPTY;

    expect(scoreCompletionBounds(board, zeroDominant).guaranteedWinner).toBe(0);
    expect(scoreCompletionBounds(board, oneDominant).guaranteedWinner).toBe(1);
  });

  it('rejects malformed positions', () => {
    const board = getBoard(4);
    expect(() =>
      scoreCompletionBounds(board, new Int8Array(board.n - 1)),
    ).toThrow(/stones length/);
    const invalid = new Int8Array(board.n).fill(EMPTY);
    invalid[7] = 3;
    expect(() => scoreCompletionBounds(board, invalid)).toThrow(
      /invalid stone 3 at node 7/,
    );
  });
});

describe('completion-bound properties', () => {
  it('terminal score is monotone under an opponent-to-player recoloring', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...SUPPORTED_RINGS),
        fc.array(fc.integer({ min: 0, max: 1 }), {
          minLength: 275,
          maxLength: 275,
        }),
        fc.nat(),
        (rings, raw, selector) => {
          const board = getBoard(rings);
          const node = selector % board.n;

          const towardZero = Int8Array.from(raw.slice(0, board.n));
          towardZero[node] = 1;
          const zeroBefore = validateTerminalWinner(board, towardZero);
          towardZero[node] = 0;
          const zeroAfter = validateTerminalWinner(board, towardZero);
          expect(zeroAfter.score.players[0].total).toBeGreaterThanOrEqual(
            zeroBefore.score.players[0].total,
          );
          expect(zeroAfter.margin).toBeGreaterThanOrEqual(zeroBefore.margin);

          const towardOne = Int8Array.from(raw.slice(0, board.n));
          towardOne[node] = 0;
          const oneBefore = validateTerminalWinner(board, towardOne);
          towardOne[node] = 1;
          const oneAfter = validateTerminalWinner(board, towardOne);
          expect(oneAfter.score.players[1].total).toBeGreaterThanOrEqual(
            oneBefore.score.players[1].total,
          );
          expect(oneAfter.margin).toBeLessThanOrEqual(oneBefore.margin);
        },
      ),
      { numRuns: 120 },
    );
  });

  it('bounds arbitrary full completions and is color symmetric', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...SUPPORTED_RINGS),
        fc.array(fc.integer({ min: 0, max: 1 }), {
          minLength: 275,
          maxLength: 275,
        }),
        fc.array(fc.boolean(), { minLength: 275, maxLength: 275 }),
        (rings, raw, emptyMask) => {
          const board = getBoard(rings);
          const full = Int8Array.from(raw.slice(0, board.n));
          const partial = full.slice();
          for (let node = 0; node < board.n; node++) {
            if (emptyMask[node]) partial[node] = EMPTY;
          }

          const bounds = scoreCompletionBounds(board, partial);
          const actual = validateTerminalWinner(board, full);
          const allZero = bounds.scenarios[0].score.players;
          const allOne = bounds.scenarios[1].score.players;

          expect(actual.score.players[0].total).toBeGreaterThanOrEqual(
            allOne[0].total,
          );
          expect(actual.score.players[0].total).toBeLessThanOrEqual(
            allZero[0].total,
          );
          expect(actual.score.players[1].total).toBeGreaterThanOrEqual(
            allZero[1].total,
          );
          expect(actual.score.players[1].total).toBeLessThanOrEqual(
            allOne[1].total,
          );
          if (bounds.guaranteedWinner !== null) {
            expect(actual.winner).toBe(bounds.guaranteedWinner);
          }

          const swapped = Int8Array.from(partial, (stone) =>
            stone === EMPTY ? EMPTY : 1 - stone,
          );
          const mirrored = scoreCompletionBounds(board, swapped);
          expect(mirrored.scenarios[0].score.players).toEqual([
            bounds.scenarios[1].score.players[1],
            bounds.scenarios[1].score.players[0],
          ]);
          expect(mirrored.scenarios[1].score.players).toEqual([
            bounds.scenarios[0].score.players[1],
            bounds.scenarios[0].score.players[0],
          ]);
          expect(mirrored.guaranteedWinner).toBe(
            bounds.guaranteedWinner === null
              ? null
              : 1 - bounds.guaranteedWinner,
          );
        },
      ),
      { numRuns: 80 },
    );
  });

  it('does not confuse terminal monotonicity with the live score', () => {
    const board = getBoard(4);
    const labels = ['S10', 'A41', 'A42', 'A40', 'S41', '*42', '*43'];
    const actions = labels.map(
      (label): GameAction => ({ type: 'place', node: parseLabel(board, label) }),
    );
    const before = replay(configFor(4), actions.slice(0, -1));
    const after = replay(configFor(4), actions);

    expect(before.toMove).toBe(1);
    expect(scorePosition(board, before.stones).players[1].total).toBe(19);
    expect(scorePosition(board, after.stones).players[1].total).toBe(17);
  });
});
