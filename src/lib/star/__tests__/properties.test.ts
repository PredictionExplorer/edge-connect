import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { getBoard, SUPPORTED_RINGS } from '../board';
import {
  applyAction,
  initialState,
  isLegalAction,
  replay,
  type GameAction,
} from '../game';
import { EMPTY, scorePosition, validateTerminalWinner } from '../scoring';

const configFor = (rings: number) => ({
  rings,
  mode: 'double' as const,
  pieRule: false,
  playerNames: ['Zero', 'One'] as [string, string],
});

describe('game properties', () => {
  it('incremental placement and replay agree for arbitrary legal traces', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...SUPPORTED_RINGS),
        fc.array(fc.nat(), { maxLength: 275 }),
        (rings, selectors) => {
          const config = configFor(rings);
          let state = initialState(config);
          const log: GameAction[] = [];
          for (const selector of selectors) {
            if (state.over) break;
            const empty = Array.from(
              { length: state.board.n },
              (_, node) => node,
            ).filter((node) => state.stones[node] === EMPTY);
            const action: GameAction = {
              type: 'place',
              node: empty[selector % empty.length],
            };
            expect(isLegalAction(state, action)).toBe(true);
            state = applyAction(state, action);
            log.push(action);

            expect(state.stonesPlaced).toBe(log.length);
            expect(state.over).toBe(log.length === state.board.n);

            const rebuilt = replay(config, log);
            expect(Array.from(rebuilt.stones)).toEqual(Array.from(state.stones));
            expect(rebuilt).toMatchObject({
              toMove: state.toMove,
              movesLeft: state.movesLeft,
              over: state.over,
              turnCount: state.turnCount,
            });
          }
        },
      ),
      { numRuns: 120 },
    );
  });

  it('the two placements of a completed Double *Star turn commute', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...SUPPORTED_RINGS),
        fc.uniqueArray(fc.integer({ min: 0, max: 49 }), {
          minLength: 3,
          maxLength: 3,
        }),
        (rings, [opening, first, second]) => {
          const config = configFor(rings);
          const start = applyAction(initialState(config), {
            type: 'place',
            node: opening,
          });
          const left = applyAction(
            applyAction(start, { type: 'place', node: first }),
            { type: 'place', node: second },
          );
          const right = applyAction(
            applyAction(start, { type: 'place', node: second }),
            { type: 'place', node: first },
          );
          expect(Array.from(left.stones)).toEqual(Array.from(right.stones));
          expect(left).toMatchObject({
            toMove: right.toMove,
            movesLeft: right.movesLeft,
            over: right.over,
          });
        },
      ),
      { numRuns: 100 },
    );
  });

  it('swapping colors swaps every generic scoring result', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...SUPPORTED_RINGS),
        fc.array(fc.integer({ min: -1, max: 1 }), {
          minLength: 275,
          maxLength: 275,
        }),
        (rings, raw) => {
          const board = getBoard(rings);
          const stones = Int8Array.from(raw.slice(0, board.n));
          const swapped = Int8Array.from(stones, (stone) =>
            stone === EMPTY ? EMPTY : 1 - stone,
          );
          const original = scorePosition(board, stones);
          const mirrored = scorePosition(board, swapped);
          expect(mirrored.players).toEqual([
            original.players[1],
            original.players[0],
          ]);
          expect(mirrored.contestedPeries).toBe(original.contestedPeries);
          expect(mirrored.leader).toBe(
            original.leader === -1 ? -1 : 1 - original.leader,
          );
          expect(Array.from(mirrored.nodeOwner)).toEqual(
            Array.from(original.nodeOwner, (owner) =>
              owner === -1 ? -1 : 1 - owner,
            ),
          );
          expect(Array.from(mirrored.aliveStone)).toEqual(
            Array.from(original.aliveStone),
          );
        },
      ),
      { numRuns: 80 },
    );
  });
});

describe('full-board winner properties', () => {
  for (const rings of SUPPORTED_RINGS) {
    it(`shrinks any no-draw invariant failure on ${rings} rings`, () => {
      const board = getBoard(rings);
      fc.assert(
        fc.property(
          fc.array(fc.integer({ min: 0, max: 1 }), {
            minLength: board.n,
            maxLength: board.n,
          }),
          (raw) => {
            const terminal = validateTerminalWinner(
              board,
              Int8Array.from(raw),
            );
            expect(terminal.winner === 0 || terminal.winner === 1).toBe(true);
            expect(terminal.score.contestedPeries).toBe(0);
            expect(
              terminal.score.players[0].total +
                terminal.score.players[1].total,
            ).toBe(5 * rings + 1);
            expect(terminal.margin).not.toBe(0);
            expect(Math.abs(terminal.margin) % 2).toBe(1);
          },
        ),
        { numRuns: 100 },
      );
    });
  }
});
