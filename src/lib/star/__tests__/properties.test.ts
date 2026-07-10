import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { applyAction, initialState, isLegalAction, replay, type GameAction } from '../game';
import { EMPTY, scorePosition } from '../scoring';

describe('game properties', () => {
  it('incremental application and replay agree for arbitrary legal traces', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 3, max: 12 }),
        fc.array(fc.nat(), { maxLength: 250 }),
        (rings, selectors) => {
          const config = {
            rings,
            mode: 'double' as const,
            pieRule: false,
            playerNames: ['Zero', 'One'] as [string, string],
          };
          let state = initialState(config);
          const log: GameAction[] = [];
          for (const selector of selectors) {
            if (state.over) break;
            const empty = Array.from(
              { length: state.board.n },
              (_, node) => node,
            ).filter((node) => state.stones[node] === EMPTY);
            const action: GameAction =
              selector % 11 === 0 || empty.length === 0
                ? { type: 'pass' }
                : { type: 'place', node: empty[selector % empty.length] };
            expect(isLegalAction(state, action)).toBe(true);
            state = applyAction(state, action);
            log.push(action);

            const occupied = Array.from(state.stones).filter(
              (stone) => stone !== EMPTY,
            ).length;
            expect(state.stonesPlaced).toBe(occupied);
            expect(state.over).toBe(
              occupied === state.board.n || state.passStreak === 2,
            );

            const rebuilt = replay(config, log);
            expect(Array.from(rebuilt.stones)).toEqual(Array.from(state.stones));
            expect({
              toMove: rebuilt.toMove,
              movesLeft: rebuilt.movesLeft,
              passStreak: rebuilt.passStreak,
              over: rebuilt.over,
              turnCount: rebuilt.turnCount,
            }).toEqual({
              toMove: state.toMove,
              movesLeft: state.movesLeft,
              passStreak: state.passStreak,
              over: state.over,
              turnCount: state.turnCount,
            });
          }
        },
      ),
      { numRuns: 150 },
    );
  });

  it('the two placements of a completed Double Star turn commute semantically', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 3, max: 12 }),
        fc.uniqueArray(fc.integer({ min: 0, max: 29 }), {
          minLength: 3,
          maxLength: 3,
        }),
        (rings, [opening, first, second]) => {
          const config = {
            rings,
            mode: 'double' as const,
            pieRule: false,
            playerNames: ['Zero', 'One'] as [string, string],
          };
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
          expect({
            toMove: left.toMove,
            movesLeft: left.movesLeft,
            passStreak: left.passStreak,
            over: left.over,
          }).toEqual({
            toMove: right.toMove,
            movesLeft: right.movesLeft,
            passStreak: right.passStreak,
            over: right.over,
          });
        },
      ),
      { numRuns: 100 },
    );
  });

  it('swapping colors swaps every scoring result', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 3, max: 7 }),
        fc.array(fc.integer({ min: -1, max: 1 }), {
          minLength: 390,
          maxLength: 390,
        }),
        (rings, raw) => {
          const state = initialState({
            rings,
            mode: 'double',
            pieRule: false,
            playerNames: ['Zero', 'One'],
          });
          const stones = Int8Array.from(
            raw.slice(0, state.board.n),
            (stone) => stone,
          );
          const swapped = Int8Array.from(stones, (stone) =>
            stone === EMPTY ? EMPTY : 1 - stone,
          );
          const original = scorePosition(state.board, stones);
          const mirrored = scorePosition(state.board, swapped);
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
      { numRuns: 100 },
    );
  });
});

