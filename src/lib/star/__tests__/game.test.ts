import { describe, expect, it } from 'vitest';
import { EMPTY } from '../scoring';
import {
  applyAction,
  initialState,
  isLegalAction,
  replay,
  type GameAction,
  type GameConfig,
  type GameState,
} from '../game';

const base: GameConfig = {
  rings: 4,
  mode: 'classic',
  pieRule: false,
  playerNames: ['Aurora', 'Vega'],
};

function play(state: GameState, ...nodes: number[]): GameState {
  for (const node of nodes) state = applyAction(state, { type: 'place', node });
  return state;
}

function fill(state: GameState, count = state.board.n): GameState {
  for (let node = 0; node < count; node++) state = play(state, node);
  return state;
}

describe('placement-only game protocol', () => {
  it('alternates classic placements and rejects illegal nodes', () => {
    let state = initialState(base);
    expect(state.toMove).toBe(0);
    expect(state.movesLeft).toBe(1);
    state = play(state, 0, 1);
    expect(Array.from(state.stones.slice(0, 2))).toEqual([0, 1]);
    expect(state.toMove).toBe(0);
    expect(isLegalAction(state, { type: 'place', node: 0 })).toBe(false);
    expect(isLegalAction(state, { type: 'place', node: -1 })).toBe(false);
    expect(isLegalAction(state, { type: 'place', node: state.board.n })).toBe(
      false,
    );
    expect(() => applyAction(state, { type: 'place', node: 0 })).toThrow(
      /illegal action/,
    );
  });

  it('has no pass action or state and rejects legacy input at runtime', () => {
    const state = initialState(base);
    const legacy = { type: 'pass' } as unknown as GameAction;
    expect('passStreak' in state).toBe(false);
    expect(isLegalAction(state, legacy)).toBe(false);
    expect(() => applyAction(state, legacy)).toThrow(/illegal action/);
  });

  it('is terminal exactly when the board becomes full', () => {
    let state = initialState(base);
    state = fill(state, state.board.n - 1);
    expect(state.over).toBe(false);
    expect(state.stonesPlaced).toBe(state.board.n - 1);
    state = play(state, state.board.n - 1);
    expect(state.over).toBe(true);
    expect(state.stonesPlaced).toBe(state.board.n);
    expect(isLegalAction(state, { type: 'place', node: 0 })).toBe(false);
  });

  it.each([3, 5, 7, 9, 11, 12, 4.5])(
    'rejects unsupported ring count %s',
    (rings) => {
      expect(() => initialState({ ...base, rings })).toThrow(/one of 4, 6, 8, 10/);
    },
  );
});

describe('Double *Star protocol', () => {
  const double: GameConfig = { ...base, mode: 'double' };

  it('gives the opener one stone and later turns two', () => {
    let state = initialState(double);
    expect(state.movesLeft).toBe(1);
    state = play(state, 0);
    expect(state.toMove).toBe(1);
    expect(state.movesLeft).toBe(2);
    state = play(state, 1);
    expect(state.toMove).toBe(1);
    expect(state.movesLeft).toBe(1);
    expect(state.midTurn).toBe(true);
    state = play(state, 2);
    expect(state.toMove).toBe(0);
    expect(state.movesLeft).toBe(2);
    expect(Array.from(state.stones.slice(0, 3))).toEqual([0, 1, 1]);
  });

  it('preserves the final partial-turn residual on 4 rings', () => {
    const state = fill(initialState(double));
    expect(state.over).toBe(true);
    expect(state.movesLeft).toBe(1);
    expect(state.midTurn).toBe(true);
    expect(state.currentTurnMoves).toEqual([state.board.n - 1]);
  });

  it('preserves zero residual moves after a final pair on 6 rings', () => {
    const state = fill(initialState({ ...double, rings: 6 }));
    expect(state.over).toBe(true);
    expect(state.movesLeft).toBe(0);
    expect(state.midTurn).toBe(false);
    expect(state.currentTurnMoves).toEqual([
      state.board.n - 2,
      state.board.n - 1,
    ]);
  });
});

describe('web-only pie rule', () => {
  it('lets the second player steal only the opening stone', () => {
    let state = initialState({ ...base, pieRule: true });
    state = play(state, 7);
    expect(state.canSwap).toBe(true);
    state = applyAction(state, { type: 'swap' });
    expect(state.stones[7]).toBe(1);
    expect(state.toMove).toBe(0);
    expect(state.swapped).toBe(true);
    expect(state.canSwap).toBe(false);
    state = play(state, 8);
    expect(isLegalAction(state, { type: 'swap' })).toBe(false);
  });

  it('expires when the second player places', () => {
    let state = initialState({ ...base, pieRule: true });
    state = play(state, 7, 8);
    expect(state.canSwap).toBe(false);
  });

  it('preserves Double *Star turn size after a swap', () => {
    const config: GameConfig = { ...base, mode: 'double', pieRule: true };
    let state = play(initialState(config), 7);
    state = applyAction(state, { type: 'swap' });
    expect(state.toMove).toBe(0);
    expect(state.movesLeft).toBe(2);
    state = play(state, 8, 9);
    expect(state.toMove).toBe(1);
  });
});

describe('replay', () => {
  it('rebuilds placement and swap logs exactly', () => {
    const config: GameConfig = { ...base, mode: 'double', pieRule: true };
    const log: GameAction[] = [
      { type: 'place', node: 3 },
      { type: 'swap' },
      { type: 'place', node: 10 },
      { type: 'place', node: 11 },
      { type: 'place', node: 20 },
      { type: 'place', node: 21 },
      { type: 'place', node: 30 },
    ];
    let stepped = initialState(config);
    for (const action of log) stepped = applyAction(stepped, action);
    const rebuilt = replay(config, log);
    expect(Array.from(rebuilt.stones)).toEqual(Array.from(stepped.stones));
    expect(rebuilt).toMatchObject({
      toMove: stepped.toMove,
      movesLeft: stepped.movesLeft,
      over: stepped.over,
      turnCount: stepped.turnCount,
    });

    const undone = replay(config, log.slice(0, -1));
    expect(undone.stones[30]).toBe(EMPTY);
    expect(undone.movesLeft).toBe(2);
  });
});
