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

describe('classic protocol', () => {
  it('alternates single placements', () => {
    let s = initialState(base);
    expect(s.toMove).toBe(0);
    expect(s.movesLeft).toBe(1);
    s = play(s, 0);
    expect(s.stones[0]).toBe(0);
    expect(s.toMove).toBe(1);
    s = play(s, 1);
    expect(s.stones[1]).toBe(1);
    expect(s.toMove).toBe(0);
  });

  it('rejects occupied nodes and out-of-range nodes', () => {
    let s = initialState(base);
    s = play(s, 5);
    expect(isLegalAction(s, { type: 'place', node: 5 })).toBe(false);
    expect(() => applyAction(s, { type: 'place', node: 5 })).toThrow();
    expect(isLegalAction(s, { type: 'place', node: -1 })).toBe(false);
    expect(isLegalAction(s, { type: 'place', node: s.board.n })).toBe(false);
  });

  it('ends after two consecutive passes and rejects further actions', () => {
    let s = initialState(base);
    s = play(s, 0, 1);
    s = applyAction(s, { type: 'pass' });
    expect(s.over).toBe(false);
    // A placement resets the streak.
    s = play(s, 2);
    s = applyAction(s, { type: 'pass' });
    s = applyAction(s, { type: 'pass' });
    expect(s.over).toBe(true);
    expect(isLegalAction(s, { type: 'place', node: 3 })).toBe(false);
    expect(isLegalAction(s, { type: 'pass' })).toBe(false);
  });

  it('ends when the board fills up', () => {
    let s = initialState({ ...base, rings: 3 });
    for (let u = 0; u < s.board.n; u++) s = play(s, u);
    expect(s.over).toBe(true);
    expect(s.stonesPlaced).toBe(s.board.n);
  });
});

describe('Double *Star protocol', () => {
  const dbl: GameConfig = { ...base, mode: 'double' };

  it('gives the first player one stone, then two per turn', () => {
    let s = initialState(dbl);
    expect(s.movesLeft).toBe(1);
    s = play(s, 0); // P0 single opening stone
    expect(s.toMove).toBe(1);
    expect(s.movesLeft).toBe(2);
    s = play(s, 1);
    expect(s.toMove).toBe(1); // still P1, mid-turn
    expect(s.movesLeft).toBe(1);
    expect(s.midTurn).toBe(true);
    s = play(s, 2);
    expect(s.toMove).toBe(0);
    expect(s.movesLeft).toBe(2);
    const colors = [s.stones[0], s.stones[1], s.stones[2]];
    expect(colors).toEqual([0, 1, 1]);
  });

  it('ends mid-turn when the last node is filled', () => {
    let s = initialState({ ...dbl, rings: 3 });
    const n = s.board.n; // 30 nodes: 1 + 14*2 + 1 leaves one dangling stone
    for (let u = 0; u < n; u++) s = play(s, u);
    expect(s.over).toBe(true);
    expect(s.stonesPlaced).toBe(n);
    expect(s.movesLeft).toBe(1);
    expect(s.midTurn).toBe(true);
    expect(s.passStreak).toBe(0);
  });

  it('preserves zero residual moves when a full board ends on a pair', () => {
    let s = initialState({ ...dbl, rings: 5 });
    for (let u = 0; u < s.board.n; u++) s = play(s, u);
    expect(s.over).toBe(true);
    expect(s.stonesPlaced).toBe(s.board.n);
    expect(s.movesLeft).toBe(0);
    expect(s.midTurn).toBe(false);
    expect(s.passStreak).toBe(0);
  });

  it('a pass forfeits the remainder of the turn', () => {
    let s = initialState(dbl);
    s = play(s, 0);
    s = play(s, 1); // P1 places one of two...
    s = applyAction(s, { type: 'pass' }); // ...then passes the second
    expect(s.toMove).toBe(0);
    expect(s.movesLeft).toBe(2);
    expect(s.over).toBe(false);
  });

  it('retains passStreak 2 and movesLeft 2 after a terminal double pass', () => {
    let s = initialState(dbl);
    s = applyAction(s, { type: 'pass' });
    expect(s.passStreak).toBe(1);
    expect(s.movesLeft).toBe(2);
    s = applyAction(s, { type: 'pass' });
    expect(s.over).toBe(true);
    expect(s.passStreak).toBe(2);
    expect(s.movesLeft).toBe(2);
    expect(s.toMove).toBe(1);
  });
});

describe('pie rule', () => {
  it('lets the second player steal the opening stone', () => {
    const cfg: GameConfig = { ...base, pieRule: true };
    let s = initialState(cfg);
    s = play(s, 7);
    expect(s.canSwap).toBe(true);
    s = applyAction(s, { type: 'swap' });
    expect(s.stones[7]).toBe(1); // recolored to player 1
    expect(s.toMove).toBe(0);
    expect(s.swapped).toBe(true);
    expect(s.canSwap).toBe(false);
    // No second swap ever.
    s = play(s, 8);
    expect(s.canSwap).toBe(false);
  });

  it('expires if the second player places instead', () => {
    const cfg: GameConfig = { ...base, pieRule: true };
    let s = initialState(cfg);
    s = play(s, 7);
    expect(s.canSwap).toBe(true);
    s = play(s, 8);
    expect(s.canSwap).toBe(false);
    expect(isLegalAction(s, { type: 'swap' })).toBe(false);
  });

  it('is unavailable when disabled', () => {
    let s = initialState(base);
    s = play(s, 7);
    expect(s.canSwap).toBe(false);
  });

  it('in double mode, swap consumes the turn and the opener gets two stones', () => {
    const cfg: GameConfig = { ...base, mode: 'double', pieRule: true };
    let s = initialState(cfg);
    s = play(s, 7); // single opening stone
    expect(s.canSwap).toBe(true);
    s = applyAction(s, { type: 'swap' });
    expect(s.toMove).toBe(0);
    expect(s.movesLeft).toBe(2);
    s = play(s, 8, 9);
    expect(s.toMove).toBe(1);
    expect(s.movesLeft).toBe(2);
  });
});

describe('replay (undo/redo backbone)', () => {
  it('rebuilds identical state from the action log', () => {
    const cfg: GameConfig = { ...base, mode: 'double', pieRule: true };
    const log: GameAction[] = [
      { type: 'place', node: 3 },
      { type: 'swap' },
      { type: 'place', node: 10 },
      { type: 'place', node: 11 },
      { type: 'place', node: 20 },
      { type: 'pass' },
      { type: 'place', node: 30 },
    ];
    let stepped = initialState(cfg);
    for (const a of log) stepped = applyAction(stepped, a);
    const replayed = replay(cfg, log);
    expect(Array.from(replayed.stones)).toEqual(Array.from(stepped.stones));
    expect(replayed.toMove).toBe(stepped.toMove);
    expect(replayed.movesLeft).toBe(stepped.movesLeft);
    expect(replayed.over).toBe(stepped.over);
    expect(replayed.turnCount).toBe(stepped.turnCount);
    // Undo = replaying a truncated log. The final stone was the first of
    // player 0's two-stone turn, so undoing it stays on player 0 but restores
    // both moves.
    const undone = replay(cfg, log.slice(0, -1));
    expect(undone.stones[30]).toBe(EMPTY);
    expect(stepped.toMove).toBe(0);
    expect(stepped.movesLeft).toBe(1);
    expect(undone.toMove).toBe(0);
    expect(undone.movesLeft).toBe(2);
    expect(undone.midTurn).toBe(false);
  });
});
