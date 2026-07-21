import { describe, expect, it } from 'vitest';
import { getBoard } from '../board';
import type { GameAction, GameConfig } from '../game';
import { buildTimeline, lastCompletedTurnMoves } from '../timeline';

const classic: GameConfig = {
  rings: 4,
  mode: 'classic',
  pieRule: false,
  playerNames: ['Ada', 'Grace'],
};
const double: GameConfig = { ...classic, mode: 'double' };

function places(...nodes: number[]): GameAction[] {
  return nodes.map((node) => ({ type: 'place', node }));
}

describe('buildTimeline', () => {
  it('annotates classic actions with alternating single-stone turns', () => {
    const timeline = buildTimeline(classic, places(0, 1, 2));
    expect(timeline.entries).toHaveLength(3);
    expect(timeline.entries.map((entry) => entry.player)).toEqual([0, 1, 0]);
    expect(timeline.entries.map((entry) => entry.turnNumber)).toEqual([0, 1, 2]);
    expect(timeline.entries.every((entry) => entry.endsTurn)).toBe(true);
    expect(timeline.turns.map((turn) => turn.capacity)).toEqual([1, 1, 1]);
    expect(timeline.entries[0].label).toBe(getBoard(4).labels[0]);
  });

  it('groups Double *Star pairs after the single opening stone', () => {
    const timeline = buildTimeline(double, places(0, 1, 2, 3));
    expect(timeline.turns).toHaveLength(3);
    expect(timeline.turns[0]).toMatchObject({
      turnNumber: 0,
      player: 0,
      capacity: 1,
    });
    expect(timeline.turns[1]).toMatchObject({
      turnNumber: 1,
      player: 1,
      capacity: 2,
    });
    expect(timeline.turns[1].entries.map((entry) => entry.node)).toEqual([1, 2]);
    expect(timeline.turns[1].entries.map((entry) => entry.indexInTurn)).toEqual([
      0, 1,
    ]);
    expect(timeline.turns[1].entries.map((entry) => entry.endsTurn)).toEqual([
      false,
      true,
    ]);
    // Trailing in-progress turn: first stone of an unfinished pair.
    expect(timeline.turns[2]).toMatchObject({ player: 0, capacity: 2 });
    expect(timeline.turns[2].entries[0].endsTurn).toBe(false);
  });

  it('reports the swap as its own turn on the recolored opening stone', () => {
    const config: GameConfig = { ...double, pieRule: true };
    const log: GameAction[] = [
      { type: 'place', node: 7 },
      { type: 'swap' },
      { type: 'place', node: 8 },
    ];
    const timeline = buildTimeline(config, log);
    expect(timeline.turns).toHaveLength(3);
    expect(timeline.turns[1]).toMatchObject({
      turnNumber: 1,
      player: 1,
      swap: true,
    });
    expect(timeline.turns[1].entries[0]).toMatchObject({
      node: 7,
      label: 'Swap',
      endsTurn: true,
    });
    // The opener moves again after the swap.
    expect(timeline.turns[2].player).toBe(0);
  });
});

describe('lastCompletedTurnMoves', () => {
  it('returns the last finished turn and skips in-progress placements', () => {
    const timeline = buildTimeline(double, places(0, 1, 2, 3));
    expect(lastCompletedTurnMoves(timeline, 0)).toEqual([]);
    expect(lastCompletedTurnMoves(timeline, 1)).toEqual([0]);
    // Mid-pair: the opponent's opening stone is still the last finished turn.
    expect(lastCompletedTurnMoves(timeline, 2)).toEqual([0]);
    expect(lastCompletedTurnMoves(timeline, 3)).toEqual([1, 2]);
    // Player 0 started a new pair; the finished pair keeps the highlight.
    expect(lastCompletedTurnMoves(timeline, 4)).toEqual([1, 2]);
    // A ply beyond the log clamps to the final position.
    expect(lastCompletedTurnMoves(timeline, 99)).toEqual([1, 2]);
  });

  it('highlights the recolored opening stone after a pie swap', () => {
    const config: GameConfig = { ...double, pieRule: true };
    const timeline = buildTimeline(config, [
      { type: 'place', node: 7 },
      { type: 'swap' },
    ]);
    expect(lastCompletedTurnMoves(timeline, 2)).toEqual([7]);
  });
});
