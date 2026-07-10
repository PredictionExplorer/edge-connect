/**
 * Game state for *Star, modeled as a pure reducer over an action log.
 *
 * Modes:
 *   - classic:  one stone per turn.
 *   - double:   Double *Star — two stones per turn, except the very first
 *     turn of the game, when the first player places a single stone.
 *
 * The game ends when the board is full, or after two consecutive pass
 *  actions (the players agree the score is decided).
 *
 * Optional pie rule: immediately after the game's first turn, the second
 * player may swap — the opening stone changes to their color and the first
 * player moves next.
 *
 * Keeping the log (rather than mutable state) as the source of truth makes
 * undo/redo and localStorage persistence trivial and bug-resistant: state is
 * always rebuilt by replaying actions through the reducer.
 */

import { getBoard, type Board } from './board';
import { EMPTY } from './scoring';

export type Mode = 'classic' | 'double';

export interface GameConfig {
  rings: number;
  mode: Mode;
  pieRule: boolean;
  playerNames: [string, string];
}

export type GameAction =
  | { type: 'place'; node: number }
  | { type: 'pass' }
  | { type: 'swap' };

export interface GameState {
  config: GameConfig;
  board: Board;
  /** EMPTY (-1), 0, or 1 per node. */
  stones: Int8Array;
  stonesPlaced: number;
  /** Player to act, 0 or 1. */
  toMove: 0 | 1;
  /** Stones the current player may still place this turn. */
  movesLeft: number;
  /** Whether the current player has already placed a stone this turn. */
  midTurn: boolean;
  /** Consecutive pass actions. */
  passStreak: number;
  over: boolean;
  /** True while the pie-rule swap is available (second player, first action). */
  canSwap: boolean;
  /** True if the pie swap was taken. */
  swapped: boolean;
  /** Node of the most recent placement, or -1. */
  lastMove: number;
  /** Nodes placed in the current (unfinished) turn. */
  currentTurnMoves: number[];
  turnCount: number;
}

function turnSize(config: GameConfig, turnIndex: number): number {
  if (config.mode === 'classic') return 1;
  return turnIndex === 0 ? 1 : 2;
}

export function initialState(config: GameConfig): GameState {
  const board = getBoard(config.rings);
  return {
    config,
    board,
    stones: new Int8Array(board.n).fill(EMPTY),
    stonesPlaced: 0,
    toMove: 0,
    movesLeft: turnSize(config, 0),
    midTurn: false,
    passStreak: 0,
    over: false,
    canSwap: false,
    swapped: false,
    lastMove: -1,
    currentTurnMoves: [],
    turnCount: 0,
  };
}

function boardFull(state: GameState): boolean {
  return state.stonesPlaced === state.board.n;
}

function endTurn(state: GameState): void {
  state.toMove = (1 - state.toMove) as 0 | 1;
  state.turnCount++;
  state.movesLeft = turnSize(state.config, state.turnCount);
  state.midTurn = false;
  state.currentTurnMoves = [];
}

export function isLegalAction(state: GameState, action: GameAction): boolean {
  if (state.over) return false;
  switch (action.type) {
    case 'place':
      return (
        action.node >= 0 &&
        action.node < state.board.n &&
        state.stones[action.node] === EMPTY
      );
    case 'pass':
      return true;
    case 'swap':
      return state.canSwap;
  }
}

/** Apply an action, returning a new state. Throws on illegal actions. */
export function applyAction(prev: GameState, action: GameAction): GameState {
  if (!isLegalAction(prev, action)) {
    throw new Error(`illegal action ${JSON.stringify(action)}`);
  }
  const state: GameState = {
    ...prev,
    stones: prev.stones.slice(),
    currentTurnMoves: prev.currentTurnMoves.slice(),
  };

  switch (action.type) {
    case 'place': {
      state.stones[action.node] = state.toMove;
      state.stonesPlaced++;
      state.lastMove = action.node;
      state.currentTurnMoves.push(action.node);
      state.passStreak = 0;
      state.movesLeft--;
      state.midTurn = state.movesLeft > 0;
      if (boardFull(state)) {
        state.over = true;
        state.canSwap = false;
        return state;
      }
      if (state.movesLeft === 0) {
        const wasFirstTurn = state.turnCount === 0;
        endTurn(state);
        state.canSwap = state.config.pieRule && wasFirstTurn && !state.swapped;
      } else {
        state.canSwap = false;
      }
      return state;
    }
    case 'pass': {
      state.passStreak++;
      state.canSwap = false;
      if (state.passStreak >= 2) {
        state.over = true;
        return state;
      }
      endTurn(state);
      return state;
    }
    case 'swap': {
      // Recolor the single opening stone; the swap consumes this player's
      // turn, so the opener moves again next.
      for (let u = 0; u < state.stones.length; u++) {
        if (state.stones[u] !== EMPTY) state.stones[u] = 1;
      }
      state.swapped = true;
      state.canSwap = false;
      state.passStreak = 0;
      state.toMove = 0;
      state.turnCount++;
      state.movesLeft = turnSize(state.config, state.turnCount);
      state.midTurn = false;
      state.currentTurnMoves = [];
      return state;
    }
  }
}

/** Rebuild state by replaying a log from scratch. */
export function replay(config: GameConfig, log: GameAction[]): GameState {
  let state = initialState(config);
  for (const action of log) state = applyAction(state, action);
  return state;
}
