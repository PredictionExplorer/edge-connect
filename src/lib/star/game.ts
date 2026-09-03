/**
 * Game state for *Star, modeled as a pure reducer over an action log.
 *
 * Modes:
 *   - classic:  one stone per turn.
 *   - double:   Double *Star — two stones per turn, except the very first
 *     turn of the game, when the first player places a single stone.
 *
 * Handicap: the first player places `handicap` stones consecutively before
 * the second player's first turn (1..9; 1 is the standard game).
 *
 * Optional pie rule (handicap 1 only): immediately after the game's first
 * turn, the second player may swap — the opening stone changes to their color
 * and the first player moves next with a full turn.
 *
 * The game ends only when the board is full.
 *
 * Keeping the log (rather than mutable state) as the source of truth makes
 * undo/redo and localStorage persistence trivial and bug-resistant: state is
 * always rebuilt by replaying actions through the reducer.
 */

import { getBoard, type Board } from './board';
import { STAR_MAX_HANDICAP } from './rules';
import { EMPTY } from './scoring';

export type Mode = 'classic' | 'double';

export interface GameConfig {
  rings: number;
  mode: Mode;
  pieRule: boolean;
  /** Consecutive opening placements by the first player; defaults to 1. */
  handicap?: number;
  playerNames: [string, string];
}

export type GameAction =
  | { type: 'place'; node: number }
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
  over: boolean;
  /** True while the pie-rule swap is available (second player, first action). */
  canSwap: boolean;
  /** True if the pie swap was taken. */
  swapped: boolean;
  /** Node of the most recent placement, or -1. */
  lastMove: number;
  /** Nodes placed in the current (unfinished) turn. */
  currentTurnMoves: number[];
  /** Nodes placed in the most recently completed turn (the opponent's). */
  previousTurnMoves: number[];
  /** Nodes placed in the completed turn before that (the current player's). */
  ownPreviousTurnMoves: number[];
  /** Nodes placed during the opening phase (all `handicap` opening stones). */
  handicapStones: number[];
  turnCount: number;
}

/** Effective handicap of a config: an omitted value means the standard 1. */
export function configHandicap(config: Pick<GameConfig, 'handicap'>): number {
  return config.handicap ?? 1;
}

/** Placements per completed non-opening turn. */
export function modeTurnSize(mode: Mode): number {
  return mode === 'classic' ? 1 : 2;
}

function turnSize(config: GameConfig, turnIndex: number): number {
  if (turnIndex === 0) return configHandicap(config);
  return modeTurnSize(config.mode);
}

function validateConfig(config: GameConfig): void {
  const handicap = configHandicap(config);
  if (
    !Number.isInteger(handicap) ||
    handicap < 1 ||
    handicap > STAR_MAX_HANDICAP
  ) {
    throw new Error(`handicap must be an integer in 1..${STAR_MAX_HANDICAP}`);
  }
  if (config.pieRule && handicap !== 1) {
    throw new Error('handicap games cannot use the pie rule');
  }
}

export function initialState(config: GameConfig): GameState {
  const board = getBoard(config.rings);
  validateConfig(config);
  return {
    config,
    board,
    stones: new Int8Array(board.n).fill(EMPTY),
    stonesPlaced: 0,
    toMove: 0,
    movesLeft: turnSize(config, 0),
    midTurn: false,
    over: false,
    canSwap: false,
    swapped: false,
    lastMove: -1,
    currentTurnMoves: [],
    previousTurnMoves: [],
    ownPreviousTurnMoves: [],
    handicapStones: [],
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
  state.ownPreviousTurnMoves = state.previousTurnMoves;
  state.previousTurnMoves = state.currentTurnMoves;
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
    case 'swap':
      return state.canSwap;
    default:
      return false;
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
    previousTurnMoves: prev.previousTurnMoves.slice(),
    ownPreviousTurnMoves: prev.ownPreviousTurnMoves.slice(),
    handicapStones: prev.handicapStones.slice(),
  };

  switch (action.type) {
    case 'place': {
      state.stones[action.node] = state.toMove;
      state.stonesPlaced++;
      state.lastMove = action.node;
      state.currentTurnMoves.push(action.node);
      if (state.turnCount === 0) state.handicapStones.push(action.node);
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
    case 'swap': {
      // Recolor the single opening stone; the swap consumes this player's
      // turn, so the opener moves again next. The opening stone stays the most
      // recently completed turn, so the position reads exactly like the
      // unswapped position with colors exchanged.
      for (let u = 0; u < state.stones.length; u++) {
        if (state.stones[u] !== EMPTY) state.stones[u] = 1;
      }
      state.swapped = true;
      state.canSwap = false;
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
