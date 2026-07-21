/**
 * Derived move-history timeline for presentation.
 *
 * Replays an action log once and annotates every action with the player who
 * made it, the turn it belongs to, and its display label, then groups actions
 * into turns for the move list, review scrubbing, and board highlights.
 * Everything here is derived data — the raw log stays the single source of
 * truth.
 */

import {
  applyAction,
  initialState,
  type GameAction,
  type GameConfig,
} from './game';

export interface TimelineEntry {
  /** Index of the action in the log; the position after it is ply `index + 1`. */
  index: number;
  action: GameAction;
  /** Player who performed the action. */
  player: 0 | 1;
  /** 0-based turn the action belongs to. */
  turnNumber: number;
  /** 0-based position of the action within its turn. */
  indexInTurn: number;
  /** Board node affected: the placed node, or the recolored stone for a swap. */
  node: number;
  /** Display label: the node label (e.g. "T43"), or "Swap" for the pie swap. */
  label: string;
  /** True when this action finished its turn and play passed on. */
  endsTurn: boolean;
}

export interface TimelineTurn {
  /** 0-based turn number. */
  turnNumber: number;
  player: 0 | 1;
  /** Number of placements this turn allows. */
  capacity: number;
  entries: TimelineEntry[];
  /** True for the pie-rule swap pseudo-turn. */
  swap: boolean;
}

export interface Timeline {
  entries: TimelineEntry[];
  turns: TimelineTurn[];
}

/** Annotate and group a legal action log. Throws on illegal logs, like replay. */
export function buildTimeline(config: GameConfig, log: GameAction[]): Timeline {
  let state = initialState(config);
  const entries: TimelineEntry[] = [];
  const turns: TimelineTurn[] = [];

  for (let index = 0; index < log.length; index++) {
    const action = log[index];
    const player = state.toMove;
    const turnNumber = state.turnCount;
    const isPlace = action.type === 'place';
    const indexInTurn = isPlace ? state.currentTurnMoves.length : 0;
    const capacity = isPlace
      ? state.currentTurnMoves.length + state.movesLeft
      : 1;
    const node = isPlace ? action.node : state.lastMove;
    const label = isPlace ? state.board.labels[action.node] : 'Swap';

    state = applyAction(state, action);

    const entry: TimelineEntry = {
      index,
      action,
      player,
      turnNumber,
      indexInTurn,
      node,
      label,
      endsTurn: state.turnCount > turnNumber,
    };
    entries.push(entry);

    const turn = turns[turns.length - 1];
    if (turn && turn.turnNumber === turnNumber) {
      turn.entries.push(entry);
    } else {
      turns.push({
        turnNumber,
        player,
        capacity,
        entries: [entry],
        swap: !isPlace,
      });
    }
  }

  return { entries, turns };
}

/**
 * Nodes of the most recent turn that fully ended at or before the position
 * reached after `ply` actions, in placement order. In-progress placements are
 * reported separately by `GameState.currentTurnMoves`; together they cover
 * "everything that changed since your last turn". A completed swap turn
 * reports the recolored opening stone.
 */
export function lastCompletedTurnMoves(
  timeline: Timeline,
  ply: number,
): number[] {
  const { entries } = timeline;
  const upto = Math.min(ply, entries.length);
  for (let i = upto - 1; i >= 0; i--) {
    if (!entries[i].endsTurn) continue;
    const { turnNumber } = entries[i];
    const nodes: number[] = [];
    for (let j = i; j >= 0 && entries[j].turnNumber === turnNumber; j--) {
      nodes.unshift(entries[j].node);
    }
    return nodes;
  }
  return [];
}
