import type { Board } from './board';
import {
  EMPTY,
  validateTerminalWinner,
  type ScoreResult,
} from './scoring';

export interface CompletionScenario {
  /** Player assigned every currently empty node in this synthetic full board. */
  fillPlayer: 0 | 1;
  score: ScoreResult & { leader: 0 | 1 };
  winner: 0 | 1;
  /** Player-zero total minus player-one total. */
  margin: number;
}

export interface CompletionBounds {
  /** Indexed by the player who receives every currently empty node. */
  scenarios: [CompletionScenario, CompletionScenario];
  emptyNodes: number;
  /** Winner forced across every full completion, or null while still unsettled. */
  guaranteedWinner: 0 | 1 | null;
}

function fillEmptyNodes(
  board: Board,
  stones: ArrayLike<number>,
  fillPlayer: 0 | 1,
): Int8Array {
  if (stones.length !== board.n) {
    throw new Error(`stones length must be ${board.n}, got ${stones.length}`);
  }

  return Int8Array.from({ length: board.n }, (_, node) => {
    const stone = stones[node];
    if (stone === EMPTY) return fillPlayer;
    if (stone === 0 || stone === 1) return stone;
    throw new Error(`invalid stone ${String(stone)} at node ${node}`);
  });
}

/**
 * Score the two extremal full completions of a live position.
 *
 * These synthetic boards need not be reachable turn histories; they are
 * mathematical bounds on every reachable full completion. For a full board,
 * changing one opponent node to a player's color cannot reduce that player's
 * terminal total:
 *
 *   total0 - quarkPeri0
 *     = sum_0 max(componentPeries - 2, 0)
 *     + sum_1 min(componentPeries, 2)
 *
 * A recoloring only merges player-zero components and splits a player-one
 * component. The first summand is superadditive, the second is subadditive,
 * and owned quarks cannot move away from the player receiving the node.
 * Therefore repeated recolorings make the all-zero fill player zero's upper
 * bound and the all-one fill player one's upper bound. This terminal theorem
 * is intentionally distinct from live scoring, which is not monotone.
 */
export function scoreCompletionBounds(
  board: Board,
  stones: ArrayLike<number>,
): CompletionBounds {
  let emptyNodes = 0;
  for (let node = 0; node < board.n; node++) {
    if (stones[node] === EMPTY) emptyNodes++;
  }

  const scenarios = ([0, 1] as const).map((fillPlayer) => {
    const terminal = validateTerminalWinner(
      board,
      fillEmptyNodes(board, stones, fillPlayer),
    );
    return {
      fillPlayer,
      score: terminal.score,
      winner: terminal.winner,
      margin: terminal.margin,
    };
  }) as [CompletionScenario, CompletionScenario];

  const guaranteedWinner =
    scenarios[0].winner === 1
      ? 1
      : scenarios[1].winner === 0
        ? 0
        : null;

  return { scenarios, emptyNodes, guaranteedWinner };
}
