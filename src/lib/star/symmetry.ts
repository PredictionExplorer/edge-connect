/**
 * Exact D5 symmetries for the pentagonal *Star board.
 *
 * On ring x, flatten (sector s, position y) to the cyclic coordinate
 * t = s*x + y in 0..5*x-1. The ten dihedral maps are then:
 *
 *   rotation k:   t -> t + k*x
 *   reflection k: t -> k*x - t
 *
 * modulo 5*x. This uses only the board's discrete sector/ring/position
 * coordinates; layout coordinates are deliberately irrelevant.
 */

import type { Board } from './board';
import type { GameAction } from './game';

export type FifthTurn = 0 | 1 | 2 | 3 | 4;
export type D5Kind = 'rotation' | 'reflection';

export interface D5Symmetry {
  /** Stable wire-format identifier: r0..r4 or f0..f4. */
  readonly id: string;
  readonly kind: D5Kind;
  readonly turns: FifthTurn;
}

export interface D5Maps {
  /** forward[u] is the transformed node id. */
  readonly forward: Int32Array;
  /** inverse[v] is the original node id. */
  readonly inverse: Int32Array;
}

function makeSymmetry(kind: D5Kind, turns: FifthTurn): D5Symmetry {
  return Object.freeze({
    id: `${kind === 'rotation' ? 'r' : 'f'}${turns}`,
    kind,
    turns,
  });
}

export const D5_ROTATIONS: readonly D5Symmetry[] = Object.freeze(
  ([0, 1, 2, 3, 4] as const).map((turns) => makeSymmetry('rotation', turns)),
);

export const D5_REFLECTIONS: readonly D5Symmetry[] = Object.freeze(
  ([0, 1, 2, 3, 4] as const).map((turns) => makeSymmetry('reflection', turns)),
);

export const D5_SYMMETRIES: readonly D5Symmetry[] = Object.freeze([
  ...D5_ROTATIONS,
  ...D5_REFLECTIONS,
]);

function mod5(value: number): FifthTurn {
  return (((value % 5) + 5) % 5) as FifthTurn;
}

function assertTurns(turns: number): asserts turns is FifthTurn {
  if (!Number.isInteger(turns) || turns < 0 || turns > 4) {
    throw new Error(`D5 turns must be an integer in 0..4, got ${turns}`);
  }
}

/** Return the canonical descriptor for one of the ten D5 elements. */
export function getD5Symmetry(kind: D5Kind, turns: number): D5Symmetry {
  assertTurns(turns);
  return kind === 'rotation' ? D5_ROTATIONS[turns] : D5_REFLECTIONS[turns];
}

/** The exact inverse group element. Every reflection is self-inverse. */
export function inverseD5Symmetry(symmetry: D5Symmetry): D5Symmetry {
  return symmetry.kind === 'rotation'
    ? getD5Symmetry('rotation', mod5(-symmetry.turns))
    : getD5Symmetry('reflection', symmetry.turns);
}

/**
 * Compose two elements. The returned map applies `before` first and `after`
 * second.
 */
export function composeD5Symmetries(
  after: D5Symmetry,
  before: D5Symmetry,
): D5Symmetry {
  const afterSign = after.kind === 'rotation' ? 1 : -1;
  const beforeSign = before.kind === 'rotation' ? 1 : -1;
  const kind: D5Kind = afterSign * beforeSign === 1 ? 'rotation' : 'reflection';
  const turns = mod5(afterSign * before.turns + after.turns);
  return getD5Symmetry(kind, turns);
}

/** Transform a node id using its discrete (sector, ring, position) address. */
export function transformNode(
  board: Board,
  node: number,
  symmetry: D5Symmetry,
): number {
  if (!Number.isInteger(node) || node < 0 || node >= board.n) {
    throw new Error(`node must be an integer in 0..${board.n - 1}, got ${node}`);
  }
  const x = board.ringOf[node];
  const t = board.sectorOf[node] * x + board.posOf[node];
  const circumference = 5 * x;
  const transformed =
    symmetry.kind === 'rotation' ? t + symmetry.turns * x : symmetry.turns * x - t;
  const tt = ((transformed % circumference) + circumference) % circumference;
  const sector = Math.floor(tt / x);
  const position = tt % x;
  return board.idx(sector, x, position);
}

/** Build forward and inverse node maps for a board and D5 element. */
export function getD5Maps(board: Board, symmetry: D5Symmetry): D5Maps {
  const forward = new Int32Array(board.n);
  const inverse = new Int32Array(board.n);
  for (let u = 0; u < board.n; u++) {
    const v = transformNode(board, u, symmetry);
    forward[u] = v;
    inverse[v] = u;
  }
  return { forward, inverse };
}

/**
 * Transform a complete stone array. Values move with nodes:
 * result[map[u]] = stones[u].
 */
export function transformStones(
  board: Board,
  stones: ArrayLike<number>,
  symmetry: D5Symmetry,
): Int8Array {
  if (stones.length !== board.n) {
    throw new Error(`stones length must be ${board.n}, got ${stones.length}`);
  }
  const { forward } = getD5Maps(board, symmetry);
  const transformed = new Int8Array(board.n);
  for (let u = 0; u < board.n; u++) transformed[forward[u]] = stones[u];
  return transformed;
}

/** Transform a game action; pass and swap have no board coordinate. */
export function transformAction(
  board: Board,
  action: GameAction,
  symmetry: D5Symmetry,
): GameAction {
  switch (action.type) {
    case 'place':
      return { type: 'place', node: transformNode(board, action.node, symmetry) };
    case 'pass':
      return { type: 'pass' };
    case 'swap':
      return { type: 'swap' };
  }
}

/** Transform an action log without mutating it. */
export function transformActions(
  board: Board,
  actions: readonly GameAction[],
  symmetry: D5Symmetry,
): GameAction[] {
  return actions.map((action) => transformAction(board, action, symmetry));
}
