import { getBoard, MAX_RINGS, type Board } from '../board';
import { EMPTY, scorePosition } from '../scoring';
import type { StarAiSemanticState } from './protocol';

export const STAR_NODE_FEATURE_NAMES = [
  'empty',
  'current_stone',
  'opponent_stone',
  'owner_current',
  'owner_opponent',
  'owner_unclaimed',
  'alive_current',
  'alive_opponent',
  'is_peri',
  'is_quark',
  'ring_fraction',
  'arm_distance_fraction',
  'degree_fraction',
  'is_bridge',
  'legal',
] as const;

export const STAR_GLOBAL_FEATURE_NAMES = [
  'rings_fraction',
  'occupancy_fraction',
  'current_stone_fraction',
  'opponent_stone_fraction',
  'moves_left_fraction',
  'opening',
  'pass_streak_fraction',
  'terminal',
  'current_total_scaled',
  'opponent_total_scaled',
  'score_margin_scaled',
  'current_peries_fraction',
  'opponent_peries_fraction',
  'current_quarks_fraction',
  'opponent_quarks_fraction',
  'current_stars_fraction',
  'opponent_stars_fraction',
  'contested_peries_fraction',
] as const;

export const STAR_MODEL_INPUT_NAMES = [
  'node_features',
  'global_features',
  'neighbor_index',
  'neighbor_mask',
  'neighbor_edge_type',
  'node_mask',
  'legal_action_mask',
] as const;

export const STAR_MODEL_OUTPUT_NAMES = [
  'policy_logits',
  'wdl_logits',
  'score_margin_logits',
  'ownership_logits',
  'alive_logits',
  'soft_policy_logits',
] as const;

export const STAR_NODE_FEATURE_DIM = STAR_NODE_FEATURE_NAMES.length;
export const STAR_GLOBAL_FEATURE_DIM = STAR_GLOBAL_FEATURE_NAMES.length;
const FLOAT32_SCRATCH = new ArrayBuffer(4);
const FLOAT32_SCRATCH_FLOAT = new Float32Array(FLOAT32_SCRATCH);
const FLOAT32_SCRATCH_WORD = new Uint32Array(FLOAT32_SCRATCH);

export interface EncodedStarFeatures {
  nodeCount: number;
  maxDegree: number;
  nodeFeatures: Float32Array;
  globalFeatures: Float32Array;
  neighborIndex: BigInt64Array;
  neighborMask: Uint8Array;
  neighborEdgeType: BigInt64Array;
  nodeMask: Uint8Array;
  legalActionMask: Uint8Array;
}

function bool(value: boolean): number {
  return value ? 1 : 0;
}

function validateSemanticState(state: StarAiSemanticState): Board {
  const board = getBoard(state.rings);
  if (
    state.stones.length !== board.n ||
    state.stones.some(
      (stone) => !Number.isInteger(stone) || (stone !== EMPTY && stone !== 0 && stone !== 1),
    )
  ) {
    throw new Error(`stones must contain ${board.n} values from -1, 0, or 1`);
  }
  if (state.toMove !== 0 && state.toMove !== 1) throw new Error('toMove must be 0 or 1');
  if (![0, 1, 2].includes(state.movesLeft)) throw new Error('movesLeft must be in 0..2');
  if (![0, 1, 2].includes(state.passStreak)) throw new Error('passStreak must be in 0..2');

  const occupied = state.stones.filter((stone) => stone !== EMPTY).length;
  const boardFull = occupied === board.n;
  if (state.terminal !== (boardFull || state.passStreak === 2)) {
    throw new Error('terminal must equal board-full or passStreak == 2');
  }
  if (state.movesLeft === 0 && !boardFull) {
    throw new Error('movesLeft == 0 is valid only on a full board');
  }
  if (boardFull && state.movesLeft > 1) {
    throw new Error('a full board may retain at most one placement');
  }
  if (
    state.opening &&
    (state.toMove !== 0 ||
      state.movesLeft !== 1 ||
      state.passStreak !== 0 ||
      occupied !== 0 ||
      state.terminal)
  ) {
    throw new Error('invalid one-stone opening metadata');
  }
  return board;
}

function maximumDegree(board: Board): number {
  let maximum = 0;
  for (let node = 0; node < board.n; node++) {
    maximum = Math.max(maximum, board.adjOff[node + 1] - board.adjOff[node]);
  }
  return maximum;
}

/** Python topology.py edge classes: tangential=0, radial/diagonal=1, bridge=2. */
export function topologyEdgeType(board: Board, first: number, second: number): 0 | 1 | 2 {
  if (board.ringOf[first] === 1 && board.ringOf[second] === 1) return 2;
  return board.ringOf[first] === board.ringOf[second] ? 0 : 1;
}

export function actionCodeToModelIndex(action: number, nodeCount: number): number {
  if (action === -1) return nodeCount;
  if (Number.isInteger(action) && action >= 0 && action < nodeCount) return action;
  throw new Error(`action ${String(action)} is outside the nodes-then-pass layout`);
}

export function modelIndexToActionCode(index: number, nodeCount: number): number {
  if (!Number.isInteger(index) || index < 0 || index > nodeCount) {
    throw new Error(`model action index ${String(index)} is outside the action layout`);
  }
  return index === nodeCount ? -1 : index;
}

export function numberToFloat16Bits(value: number): number {
  FLOAT32_SCRATCH_FLOAT[0] = value;
  const bits = FLOAT32_SCRATCH_WORD[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x007f_ffff;

  if (exponent === 0xff) {
    return sign | 0x7c00 | (mantissa === 0 ? 0 : 0x0200);
  }
  if (exponent > 142) return sign | 0x7c00;
  if (exponent < 103) return sign;

  let roundedMantissa = (bits >>> 12) & 0x07ff;
  if (exponent < 113) {
    roundedMantissa |= 0x0800;
    return (
      sign |
      ((roundedMantissa >>> (114 - exponent)) +
        ((roundedMantissa >>> (113 - exponent)) & 1))
    );
  }

  const half = sign | ((exponent - 112) << 10) | (roundedMantissa >>> 1);
  return half + (roundedMantissa & 1);
}

export function float16BitsToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x03ff;
  if (exponent === 0) return sign * 2 ** -14 * (mantissa / 1024);
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : Number.NaN;
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}

export function float32ToFloat16Array(values: ArrayLike<number>): Uint16Array {
  return Uint16Array.from(values, numberToFloat16Bits);
}

export function float16ToFloat32Array(values: ArrayLike<number>): Float32Array {
  return Float32Array.from(values, float16BitsToNumber);
}

/** Exact browser port of training/startrain/features.py schema v2. */
export function encodeStarFeatures(state: StarAiSemanticState): EncodedStarFeatures {
  const board = validateSemanticState(state);
  const score = scorePosition(board, state.stones);
  const current = state.toMove;
  const opponent = 1 - current;
  const maxDegree = maximumDegree(board);
  const nodeFeatures = new Float32Array(board.n * STAR_NODE_FEATURE_DIM);
  const neighborIndex = new BigInt64Array(board.n * maxDegree);
  const neighborMask = new Uint8Array(board.n * maxDegree);
  const neighborEdgeType = new BigInt64Array(board.n * maxDegree);
  const nodeMask = new Uint8Array(board.n).fill(1);
  const legalActionMask = new Uint8Array(board.n + 1);

  let currentCount = 0;
  let opponentCount = 0;
  for (let node = 0; node < board.n; node++) {
    const stone = state.stones[node];
    const empty = stone === EMPTY;
    const currentStone = stone === current;
    const opponentStone = stone === opponent;
    if (currentStone) currentCount += 1;
    if (opponentStone) opponentCount += 1;

    const ring = board.ringOf[node];
    const position = board.posOf[node];
    const degree = board.adjOff[node + 1] - board.adjOff[node];
    const base = node * STAR_NODE_FEATURE_DIM;
    const values = [
      bool(empty),
      bool(currentStone),
      bool(opponentStone),
      bool(score.nodeOwner[node] === current),
      bool(score.nodeOwner[node] === opponent),
      bool(score.nodeOwner[node] === EMPTY),
      bool(Boolean(score.aliveStone[node]) && currentStone),
      bool(Boolean(score.aliveStone[node]) && opponentStone),
      bool(Boolean(board.isPeri[node])),
      bool(Boolean(board.isQuark[node])),
      ring / state.rings,
      Math.min(position, ring - position) / ring,
      degree / maxDegree,
      bool(ring === 1),
      bool(empty && !state.terminal),
    ];
    nodeFeatures.set(values, base);
    legalActionMask[node] = bool(empty && !state.terminal);

    for (let offset = 0; offset < degree; offset++) {
      const edge = board.adjOff[node] + offset;
      const neighbor = board.adj[edge];
      const target = node * maxDegree + offset;
      neighborIndex[target] = BigInt(neighbor);
      neighborMask[target] = 1;
      neighborEdgeType[target] = BigInt(topologyEdgeType(board, node, neighbor));
    }
  }
  legalActionMask[board.n] = bool(!state.terminal);

  const occupied = currentCount + opponentCount;
  const currentScore = score.players[current];
  const opponentScore = score.players[opponent];
  const scoreScale = 181;
  const starScale = Math.max(1, board.periCount / 2);
  const globalFeatures = new Float32Array([
    state.rings / MAX_RINGS,
    occupied / board.n,
    currentCount / board.n,
    opponentCount / board.n,
    state.movesLeft / 2,
    bool(state.opening),
    state.passStreak / 2,
    bool(state.terminal),
    currentScore.total / scoreScale,
    opponentScore.total / scoreScale,
    (currentScore.total - opponentScore.total) / scoreScale,
    currentScore.peries / board.periCount,
    opponentScore.peries / board.periCount,
    currentScore.quarks / 5,
    opponentScore.quarks / 5,
    currentScore.stars / starScale,
    opponentScore.stars / starScale,
    score.contestedPeries / board.periCount,
  ]);

  return {
    nodeCount: board.n,
    maxDegree,
    nodeFeatures,
    globalFeatures,
    neighborIndex,
    neighborMask,
    neighborEdgeType,
    nodeMask,
    legalActionMask,
  };
}
