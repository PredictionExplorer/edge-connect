import { EMPTY } from '../scoring';
import {
  configHandicap,
  isLegalAction,
  replay,
  type GameAction,
  type GameConfig,
  type GameState,
  type Mode,
} from '../game';
import {
  fnv1a64,
  STAR_ACTION_LAYOUT_SCHEMA_ID,
  STAR_FEATURE_SCHEMA_ID,
  STAR_MAX_HANDICAP,
  STAR_RULES_HASH,
  STAR_RULES_SCHEMA_ID,
} from '../rules';
import { StarAiError } from './errors';

export const STAR_AI_PROTOCOL_SCHEMA_ID = 'edgeconnect.star.ai.atomic.v3' as const;
export const STAR_AI_PROTOCOL_VERSION = 3 as const;
export const STAR_FEATURE_SCHEMA_VERSION = 4 as const;
export const STAR_ACTION_LAYOUT_VERSION = 1 as const;

/** Exact canonical feature contract from training/startrain/contracts.py. */
export const STAR_FEATURE_CONTRACT = [
  'startrain/features/v4;',
  'semantic-key=rings,stones,to_move,moves_left,opening,terminal,mode,handicap,',
  'pie_pending,swap_available,current_turn,previous_turn,own_previous_turn,',
  'handicap_stones;',
  'context=history_known,pda;',
  'perspective=current-player;',
  'node=empty,current,opponent,owner-current,owner-opponent,owner-unclaimed,',
  'alive-current,alive-opponent,peri,quark,ring-fraction,arm-distance,',
  'degree-fraction,bridge,legal,placed-this-turn,own-previous-turn,',
  'opponent-previous-turn,handicap-stone;',
  'global=rings,occupancy,current-count,opponent-count,moves-left-of-turn,',
  'opening,terminal,current-score,opponent-score,margin,current-peries,',
  'opponent-peries,current-quarks,opponent-quarks,current-stars,',
  'opponent-stars,contested-peries,turn-size,handicap,handicap-phase,',
  'handicap-remaining,pie-pending,swap-available,history-known,pda;',
  'edges=tangential,radial-diagonal,bridge;',
  'relations=ring-difference,angular-offset-bucket,shortest-path-bucket,peri-pair;',
  'sample-actions=node[0:N];',
  'batch-actions=node[0:maxN];',
  'soft-policy=katago-temperature-4',
].join('');

export const STAR_FEATURE_SCHEMA_HASH = fnv1a64(STAR_FEATURE_CONTRACT);

/** Atomic actions: one placement or the pie swap. */
export type AtomicGameAction = GameAction;

export interface StarAiPlacementHistory {
  /** Nodes placed so far in the turn in progress. */
  currentTurn: number[];
  /** Nodes of the most recently completed turn (the opponent's). */
  previousTurn: number[];
  /** Nodes of the completed turn before that (the current player's). */
  ownPreviousTurn: number[];
  /** Every opening (handicap) stone. */
  handicapStones: number[];
}

export interface StarAiSemanticState {
  rings: number;
  /** Dense node order, with values -1, 0, or 1. */
  stones: number[];
  toMove: 0 | 1;
  movesLeft: number;
  /** True while the first (handicap) turn is in progress. */
  opening: boolean;
  terminal: boolean;
  mode: Mode;
  handicap: number;
  pie: boolean;
  /** True when the player to move may take the pie swap right now. */
  swapAvailable: boolean;
  swapped: boolean;
  history: StarAiPlacementHistory;
}

export interface StarAiRequest {
  schema: typeof STAR_AI_PROTOCOL_SCHEMA_ID;
  version: typeof STAR_AI_PROTOCOL_VERSION;
  requestId: string;
  rulesSchema: typeof STAR_RULES_SCHEMA_ID;
  rulesHash: typeof STAR_RULES_HASH;
  featureSchema: typeof STAR_FEATURE_SCHEMA_ID;
  featureSchemaVersion: typeof STAR_FEATURE_SCHEMA_VERSION;
  featureSchemaHash: string;
  actionLayout: typeof STAR_ACTION_LAYOUT_SCHEMA_ID;
  actionLayoutVersion: typeof STAR_ACTION_LAYOUT_VERSION;
  stateHash: string;
  state: StarAiSemanticState;
  /** Atomic wire codes: dense node ids for placements, node count for a swap. */
  actionLog: number[];
  /** Ascending empty node ids (the swap is reported by state.swapAvailable). */
  legalActions: number[];
}

export interface StarAiResponse {
  schema: typeof STAR_AI_PROTOCOL_SCHEMA_ID;
  version: typeof STAR_AI_PROTOCOL_VERSION;
  requestId: string;
  rulesHash: typeof STAR_RULES_HASH;
  stateHash: string;
  action: AtomicGameAction;
}

export type AiResponseAcceptance =
  | { ok: true; action: AtomicGameAction; response: StarAiResponse }
  | { ok: false; code: 'stale' | 'illegal' | 'protocol'; message: string };

let requestSequence = 0;

function hex64(value: bigint): string {
  return value.toString(16).padStart(16, '0');
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return (
    actual.length === expected.length &&
    actual.every((key, index) => key === expected[index])
  );
}

export function newAiRequestId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  requestSequence += 1;
  return `star-ai-${Date.now().toString(36)}-${requestSequence.toString(36)}`;
}

/** Wire code of an atomic action: the node for placements, the node count for a swap. */
export function actionToCode(action: AtomicGameAction, nodeCount: number): number {
  if (action.type === 'swap') return nodeCount;
  return action.node;
}

export function codeToAction(code: number, nodeCount: number): AtomicGameAction {
  if (Number.isInteger(code) && code >= 0 && code < nodeCount) {
    return { type: 'place', node: code };
  }
  if (code === nodeCount) return { type: 'swap' };
  throw new StarAiError('protocol', `Invalid atomic action code: ${String(code)}.`);
}

export function semanticStateFromGame(state: GameState): StarAiSemanticState {
  return {
    rings: state.config.rings,
    stones: Array.from(state.stones),
    toMove: state.toMove,
    movesLeft: state.movesLeft,
    opening: state.turnCount === 0,
    terminal: state.over,
    mode: state.config.mode,
    handicap: configHandicap(state.config),
    pie: state.config.pieRule,
    swapAvailable: state.canSwap,
    swapped: state.swapped,
    // History sets are order-free in the engine; ascending order keeps the
    // semantic state canonical for equality checks and replay verification.
    history: {
      currentTurn: ascending(state.currentTurnMoves),
      previousTurn: ascending(state.previousTurnMoves),
      ownPreviousTurn: ascending(state.ownPreviousTurnMoves),
      handicapStones: ascending(state.handicapStones),
    },
  };
}

function ascending(nodes: readonly number[]): number[] {
  return [...nodes].sort((left, right) => left - right);
}

/** The opener's pie decision is pending before the first stone of a pie game. */
export function isPiePending(state: StarAiSemanticState): boolean {
  return state.pie && state.opening && state.movesLeft === state.handicap;
}

/**
 * Mirrors star_engine::GameState::hash64 so browser requests can be checked
 * against the replayed WASM state without trusting presentation metadata.
 */
export function semanticStateHash(state: StarAiSemanticState): string {
  if (typeof BigInt !== 'function') {
    throw new StarAiError('unavailable', 'AI controllers require BigInt browser support.');
  }
  const mask64 = BigInt('0xffffffffffffffff');
  const splitmix64 = (input: bigint): bigint => {
    let value = (input + BigInt('0x9e3779b97f4a7c15')) & mask64;
    value =
      ((value ^ (value >> BigInt(30))) * BigInt('0xbf58476d1ce4e5b9')) & mask64;
    value =
      ((value ^ (value >> BigInt(27))) * BigInt('0x94d049bb133111eb')) & mask64;
    return (value ^ (value >> BigInt(31))) & mask64;
  };
  let hash = splitmix64(BigInt('0xd0ab1e5a7a120000') ^ BigInt(state.rings));
  for (let node = 0; node < state.stones.length; node++) {
    const player = state.stones[node];
    if (player !== 0 && player !== 1) continue;
    const index = BigInt(player * 448 + node);
    hash ^= splitmix64(BigInt('0x51a7e00000000000') ^ index);
  }
  const historySets: Array<[string, number[]]> = [
    ['0x5a00000000000000', state.history.currentTurn],
    ['0x5b00000000000000', state.history.previousTurn],
    ['0x5c00000000000000', state.history.ownPreviousTurn],
    ['0x5d00000000000000', state.history.handicapStones],
  ];
  for (const [salt, nodes] of historySets) {
    for (const node of nodes) {
      hash ^= splitmix64(BigInt(salt) ^ BigInt(node));
    }
  }
  hash ^= splitmix64(BigInt('0x7000000000000000') ^ BigInt(state.toMove));
  hash ^= splitmix64(BigInt('0x7100000000000000') ^ BigInt(state.movesLeft));
  hash ^= splitmix64(BigInt('0x7200000000000000') ^ BigInt(state.opening ? 1 : 0));
  hash ^= splitmix64(BigInt('0x7400000000000000') ^ BigInt(state.terminal ? 1 : 0));
  hash ^= splitmix64(
    BigInt('0x7500000000000000') ^ BigInt(state.mode === 'classic' ? 0 : 1),
  );
  hash ^= splitmix64(BigInt('0x7600000000000000') ^ BigInt(state.handicap));
  hash ^= splitmix64(
    BigInt('0x7700000000000000') ^ BigInt(isPiePending(state) ? 1 : 0),
  );
  hash ^= splitmix64(
    BigInt('0x7800000000000000') ^ BigInt(state.swapAvailable ? 1 : 0),
  );
  return `zobrist64:${hex64(hash & mask64)}`;
}

export function legalActionCodes(state: GameState): number[] {
  if (state.over) return [];
  const actions: number[] = [];
  for (let node = 0; node < state.board.n; node++) {
    if (state.stones[node] === EMPTY) actions.push(node);
  }
  return actions;
}

export function validateAiConfig(config: GameConfig): void {
  const handicap = configHandicap(config);
  if (
    (config.mode !== 'classic' && config.mode !== 'double') ||
    !Number.isInteger(handicap) ||
    handicap < 1 ||
    handicap > STAR_MAX_HANDICAP ||
    (config.pieRule && handicap !== 1)
  ) {
    throw new StarAiError('protocol', 'AI controllers received an invalid rule variant.');
  }
}

export function buildAiRequest(
  config: GameConfig,
  log: readonly GameAction[],
  requestId = newAiRequestId(),
): StarAiRequest {
  validateAiConfig(config);
  const game = replay(config, [...log]);
  if (game.over) {
    throw new StarAiError('protocol', 'Cannot request an action for a terminal position.');
  }
  const nodeCount = game.board.n;
  const actionLog = log.map((action) => {
    if (action.type !== 'place' && action.type !== 'swap') {
      throw new StarAiError('protocol', 'AI action logs contain placements and swaps only.');
    }
    return actionToCode(action, nodeCount);
  });
  const state = semanticStateFromGame(game);

  return {
    schema: STAR_AI_PROTOCOL_SCHEMA_ID,
    version: STAR_AI_PROTOCOL_VERSION,
    requestId,
    rulesSchema: STAR_RULES_SCHEMA_ID,
    rulesHash: STAR_RULES_HASH,
    featureSchema: STAR_FEATURE_SCHEMA_ID,
    featureSchemaVersion: STAR_FEATURE_SCHEMA_VERSION,
    featureSchemaHash: STAR_FEATURE_SCHEMA_HASH,
    actionLayout: STAR_ACTION_LAYOUT_SCHEMA_ID,
    actionLayoutVersion: STAR_ACTION_LAYOUT_VERSION,
    stateHash: semanticStateHash(state),
    state,
    actionLog,
    legalActions: legalActionCodes(game),
  };
}

export function makeAiResponse(
  request: StarAiRequest,
  action: AtomicGameAction,
): StarAiResponse {
  return {
    schema: STAR_AI_PROTOCOL_SCHEMA_ID,
    version: STAR_AI_PROTOCOL_VERSION,
    requestId: request.requestId,
    rulesHash: STAR_RULES_HASH,
    stateHash: request.stateHash,
    action,
  };
}

function parseAction(rawAction: unknown): AtomicGameAction {
  if (!isRecord(rawAction)) {
    throw new StarAiError('protocol', 'AI response must contain one atomic action.');
  }
  if (rawAction.type === 'swap') {
    if (!hasExactKeys(rawAction, ['type'])) {
      throw new StarAiError('protocol', 'A swap action contains unknown fields.');
    }
    return { type: 'swap' };
  }
  if (rawAction.type !== 'place') {
    throw new StarAiError('protocol', 'AI response must contain one atomic action.');
  }
  if (
    typeof rawAction.node !== 'number' ||
    !Number.isInteger(rawAction.node) ||
    rawAction.node < 0
  ) {
    throw new StarAiError('protocol', 'A placement must contain a non-negative node id.');
  }
  if (!hasExactKeys(rawAction, ['type', 'node'])) {
    throw new StarAiError('protocol', 'A placement action contains unknown fields.');
  }
  return { type: 'place', node: rawAction.node };
}

/** Whether an atomic action is legal for the request's semantic state. */
export function isRequestLegalAction(
  request: StarAiRequest,
  action: AtomicGameAction,
): boolean {
  if (action.type === 'swap') return request.state.swapAvailable;
  return request.legalActions.includes(action.node);
}

export function parseAiResponse(request: StarAiRequest, payload: unknown): StarAiResponse {
  if (!isRecord(payload)) {
    throw new StarAiError('protocol', 'AI response must be an object.');
  }
  if (
    payload.schema !== STAR_AI_PROTOCOL_SCHEMA_ID ||
    payload.version !== STAR_AI_PROTOCOL_VERSION ||
    payload.rulesHash !== STAR_RULES_HASH
  ) {
    throw new StarAiError('protocol', 'AI response schema or rules hash is incompatible.');
  }
  if (payload.requestId !== request.requestId || payload.stateHash !== request.stateHash) {
    throw new StarAiError('stale', 'AI response belongs to an obsolete position.');
  }
  if ('actions' in payload) {
    throw new StarAiError('protocol', 'AI response must not contain a multi-action turn.');
  }
  if (
    !hasExactKeys(payload, [
      'schema',
      'version',
      'requestId',
      'rulesHash',
      'stateHash',
      'action',
    ])
  ) {
    throw new StarAiError('protocol', 'AI response contains unknown fields.');
  }

  const action = parseAction(payload.action);
  if (!isRequestLegalAction(request, action)) {
    throw new StarAiError('illegal', 'AI returned an illegal atomic action.');
  }

  return {
    schema: STAR_AI_PROTOCOL_SCHEMA_ID,
    version: STAR_AI_PROTOCOL_VERSION,
    requestId: request.requestId,
    rulesHash: STAR_RULES_HASH,
    stateHash: request.stateHash,
    action,
  };
}

/**
 * Final mutation gate. It replays current app state, rejects stale identity,
 * parses the untrusted payload, and checks legality again immediately before
 * the store may append the action.
 */
export function acceptAiResponse(
  request: StarAiRequest,
  payload: unknown,
  currentConfig: GameConfig,
  currentLog: readonly GameAction[],
): AiResponseAcceptance {
  let current: GameState;
  try {
    validateAiConfig(currentConfig);
    current = replay(currentConfig, [...currentLog]);
  } catch {
    return { ok: false, code: 'stale', message: 'The game changed before AI replied.' };
  }
  const currentHash = semanticStateHash(semanticStateFromGame(current));
  if (currentHash !== request.stateHash) {
    return { ok: false, code: 'stale', message: 'The game changed before AI replied.' };
  }

  let response: StarAiResponse;
  try {
    response = parseAiResponse(request, payload);
  } catch (error) {
    if (error instanceof StarAiError) {
      const code =
        error.code === 'stale' ? 'stale' : error.code === 'illegal' ? 'illegal' : 'protocol';
      return { ok: false, code, message: error.message };
    }
    return { ok: false, code: 'protocol', message: 'AI response is invalid.' };
  }

  if (!isLegalAction(current, response.action)) {
    return { ok: false, code: 'illegal', message: 'AI returned an illegal atomic action.' };
  }
  return { ok: true, action: response.action, response };
}
