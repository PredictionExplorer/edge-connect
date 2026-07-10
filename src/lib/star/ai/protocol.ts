import { EMPTY } from '../scoring';
import {
  isLegalAction,
  replay,
  type GameAction,
  type GameConfig,
  type GameState,
} from '../game';
import {
  STAR_ACTION_LAYOUT_SCHEMA_ID,
  STAR_FEATURE_SCHEMA_ID,
  STAR_RULES_HASH,
  STAR_RULES_SCHEMA_ID,
} from '../rules';
import { StarAiError } from './errors';

export const STAR_AI_PROTOCOL_SCHEMA_ID = 'edgeconnect.star.ai.atomic.v1' as const;
export const STAR_AI_PROTOCOL_VERSION = 1 as const;
export const STAR_FEATURE_SCHEMA_VERSION = 2 as const;
export const STAR_ACTION_LAYOUT_VERSION = 1 as const;

/** Exact canonical feature contract from training/startrain/contracts.py. */
export const STAR_FEATURE_CONTRACT = [
  'startrain/features/v2;',
  'semantic-key=rings,stones,to_move,moves_left,opening,pass_streak,terminal;',
  'perspective=current-player;',
  'node=empty,current,opponent,owner-current,owner-opponent,owner-unclaimed,',
  'alive-current,alive-opponent,peri,quark,ring-fraction,arm-distance,',
  'degree-fraction,bridge,legal;',
  'global=rings,occupancy,current-count,opponent-count,moves-left,opening,',
  'pass-streak,terminal,current-score,opponent-score,margin,current-peries,',
  'opponent-peries,current-quarks,opponent-quarks,current-stars,',
  'opponent-stars,contested-peries;',
  'edges=tangential,radial-diagonal,bridge;',
  'sample-actions=node[0:N],pass[N];',
  'batch-actions=node[0:maxN],pass[maxN];',
  'soft-policy=katago-temperature-4',
].join('');

export const STAR_FEATURE_SCHEMA_HASH = '59a7da1c00bac4d2' as const;

export type AtomicGameAction = Extract<GameAction, { type: 'place' | 'pass' }>;

export interface StarAiSemanticState {
  rings: number;
  /** Dense node order, with values -1, 0, or 1. */
  stones: number[];
  toMove: 0 | 1;
  movesLeft: number;
  opening: boolean;
  passStreak: number;
  terminal: boolean;
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
  /** Atomic wire codes: node id for place, -1 for pass. */
  actionLog: number[];
  /** Ascending empty node ids followed by pass (-1). */
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

export function newAiRequestId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  requestSequence += 1;
  return `star-ai-${Date.now().toString(36)}-${requestSequence.toString(36)}`;
}

export function actionToCode(action: AtomicGameAction): number {
  return action.type === 'pass' ? -1 : action.node;
}

export function codeToAction(code: number): AtomicGameAction {
  if (code === -1) return { type: 'pass' };
  if (Number.isInteger(code) && code >= 0) return { type: 'place', node: code };
  throw new StarAiError('protocol', `Invalid atomic action code: ${String(code)}.`);
}

export function semanticStateFromGame(state: GameState): StarAiSemanticState {
  return {
    rings: state.config.rings,
    stones: Array.from(state.stones),
    toMove: state.toMove,
    movesLeft: state.movesLeft,
    opening: state.turnCount === 0 && state.stonesPlaced === 0,
    passStreak: state.passStreak,
    terminal: state.over,
  };
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
  hash ^= splitmix64(BigInt('0x7000000000000000') ^ BigInt(state.toMove));
  hash ^= splitmix64(BigInt('0x7100000000000000') ^ BigInt(state.movesLeft));
  hash ^= splitmix64(BigInt('0x7200000000000000') ^ BigInt(state.opening ? 1 : 0));
  hash ^= splitmix64(BigInt('0x7300000000000000') ^ BigInt(state.passStreak));
  hash ^= splitmix64(BigInt('0x7400000000000000') ^ BigInt(state.terminal ? 1 : 0));
  return `zobrist64:${hex64(hash & mask64)}`;
}

export function legalActionCodes(state: GameState): number[] {
  if (state.over) return [];
  const actions: number[] = [];
  for (let node = 0; node < state.board.n; node++) {
    if (state.stones[node] === EMPTY) actions.push(node);
  }
  actions.push(-1);
  return actions;
}

export function buildAiRequest(
  config: GameConfig,
  log: readonly GameAction[],
  requestId = newAiRequestId(),
): StarAiRequest {
  if (config.mode !== 'double' || config.pieRule) {
    throw new StarAiError(
      'protocol',
      'AI controllers require Double *Star with the pie rule disabled.',
    );
  }

  const actionLog = log.map((action) => {
    if (action.type === 'swap') {
      throw new StarAiError('protocol', 'Pie-rule swaps are outside the AI protocol.');
    }
    return actionToCode(action);
  });
  const game = replay(config, [...log]);
  if (game.over) {
    throw new StarAiError('protocol', 'Cannot request an action for a terminal position.');
  }
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

  const rawAction = payload.action;
  if (!isRecord(rawAction) || (rawAction.type !== 'place' && rawAction.type !== 'pass')) {
    throw new StarAiError('protocol', 'AI response must contain one atomic action.');
  }

  let action: AtomicGameAction;
  if (rawAction.type === 'pass') {
    if ('node' in rawAction || Object.keys(rawAction).some((key) => key !== 'type')) {
      throw new StarAiError('protocol', 'A pass action cannot include a node.');
    }
    action = { type: 'pass' };
  } else {
    if (
      typeof rawAction.node !== 'number' ||
      !Number.isInteger(rawAction.node) ||
      rawAction.node < 0
    ) {
      throw new StarAiError('protocol', 'A placement must contain a non-negative node id.');
    }
    if (Object.keys(rawAction).some((key) => key !== 'type' && key !== 'node')) {
      throw new StarAiError('protocol', 'A placement action contains unknown fields.');
    }
    action = { type: 'place', node: rawAction.node };
  }

  if (!request.legalActions.includes(actionToCode(action))) {
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
  if (currentConfig.mode !== 'double' || currentConfig.pieRule) {
    return { ok: false, code: 'stale', message: 'The game changed before AI replied.' };
  }
  let current: GameState;
  try {
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
