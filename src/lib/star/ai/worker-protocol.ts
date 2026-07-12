import {
  STAR_ACTION_LAYOUT_SCHEMA_ID,
  STAR_FEATURE_SCHEMA_ID,
  STAR_RULES_HASH,
  STAR_RULES_SCHEMA_ID,
} from '../rules';
import { getBoard, isSupportedRings } from '../board';
import {
  parseUnboundStarAiDecision,
  type StarAiDecision,
  type StarAiSearchBudget,
} from './decision';
import { StarAiError, asStarAiError, type StarAiErrorCode } from './errors';
import {
  MAX_BROWSER_AI_MAX_CONSIDERED,
  MAX_BROWSER_AI_SIMULATIONS,
} from './manifest';
import {
  STAR_ACTION_LAYOUT_VERSION,
  STAR_AI_PROTOCOL_SCHEMA_ID,
  STAR_AI_PROTOCOL_VERSION,
  STAR_FEATURE_SCHEMA_HASH,
  STAR_FEATURE_SCHEMA_VERSION,
  semanticStateHash,
  type StarAiRequest,
  type StarAiSemanticState,
} from './protocol';

export type StarAiWorkerCommand =
  | {
      type: 'choose';
      taskId: string;
      request: StarAiRequest;
      search: StarAiSearchBudget | null;
    }
  | { type: 'cancel'; taskId: string };

export type StarAiWorkerEvent =
  | { type: 'ready'; protocolVersion: 2 }
  | { type: 'result'; taskId: string; decision: StarAiDecision }
  | {
      type: 'error';
      taskId: string;
      error: { code: StarAiErrorCode; message: string; retryable: boolean };
    };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isTaskId(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0 && value.length <= 200;
}

function hasExactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return (
    actual.length === expected.length &&
    actual.every((key, index) => key === expected[index])
  );
}

function isIntegerArray(value: unknown, minimum: number): value is number[] {
  return (
    Array.isArray(value) &&
    value.every(
      (item) => typeof item === 'number' && Number.isInteger(item) && item >= minimum,
    )
  );
}

export function parseBrowserSearchBudget(value: unknown): StarAiSearchBudget {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ['simulations', 'maxConsidered']) ||
    typeof value.simulations !== 'number' ||
    !Number.isSafeInteger(value.simulations) ||
    value.simulations <= 0 ||
    value.simulations > MAX_BROWSER_AI_SIMULATIONS ||
    typeof value.maxConsidered !== 'number' ||
    !Number.isSafeInteger(value.maxConsidered) ||
    value.maxConsidered <= 0 ||
    value.maxConsidered > MAX_BROWSER_AI_MAX_CONSIDERED
  ) {
    throw new StarAiError(
      'protocol',
      `Browser AI search budget must use simulations in 1..${MAX_BROWSER_AI_SIMULATIONS} ` +
        `and max-considered in 1..${MAX_BROWSER_AI_MAX_CONSIDERED}.`,
    );
  }
  return {
    simulations: value.simulations,
    maxConsidered: value.maxConsidered,
  };
}

function parseSemanticState(value: unknown): StarAiSemanticState {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      'rings',
      'stones',
      'toMove',
      'movesLeft',
      'opening',
      'terminal',
    ]) ||
    !isSupportedRings(value.rings) ||
    !isIntegerArray(value.stones, -1) ||
    value.stones.some((stone) => stone > 1) ||
    value.stones.length !== getBoard(value.rings).n ||
    (value.toMove !== 0 && value.toMove !== 1) ||
    typeof value.movesLeft !== 'number' ||
    !Number.isInteger(value.movesLeft) ||
    value.movesLeft < 0 ||
    value.movesLeft > 2 ||
    typeof value.opening !== 'boolean' ||
    typeof value.terminal !== 'boolean'
  ) {
    throw new StarAiError('protocol', 'Worker received an invalid semantic state.');
  }
  return {
    rings: value.rings,
    stones: [...value.stones],
    toMove: value.toMove,
    movesLeft: value.movesLeft,
    opening: value.opening,
    terminal: value.terminal,
  };
}

function parseRequest(value: unknown): StarAiRequest {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      'schema',
      'version',
      'requestId',
      'rulesSchema',
      'rulesHash',
      'featureSchema',
      'featureSchemaVersion',
      'featureSchemaHash',
      'actionLayout',
      'actionLayoutVersion',
      'stateHash',
      'state',
      'actionLog',
      'legalActions',
    ]) ||
    value.schema !== STAR_AI_PROTOCOL_SCHEMA_ID ||
    value.version !== STAR_AI_PROTOCOL_VERSION ||
    !isTaskId(value.requestId) ||
    value.rulesSchema !== STAR_RULES_SCHEMA_ID ||
    value.rulesHash !== STAR_RULES_HASH ||
    value.featureSchema !== STAR_FEATURE_SCHEMA_ID ||
    value.featureSchemaVersion !== STAR_FEATURE_SCHEMA_VERSION ||
    value.featureSchemaHash !== STAR_FEATURE_SCHEMA_HASH ||
    value.actionLayout !== STAR_ACTION_LAYOUT_SCHEMA_ID ||
    value.actionLayoutVersion !== STAR_ACTION_LAYOUT_VERSION ||
    typeof value.stateHash !== 'string' ||
    !isIntegerArray(value.actionLog, 0) ||
    !isIntegerArray(value.legalActions, 0)
  ) {
    throw new StarAiError('protocol', 'Worker received an incompatible AI request.');
  }
  const state = parseSemanticState(value.state);
  if (semanticStateHash(state) !== value.stateHash) {
    throw new StarAiError('protocol', 'Worker request state hash does not match its state.');
  }
  const nodeCount = state.stones.length;
  const actionLog = [...value.actionLog];
  const legalActions = [...value.legalActions];
  if (
    state.terminal ||
    legalActions.length === 0 ||
    actionLog.some((action) => action >= nodeCount) ||
    legalActions.some(
      (action, index) =>
        action >= nodeCount ||
        state.stones[action] !== -1 ||
        (index > 0 && legalActions[index - 1] >= action),
    )
  ) {
    throw new StarAiError('protocol', 'Worker received inconsistent AI actions.');
  }
  return {
    schema: STAR_AI_PROTOCOL_SCHEMA_ID,
    version: STAR_AI_PROTOCOL_VERSION,
    requestId: value.requestId,
    rulesSchema: STAR_RULES_SCHEMA_ID,
    rulesHash: STAR_RULES_HASH,
    featureSchema: STAR_FEATURE_SCHEMA_ID,
    featureSchemaVersion: STAR_FEATURE_SCHEMA_VERSION,
    featureSchemaHash: STAR_FEATURE_SCHEMA_HASH,
    actionLayout: STAR_ACTION_LAYOUT_SCHEMA_ID,
    actionLayoutVersion: STAR_ACTION_LAYOUT_VERSION,
    stateHash: value.stateHash,
    state,
    actionLog,
    legalActions,
  };
}

export function parseWorkerCommand(value: unknown): StarAiWorkerCommand {
  if (!isRecord(value) || !isTaskId(value.taskId)) {
    throw new StarAiError('protocol', 'Worker command is invalid.');
  }
  if (value.type === 'cancel') {
    if (!hasExactKeys(value, ['type', 'taskId'])) {
      throw new StarAiError('protocol', 'Worker cancel command is invalid.');
    }
    return { type: 'cancel', taskId: value.taskId };
  }
  if (value.type === 'choose') {
    if (!hasExactKeys(value, ['type', 'taskId', 'request', 'search'])) {
      throw new StarAiError('protocol', 'Worker choose command is invalid.');
    }
    const request = parseRequest(value.request);
    if (request.requestId !== value.taskId) {
      throw new StarAiError('stale', 'Worker task identity does not match its AI request.');
    }
    return {
      type: 'choose',
      taskId: value.taskId,
      request,
      search: value.search === null ? null : parseBrowserSearchBudget(value.search),
    };
  }
  throw new StarAiError('protocol', 'Worker command type is invalid.');
}

export function parseWorkerEvent(value: unknown): StarAiWorkerEvent {
  if (!isRecord(value)) {
    throw new StarAiError('protocol', 'Local AI worker returned an invalid message.');
  }
  if (
    value.type === 'ready' &&
    value.protocolVersion === 2 &&
    hasExactKeys(value, ['type', 'protocolVersion'])
  ) {
    return { type: 'ready', protocolVersion: 2 };
  }
  if (!isTaskId(value.taskId)) {
    throw new StarAiError('protocol', 'Local AI worker returned an invalid message.');
  }
  if (value.type === 'result') {
    if (!hasExactKeys(value, ['type', 'taskId', 'decision'])) {
      throw new StarAiError('protocol', 'Local AI worker returned an invalid result.');
    }
    const decision = parseUnboundStarAiDecision(value.decision);
    if (decision.response.requestId !== value.taskId) {
      throw new StarAiError('stale', 'Local AI worker result identity is incompatible.');
    }
    return {
      type: 'result',
      taskId: value.taskId,
      decision,
    };
  }
  if (
    value.type === 'error' &&
    hasExactKeys(value, ['type', 'taskId', 'error']) &&
    isRecord(value.error) &&
    hasExactKeys(value.error, ['code', 'message', 'retryable'])
  ) {
    const code = value.error.code;
    if (
      typeof code === 'string' &&
      [
        'unavailable',
        'timeout',
        'network',
        'protocol',
        'stale',
        'illegal',
        'cancelled',
        'internal',
      ].includes(code) &&
      typeof value.error.message === 'string' &&
      typeof value.error.retryable === 'boolean'
    ) {
      return {
        type: 'error',
        taskId: value.taskId,
        error: {
          code: code as StarAiErrorCode,
          message: value.error.message,
          retryable: value.error.retryable,
        },
      };
    }
  }
  throw new StarAiError('protocol', 'Local AI worker returned an invalid event.');
}

export function workerErrorEvent(taskId: string, error: unknown): StarAiWorkerEvent {
  const aiError = asStarAiError(error);
  return {
    type: 'error',
    taskId,
    error: {
      code: aiError.code,
      message: aiError.message,
      retryable: aiError.retryable,
    },
  };
}
