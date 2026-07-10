import {
  STAR_ACTION_LAYOUT_SCHEMA_ID,
  STAR_FEATURE_SCHEMA_ID,
  STAR_RULES_HASH,
  STAR_RULES_SCHEMA_ID,
} from '../rules';
import { StarAiError, asStarAiError, type StarAiErrorCode } from './errors';
import {
  STAR_ACTION_LAYOUT_VERSION,
  STAR_AI_PROTOCOL_SCHEMA_ID,
  STAR_AI_PROTOCOL_VERSION,
  STAR_FEATURE_SCHEMA_HASH,
  STAR_FEATURE_SCHEMA_VERSION,
  semanticStateHash,
  type StarAiRequest,
  type StarAiResponse,
  type StarAiSemanticState,
} from './protocol';

export type StarAiWorkerCommand =
  | { type: 'choose'; taskId: string; request: StarAiRequest }
  | { type: 'cancel'; taskId: string };

export type StarAiWorkerEvent =
  | { type: 'ready'; protocolVersion: 1 }
  | { type: 'result'; taskId: string; response: StarAiResponse }
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

function isIntegerArray(value: unknown, minimum: number): value is number[] {
  return (
    Array.isArray(value) &&
    value.every(
      (item) => typeof item === 'number' && Number.isInteger(item) && item >= minimum,
    )
  );
}

function parseSemanticState(value: unknown): StarAiSemanticState {
  if (
    !isRecord(value) ||
    typeof value.rings !== 'number' ||
    !Number.isInteger(value.rings) ||
    value.rings < 3 ||
    value.rings > 12 ||
    !isIntegerArray(value.stones, -1) ||
    value.stones.some((stone) => stone > 1) ||
    (value.toMove !== 0 && value.toMove !== 1) ||
    typeof value.movesLeft !== 'number' ||
    !Number.isInteger(value.movesLeft) ||
    value.movesLeft < 0 ||
    value.movesLeft > 2 ||
    typeof value.opening !== 'boolean' ||
    typeof value.passStreak !== 'number' ||
    !Number.isInteger(value.passStreak) ||
    value.passStreak < 0 ||
    value.passStreak > 2 ||
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
    passStreak: value.passStreak,
    terminal: value.terminal,
  };
}

function parseRequest(value: unknown): StarAiRequest {
  if (
    !isRecord(value) ||
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
    !isIntegerArray(value.actionLog, -1) ||
    !isIntegerArray(value.legalActions, -1)
  ) {
    throw new StarAiError('protocol', 'Worker received an incompatible AI request.');
  }
  const state = parseSemanticState(value.state);
  if (semanticStateHash(state) !== value.stateHash) {
    throw new StarAiError('protocol', 'Worker request state hash does not match its state.');
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
    actionLog: [...value.actionLog],
    legalActions: [...value.legalActions],
  };
}

export function parseWorkerCommand(value: unknown): StarAiWorkerCommand {
  if (!isRecord(value) || !isTaskId(value.taskId)) {
    throw new StarAiError('protocol', 'Worker command is invalid.');
  }
  if (value.type === 'cancel') return { type: 'cancel', taskId: value.taskId };
  if (value.type === 'choose') {
    return { type: 'choose', taskId: value.taskId, request: parseRequest(value.request) };
  }
  throw new StarAiError('protocol', 'Worker command type is invalid.');
}

export function parseWorkerEvent(value: unknown): StarAiWorkerEvent {
  if (!isRecord(value)) {
    throw new StarAiError('protocol', 'Local AI worker returned an invalid message.');
  }
  if (value.type === 'ready' && value.protocolVersion === 1) {
    return { type: 'ready', protocolVersion: 1 };
  }
  if (!isTaskId(value.taskId)) {
    throw new StarAiError('protocol', 'Local AI worker returned an invalid message.');
  }
  if (value.type === 'result') {
    return {
      type: 'result',
      taskId: value.taskId,
      response: value.response as StarAiResponse,
    };
  }
  if (value.type === 'error' && isRecord(value.error)) {
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
