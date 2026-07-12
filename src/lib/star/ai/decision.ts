import { STAR_RULES_HASH } from '../rules';
import { StarAiError } from './errors';
import {
  STAR_AI_PROTOCOL_SCHEMA_ID,
  STAR_AI_PROTOCOL_VERSION,
  parseAiResponse,
  type AtomicGameAction,
  type StarAiRequest,
  type StarAiResponse,
} from './protocol';

export const STAR_SCORE_MARGIN_MIN = -151;
export const STAR_SCORE_MARGIN_MAX = 151;

export interface StarAiSearchBudget {
  simulations: number;
  maxConsidered: number;
}

export interface StarAiOutcomeBelief {
  loss: number;
  win: number;
}

export interface StarAiTiming {
  queue: number;
  modelLoad: number;
  inferenceSearch: number;
  total: number;
}

/**
 * Compact, runtime-neutral analysis from the perspective of the player who
 * was to move in `stateHash`. Arrays share one stable root-action order.
 */
export interface StarAiAnalysis {
  perspective: 0 | 1;
  stateHash: string;
  outcome: StarAiOutcomeBelief;
  modelValue: number;
  searchValue: number;
  expectedMargin: number;
  rootActions: AtomicGameAction[];
  rootPolicy: number[];
  rootQ: number[];
  rootVisits: number[];
  modelVersion: string;
  modelStep: number | null;
  modelIdentity: string | null;
  simulations: number;
  maxConsidered: number;
  timingMs: StarAiTiming;
}

export interface StarAiDecision {
  response: StarAiResponse;
  analysis: StarAiAnalysis;
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

function finiteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function approximatelyEqual(left: number, right: number, tolerance: number): boolean {
  return Math.abs(left - right) <= tolerance;
}

function parseAtomicAction(value: unknown, label: string): AtomicGameAction {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ['type', 'node']) ||
    value.type !== 'place' ||
    typeof value.node !== 'number' ||
    !Number.isSafeInteger(value.node) ||
    value.node < 0
  ) {
    throw new StarAiError('protocol', `${label} is not a valid atomic action.`);
  }
  return { type: 'place', node: value.node };
}

function parseUnboundResponse(value: unknown): StarAiResponse {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      'schema',
      'version',
      'requestId',
      'rulesHash',
      'stateHash',
      'action',
    ]) ||
    value.schema !== STAR_AI_PROTOCOL_SCHEMA_ID ||
    value.version !== STAR_AI_PROTOCOL_VERSION ||
    value.rulesHash !== STAR_RULES_HASH ||
    typeof value.requestId !== 'string' ||
    value.requestId.length === 0 ||
    value.requestId.length > 200 ||
    typeof value.stateHash !== 'string' ||
    !/^zobrist64:[0-9a-f]{16}$/.test(value.stateHash)
  ) {
    throw new StarAiError('protocol', 'AI decision response is invalid.');
  }
  return {
    schema: STAR_AI_PROTOCOL_SCHEMA_ID,
    version: STAR_AI_PROTOCOL_VERSION,
    requestId: value.requestId,
    rulesHash: STAR_RULES_HASH,
    stateHash: value.stateHash,
    action: parseAtomicAction(value.action, 'AI decision action'),
  };
}

function parseFiniteArray(
  value: unknown,
  label: string,
  length: number,
  predicate: (item: number) => boolean,
): number[] {
  if (
    !Array.isArray(value) ||
    value.length !== length ||
    value.some((item) => !finiteNumber(item) || !predicate(item))
  ) {
    throw new StarAiError('protocol', `${label} is invalid.`);
  }
  return [...value] as number[];
}

function parseTiming(value: unknown): StarAiTiming {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ['queue', 'modelLoad', 'inferenceSearch', 'total'])
  ) {
    throw new StarAiError('protocol', 'AI decision timing is invalid.');
  }
  const values = [value.queue, value.modelLoad, value.inferenceSearch, value.total];
  if (values.some((item) => !finiteNumber(item) || item < 0)) {
    throw new StarAiError('protocol', 'AI decision timing is invalid.');
  }
  const queue = value.queue as number;
  const modelLoad = value.modelLoad as number;
  const inferenceSearch = value.inferenceSearch as number;
  const total = value.total as number;
  if (
    total + 1e-6 < queue ||
    total + 1e-6 < modelLoad ||
    total + 1e-6 < inferenceSearch
  ) {
    throw new StarAiError('protocol', 'AI decision timing total is inconsistent.');
  }
  return { queue, modelLoad, inferenceSearch, total };
}

function parseAnalysis(value: unknown, response: StarAiResponse): StarAiAnalysis {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      'perspective',
      'stateHash',
      'outcome',
      'modelValue',
      'searchValue',
      'expectedMargin',
      'rootActions',
      'rootPolicy',
      'rootQ',
      'rootVisits',
      'modelVersion',
      'modelStep',
      'modelIdentity',
      'simulations',
      'maxConsidered',
      'timingMs',
    ]) ||
    (value.perspective !== 0 && value.perspective !== 1) ||
    value.stateHash !== response.stateHash
  ) {
    throw new StarAiError('protocol', 'AI decision analysis identity is invalid.');
  }

  if (
    !isRecord(value.outcome) ||
    !hasExactKeys(value.outcome, ['loss', 'win']) ||
    !finiteNumber(value.outcome.loss) ||
    !finiteNumber(value.outcome.win) ||
    value.outcome.loss < 0 ||
    value.outcome.loss > 1 ||
    value.outcome.win < 0 ||
    value.outcome.win > 1 ||
    !approximatelyEqual(value.outcome.loss + value.outcome.win, 1, 1e-5)
  ) {
    throw new StarAiError('protocol', 'AI decision outcome belief is invalid.');
  }
  const outcome = { loss: value.outcome.loss, win: value.outcome.win };

  if (
    !finiteNumber(value.modelValue) ||
    value.modelValue < -1 ||
    value.modelValue > 1 ||
    !approximatelyEqual(value.modelValue, outcome.win - outcome.loss, 1e-5) ||
    !finiteNumber(value.searchValue) ||
    value.searchValue < -1 ||
    value.searchValue > 1 ||
    !finiteNumber(value.expectedMargin) ||
    value.expectedMargin < STAR_SCORE_MARGIN_MIN ||
    value.expectedMargin > STAR_SCORE_MARGIN_MAX
  ) {
    throw new StarAiError('protocol', 'AI decision value belief is invalid.');
  }

  if (!Array.isArray(value.rootActions) || value.rootActions.length === 0) {
    throw new StarAiError('protocol', 'AI decision root actions are invalid.');
  }
  const rootActions = value.rootActions.map((action, index) =>
    parseAtomicAction(action, `AI decision root action ${index}`),
  );
  const rootNodes = rootActions.map((action) => action.node);
  if (
    new Set(rootNodes).size !== rootNodes.length ||
    !rootNodes.includes(response.action.node)
  ) {
    throw new StarAiError('protocol', 'AI decision root action identity is invalid.');
  }

  const rootPolicy = parseFiniteArray(
    value.rootPolicy,
    'AI decision root policy',
    rootActions.length,
    (item) => item >= 0 && item <= 1,
  );
  const rootQ = parseFiniteArray(
    value.rootQ,
    'AI decision root Q values',
    rootActions.length,
    (item) => item >= -1 && item <= 1,
  );
  if (
    !Array.isArray(value.rootVisits) ||
    value.rootVisits.length !== rootActions.length ||
    value.rootVisits.some(
      (item) =>
        typeof item !== 'number' ||
        !Number.isSafeInteger(item) ||
        item < 0,
    )
  ) {
    throw new StarAiError('protocol', 'AI decision root visits are invalid.');
  }
  const rootVisits = [...value.rootVisits] as number[];
  if (
    !approximatelyEqual(
      rootPolicy.reduce((total, item) => total + item, 0),
      1,
      1e-4,
    )
  ) {
    throw new StarAiError('protocol', 'AI decision root policy is not normalized.');
  }

  if (
    typeof value.simulations !== 'number' ||
    !Number.isSafeInteger(value.simulations) ||
    value.simulations <= 0 ||
    typeof value.maxConsidered !== 'number' ||
    !Number.isSafeInteger(value.maxConsidered) ||
    value.maxConsidered <= 0
  ) {
    throw new StarAiError('protocol', 'AI decision search budget is invalid.');
  }
  const totalVisits = rootVisits.reduce((total, visits) => total + visits, 0);
  if (!Number.isSafeInteger(totalVisits) || totalVisits !== value.simulations) {
    throw new StarAiError('protocol', 'AI decision visits do not match its search budget.');
  }

  if (
    typeof value.modelVersion !== 'string' ||
    value.modelVersion.length === 0 ||
    value.modelVersion.length > 256 ||
    (value.modelStep !== null &&
      (typeof value.modelStep !== 'number' ||
        !Number.isSafeInteger(value.modelStep) ||
        value.modelStep < 0)) ||
    (value.modelIdentity !== null &&
      (typeof value.modelIdentity !== 'string' ||
        value.modelIdentity.length === 0 ||
        value.modelIdentity.length > 256))
  ) {
    throw new StarAiError('protocol', 'AI decision model identity is invalid.');
  }

  return {
    perspective: value.perspective,
    stateHash: response.stateHash,
    outcome,
    modelValue: value.modelValue,
    searchValue: value.searchValue,
    expectedMargin: value.expectedMargin,
    rootActions,
    rootPolicy,
    rootQ,
    rootVisits,
    modelVersion: value.modelVersion,
    modelStep: value.modelStep,
    modelIdentity: value.modelIdentity,
    simulations: value.simulations,
    maxConsidered: value.maxConsidered,
    timingMs: parseTiming(value.timingMs),
  };
}

/** Strict structural validation for messages before a pending request is known. */
export function parseUnboundStarAiDecision(payload: unknown): StarAiDecision {
  if (!isRecord(payload) || !hasExactKeys(payload, ['response', 'analysis'])) {
    throw new StarAiError('protocol', 'AI decision must contain response and analysis.');
  }
  const response = parseUnboundResponse(payload.response);
  return {
    response,
    analysis: parseAnalysis(payload.analysis, response),
  };
}

/**
 * Binds a structurally valid decision to its originating request, including
 * stale identity, selected-action legality, root-action legality, and
 * perspective checks.
 */
export function parseStarAiDecision(
  request: StarAiRequest,
  payload: unknown,
): StarAiDecision {
  const decision = parseUnboundStarAiDecision(payload);
  const response = parseAiResponse(request, decision.response);
  if (
    decision.analysis.perspective !== request.state.toMove ||
    decision.analysis.stateHash !== request.stateHash
  ) {
    throw new StarAiError('stale', 'AI decision belongs to an obsolete position.');
  }
  if (
    decision.analysis.rootActions.some(
      (action) => !request.legalActions.includes(action.node),
    )
  ) {
    throw new StarAiError('protocol', 'AI decision contains an illegal root action.');
  }
  return { response, analysis: decision.analysis };
}

/** Compatibility adapter for mutation paths that still consume one response. */
export function responseFromStarAiDecision(decision: StarAiDecision): StarAiResponse {
  return decision.response;
}
