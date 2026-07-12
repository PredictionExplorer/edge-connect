import { StarAiError } from './errors';
import {
  parseStarAiDecision,
  responseFromStarAiDecision,
  type StarAiDecision,
  type StarAiSearchBudget,
} from './decision';
import {
  codeToAction,
  makeAiResponse,
  semanticStateHash,
  type StarAiRequest,
  type StarAiResponse,
} from './protocol';

export const DEFAULT_STAR_AI_TIMEOUT_MS = 65_000;
export const DEFAULT_STAR_AI_MOVE_URL = '/v2/move' as const;
export const DEFAULT_STAR_AI_HEALTH_URL = '/v2/health' as const;
export const DEFAULT_SERVER_AI_SIMULATIONS = 4_096;
export const MAX_SERVER_AI_SIMULATIONS = 16_384;
export const DEFAULT_SERVER_AI_MAX_CONSIDERED = 32;
export const MAX_SERVER_AI_MAX_CONSIDERED = 128;

export type ServerSearchBudget = StarAiSearchBudget;

export interface AnalyzeRequestV2 {
  schema_version: 2;
  rules_hash: StarAiRequest['rulesHash'];
  rings: number;
  stones: number[];
  to_move: 0 | 1;
  moves_left: number;
  opening: boolean;
  terminal: false;
  search: {
    simulations: number;
    max_considered: number;
    seed: number;
  };
}

export interface ServerAiRequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
  url?: string;
  search?: Partial<ServerSearchBudget>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return actual.length === expected.length && actual.every((key, index) => key === expected[index]);
}

function finiteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function boundedPublicInteger(value: string | undefined, fallback: number, maximum: number): number {
  const normalized = value?.trim();
  if (!normalized || !/^[1-9][0-9]*$/.test(normalized)) return fallback;
  const parsed = Number(normalized);
  return Number.isSafeInteger(parsed) && parsed <= maximum ? parsed : fallback;
}

function strictBudgetInteger(name: string, value: unknown, maximum: number): number {
  if (
    typeof value !== 'number' ||
    !Number.isSafeInteger(value) ||
    value <= 0 ||
    value > maximum
  ) {
    throw new StarAiError('protocol', `${name} must be an integer in 1..${maximum}.`);
  }
  return value;
}

export function resolveServerSearchBudget(
  overrides: Partial<ServerSearchBudget> = {},
  publicValues: { simulations?: string; maxConsidered?: string } = {
    simulations: process.env.NEXT_PUBLIC_STAR_AI_SIMULATIONS,
    maxConsidered: process.env.NEXT_PUBLIC_STAR_AI_MAX_CONSIDERED,
  },
): ServerSearchBudget {
  const configured = {
    simulations: boundedPublicInteger(
      publicValues.simulations,
      DEFAULT_SERVER_AI_SIMULATIONS,
      MAX_SERVER_AI_SIMULATIONS,
    ),
    maxConsidered: boundedPublicInteger(
      publicValues.maxConsidered,
      DEFAULT_SERVER_AI_MAX_CONSIDERED,
      MAX_SERVER_AI_MAX_CONSIDERED,
    ),
  };
  return {
    simulations:
      overrides.simulations === undefined
        ? configured.simulations
        : strictBudgetInteger(
            'Server AI simulations',
            overrides.simulations,
            MAX_SERVER_AI_SIMULATIONS,
          ),
    maxConsidered:
      overrides.maxConsidered === undefined
        ? configured.maxConsidered
        : strictBudgetInteger(
            'Server AI max-considered',
            overrides.maxConsidered,
            MAX_SERVER_AI_MAX_CONSIDERED,
          ),
  };
}

export function deterministicServerSeed(stateHash: string): number {
  const match = /^zobrist64:([0-9a-f]{16})$/.exec(stateHash);
  if (!match) throw new StarAiError('protocol', 'AI state hash cannot seed server search.');
  return Number(BigInt(`0x${match[1]}`) & BigInt(Number.MAX_SAFE_INTEGER));
}

export function toAnalyzeRequest(
  request: StarAiRequest,
  search: ServerSearchBudget = resolveServerSearchBudget(),
): AnalyzeRequestV2 {
  const simulations = strictBudgetInteger(
    'Server AI simulations',
    search.simulations,
    MAX_SERVER_AI_SIMULATIONS,
  );
  const maxConsidered = strictBudgetInteger(
    'Server AI max-considered',
    search.maxConsidered,
    MAX_SERVER_AI_MAX_CONSIDERED,
  );
  if (request.state.terminal) {
    throw new StarAiError('protocol', 'Starserve accepts only active positions.');
  }
  if (semanticStateHash(request.state) !== request.stateHash) {
    throw new StarAiError('protocol', 'Internal AI state hash is inconsistent.');
  }
  return {
    schema_version: 2,
    rules_hash: request.rulesHash,
    rings: request.state.rings,
    stones: [...request.state.stones],
    to_move: request.state.toMove,
    moves_left: request.state.movesLeft,
    opening: request.state.opening,
    terminal: false,
    search: {
      simulations,
      max_considered: maxConsidered,
      seed: deterministicServerSeed(request.stateHash),
    },
  };
}

function endpointPath(pathname: string): string {
  const normalized = pathname.replace(/\/+$/, '');
  if (normalized.endsWith('/v2/move')) return normalized || '/v2/move';
  if (normalized.endsWith('/v2/analyze')) {
    return `${normalized.slice(0, -'/analyze'.length)}/move`;
  }
  if (normalized.endsWith('/v2/health')) {
    return `${normalized.slice(0, -'/health'.length)}/move`;
  }
  if (normalized.endsWith('/v2')) return `${normalized}/move`;
  return `${normalized}/v2/move` || '/v2/move';
}

/**
 * NEXT_PUBLIC_STAR_AI_URL may be the full /v2/move URL or a deployment base.
 * Browser clients intentionally never read or send bearer-token environment values.
 */
export function resolveStarAiMoveUrl(value: string): string {
  const normalized = value.trim();
  if (!normalized) throw new StarAiError('unavailable', 'Server AI URL is empty.');
  if (
    /(?:^|\/)v\d+(?:\/|$)/.test(normalized) &&
    !/(?:^|\/)v2(?:\/|$)/.test(normalized)
  ) {
    throw new StarAiError('protocol', 'Server AI URL must use the v2 API.');
  }
  if (normalized.startsWith('/')) {
    if (normalized.includes('?') || normalized.includes('#')) {
      throw new StarAiError('protocol', 'Server AI URL must not contain a query or fragment.');
    }
    return endpointPath(normalized);
  }

  let url: URL;
  try {
    url = new URL(normalized);
  } catch (error) {
    throw new StarAiError('protocol', 'Server AI URL must be absolute or root-relative.', false, error);
  }
  if (
    (url.protocol !== 'http:' && url.protocol !== 'https:') ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  ) {
    throw new StarAiError(
      'protocol',
      'Server AI URL must be an HTTP(S) base without credentials, query, or fragment.',
    );
  }
  url.pathname = endpointPath(url.pathname);
  return url.toString();
}

export function resolveStarAiHealthUrl(value: string): string {
  const moveUrl = resolveStarAiMoveUrl(value);
  if (moveUrl.startsWith('/')) {
    return moveUrl.replace(/\/v2\/move$/, '/v2/health');
  }
  const url = new URL(moveUrl);
  url.pathname = url.pathname.replace(/\/v2\/move$/, '/v2/health');
  return url.toString();
}

export function configuredServerAiUrl(): string {
  const value = process.env.NEXT_PUBLIC_STAR_AI_URL;
  return value?.trim() ? resolveStarAiMoveUrl(value) : DEFAULT_STAR_AI_MOVE_URL;
}

export function configuredServerHealthUrl(): string {
  const value = process.env.NEXT_PUBLIC_STAR_AI_URL;
  return value?.trim() ? resolveStarAiHealthUrl(value) : DEFAULT_STAR_AI_HEALTH_URL;
}

interface ParsedAtomicAction {
  code: number;
  kind: 'place';
  node: number;
}

function parseAtomicAction(value: unknown, label: string): ParsedAtomicAction {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ['code', 'kind', 'node']) ||
    typeof value.code !== 'number' ||
    !Number.isInteger(value.code)
  ) {
    throw new StarAiError('protocol', `${label} is not a valid atomic action.`);
  }
  if (
    value.kind === 'place' &&
    value.code >= 0 &&
    typeof value.node === 'number' &&
    Number.isInteger(value.node) &&
    value.node === value.code
  ) {
    return { code: value.code, kind: 'place', node: value.node };
  }
  throw new StarAiError('protocol', `${label} code, kind, and node disagree.`);
}

function parseFiniteArray(
  value: unknown,
  label: string,
  length: number,
  predicate: (item: number) => boolean = () => true,
): number[] {
  if (
    !Array.isArray(value) ||
    value.length !== length ||
    value.some((item) => !finiteNumber(item) || !predicate(item))
  ) {
    throw new StarAiError('protocol', `${label} is invalid.`);
  }
  return value as number[];
}

function approximatelyOne(values: readonly number[], tolerance: number): boolean {
  return Math.abs(values.reduce((total, value) => total + value, 0) - 1) <= tolerance;
}

export function parseAnalyzeResponse(
  request: StarAiRequest,
  payload: unknown,
  headerRequestId?: string | null,
  search?: ServerSearchBudget,
): StarAiDecision {
  const responseKeys = [
    'schema_version',
    'request_id',
    'action',
    'root_actions',
    'root_policy',
    'root_q',
    'root_visits',
    'outcome',
    'value',
    'search_value',
    'score_belief',
    'model_version',
    'model_step',
    'timing_ms',
  ] as const;
  if (!isRecord(payload) || !hasExactKeys(payload, responseKeys) || payload.schema_version !== 2) {
    throw new StarAiError('protocol', 'Starserve response schema is incompatible.');
  }
  if (
    payload.request_id !== request.requestId ||
    (headerRequestId !== undefined &&
      headerRequestId !== null &&
      headerRequestId !== request.requestId)
  ) {
    throw new StarAiError('stale', 'Starserve response identity is incompatible.');
  }

  const action = parseAtomicAction(payload.action, 'Starserve action');
  if (!request.legalActions.includes(action.code)) {
    throw new StarAiError('illegal', 'Starserve selected an illegal action.');
  }
  if (!Array.isArray(payload.root_actions) || payload.root_actions.length === 0) {
    throw new StarAiError('protocol', 'Starserve root actions are invalid.');
  }
  const rootActions = payload.root_actions.map((item, index) =>
    parseAtomicAction(item, `Starserve root action ${index}`),
  );
  const rootCodes = rootActions.map((item) => item.code);
  if (
    new Set(rootCodes).size !== rootCodes.length ||
    rootCodes.some((code) => !request.legalActions.includes(code)) ||
    !rootCodes.includes(action.code)
  ) {
    throw new StarAiError('protocol', 'Starserve root action identity is inconsistent.');
  }

  const rootPolicy = parseFiniteArray(
    payload.root_policy,
    'Starserve root policy',
    rootActions.length,
    (item) => item >= 0,
  );
  const rootQ = parseFiniteArray(
    payload.root_q,
    'Starserve root Q values',
    rootActions.length,
    (item) => item >= -1 && item <= 1,
  );
  if (
    !Array.isArray(payload.root_visits) ||
    payload.root_visits.length !== rootActions.length ||
    payload.root_visits.some(
      (item) => typeof item !== 'number' || !Number.isInteger(item) || item < 0,
    ) ||
    !approximatelyOne(rootPolicy, 1e-4)
  ) {
    throw new StarAiError('protocol', 'Starserve root statistics are invalid.');
  }
  const rootVisits = [...payload.root_visits] as number[];

  if (
    !isRecord(payload.outcome) ||
    !hasExactKeys(payload.outcome, ['loss', 'win'])
  ) {
    throw new StarAiError('protocol', 'Starserve outcome belief is invalid.');
  }
  const outcome = [payload.outcome.loss, payload.outcome.win];
  if (
    outcome.some((item) => !finiteNumber(item) || item < 0) ||
    !approximatelyOne(outcome as number[], 1e-5) ||
    !finiteNumber(payload.value) ||
    payload.value < -1 ||
    payload.value > 1 ||
    Math.abs(
      payload.value -
        ((payload.outcome.win as number) - (payload.outcome.loss as number)),
    ) > 1e-5 ||
    !finiteNumber(payload.search_value) ||
    payload.search_value < -1 ||
    payload.search_value > 1
  ) {
    throw new StarAiError('protocol', 'Starserve value belief is invalid.');
  }

  if (
    !isRecord(payload.score_belief) ||
    !hasExactKeys(payload.score_belief, [
      'support_min',
      'support_max',
      'expected_margin',
      'probabilities',
    ]) ||
    payload.score_belief.support_min !== -151 ||
    payload.score_belief.support_max !== 151 ||
    !finiteNumber(payload.score_belief.expected_margin) ||
    payload.score_belief.expected_margin < -151 ||
    payload.score_belief.expected_margin > 151
  ) {
    throw new StarAiError('protocol', 'Starserve score belief is invalid.');
  }
  const scoreProbabilities = parseFiniteArray(
    payload.score_belief.probabilities,
    'Starserve score probabilities',
    303,
    (item) => item >= 0,
  );
  if (!approximatelyOne(scoreProbabilities, 1e-5)) {
    throw new StarAiError('protocol', 'Starserve score probabilities are not normalized.');
  }
  const expectedMargin = scoreProbabilities.reduce(
    (total, probability, index) => total + probability * (index - 151),
    0,
  );
  if (
    Math.abs(expectedMargin - (payload.score_belief.expected_margin as number)) >
    1e-3
  ) {
    throw new StarAiError('protocol', 'Starserve expected score margin is inconsistent.');
  }

  if (
    typeof payload.model_version !== 'string' ||
    payload.model_version.length === 0 ||
    typeof payload.model_step !== 'number' ||
    !Number.isInteger(payload.model_step) ||
    payload.model_step < 0 ||
    !isRecord(payload.timing_ms) ||
    !hasExactKeys(payload.timing_ms, ['queue', 'model_reload', 'inference_search', 'total']) ||
    [payload.timing_ms.queue, payload.timing_ms.model_reload, payload.timing_ms.inference_search, payload.timing_ms.total].some(
      (item) => !finiteNumber(item) || item < 0,
    )
  ) {
    throw new StarAiError('protocol', 'Starserve model or timing metadata is invalid.');
  }

  const effectiveSearch = search ?? {
    simulations: rootVisits.reduce((total, visits) => total + visits, 0),
    maxConsidered: rootActions.length,
  };
  strictBudgetInteger(
    'Server AI simulations',
    effectiveSearch.simulations,
    MAX_SERVER_AI_SIMULATIONS,
  );
  strictBudgetInteger(
    'Server AI max-considered',
    effectiveSearch.maxConsidered,
    MAX_SERVER_AI_MAX_CONSIDERED,
  );
  const response = makeAiResponse(request, codeToAction(action.code));
  return parseStarAiDecision(request, {
    response,
    analysis: {
      perspective: request.state.toMove,
      stateHash: request.stateHash,
      outcome: {
        loss: payload.outcome.loss,
        win: payload.outcome.win,
      },
      modelValue: payload.value,
      searchValue: payload.search_value,
      expectedMargin: payload.score_belief.expected_margin,
      rootActions: rootActions.map((rootAction) => codeToAction(rootAction.code)),
      rootPolicy,
      rootQ,
      rootVisits,
      modelVersion: payload.model_version,
      modelStep: payload.model_step,
      modelIdentity: null,
      simulations: effectiveSearch.simulations,
      maxConsidered: effectiveSearch.maxConsidered,
      timingMs: {
        queue: payload.timing_ms.queue,
        modelLoad: payload.timing_ms.model_reload,
        inferenceSearch: payload.timing_ms.inference_search,
        total: payload.timing_ms.total,
      },
    },
  });
}

export async function requestServerAiDecision(
  request: StarAiRequest,
  options: ServerAiRequestOptions = {},
): Promise<StarAiDecision> {
  const url = options.url?.trim()
    ? resolveStarAiMoveUrl(options.url)
    : configuredServerAiUrl();

  const timeoutMs = options.timeoutMs ?? DEFAULT_STAR_AI_TIMEOUT_MS;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new StarAiError('protocol', 'AI timeout must be a positive number.');
  }
  if (options.signal?.aborted) {
    throw new StarAiError('cancelled', 'AI request cancelled.');
  }
  if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(request.requestId)) {
    throw new StarAiError('protocol', 'AI request id is not accepted by starserve.');
  }
  const search = resolveServerSearchBudget(options.search);
  const analyzeRequest = toAnalyzeRequest(request, search);

  const controller = new AbortController();
  let timedOut = false;
  const abortFromCaller = () => controller.abort(options.signal?.reason);
  options.signal?.addEventListener('abort', abortFromCaller, { once: true });
  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  try {
    const response = await fetch(url, {
      method: 'POST',
      cache: 'no-store',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
        'X-Request-ID': request.requestId,
      },
      body: JSON.stringify(analyzeRequest),
      signal: controller.signal,
    });
    let payload: unknown;
    try {
      payload = await response.json();
    } catch (error) {
      throw new StarAiError('protocol', 'Server AI returned invalid JSON.', false, error);
    }
    if (!response.ok) {
      if (isRecord(payload) && isRecord(payload.error)) {
        const message =
          typeof payload.error.message === 'string'
            ? payload.error.message
            : `Server AI returned HTTP ${response.status}.`;
        const retryable =
          typeof payload.error.retryable === 'boolean'
            ? payload.error.retryable
            : response.status >= 500 || response.status === 429;
        const upstreamCode = payload.error.code;
        const code =
          upstreamCode === 'star_ai_timeout'
            ? 'timeout'
            : upstreamCode === 'star_ai_unavailable' || response.status >= 500
              ? 'unavailable'
              : response.status === 429
                ? 'network'
                : 'protocol';
        throw new StarAiError(code, message, retryable);
      }
      throw new StarAiError(
        response.status >= 500 ? 'unavailable' : 'protocol',
        `Server AI returned HTTP ${response.status}.`,
        response.status >= 500 || response.status === 429,
      );
    }
    return parseAnalyzeResponse(
      request,
      payload,
      response.headers.get('X-Request-ID'),
      search,
    );
  } catch (error) {
    if (error instanceof StarAiError) throw error;
    if (options.signal?.aborted) {
      throw new StarAiError('cancelled', 'AI request cancelled.', false, error);
    }
    if (timedOut) {
      throw new StarAiError('timeout', 'Server AI timed out.', true, error);
    }
    throw new StarAiError('network', 'Could not reach Server AI.', true, error);
  } finally {
    clearTimeout(timeout);
    options.signal?.removeEventListener('abort', abortFromCaller);
  }
}

export async function requestServerAiAction(
  request: StarAiRequest,
  options: ServerAiRequestOptions = {},
): Promise<StarAiResponse> {
  return responseFromStarAiDecision(await requestServerAiDecision(request, options));
}
