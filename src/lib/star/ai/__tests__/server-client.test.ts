import { afterEach, describe, expect, it, vi } from 'vitest';
import type { GameConfig } from '../../game';
import { buildAiRequest, makeAiResponse } from '../protocol';
import {
  DEFAULT_SERVER_AI_MAX_CONSIDERED,
  DEFAULT_SERVER_AI_SIMULATIONS,
  configuredServerAiUrl,
  configuredServerHealthUrl,
  deterministicServerSeed,
  parseAnalyzeResponse,
  requestServerAiAction,
  resolveServerSearchBudget,
  resolveStarAiHealthUrl,
  resolveStarAiMoveUrl,
  toAnalyzeRequest,
} from '../server-client';

const config: GameConfig = {
  rings: 3,
  mode: 'double',
  pieRule: false,
  playerNames: ['A', 'B'],
};
const request = buildAiRequest(config, [], 'starserve-test');

function representativeAnalyzeResponse() {
  const score = new Array<number>(363).fill(0);
  score[181] = 1;
  return {
    schema_version: 1,
    request_id: request.requestId,
    action: { code: 0, kind: 'place', node: 0 },
    root_actions: [
      { code: 0, kind: 'place', node: 0 },
      { code: -1, kind: 'pass', node: null },
    ],
    root_policy: [0.75, 0.25],
    root_q: [0.2, -0.1],
    root_visits: [3, 1],
    wdl: { loss: 0.2, draw: 0.3, win: 0.5 },
    value: 0.3,
    search_value: 0.3,
    score_belief: {
      support_min: -181,
      support_max: 181,
      expected_margin: 0,
      probabilities: score,
    },
    model_version: 'fake-v1',
    model_step: 5,
    timing_ms: {
      queue: 0,
      model_reload: 0,
      inference_search: 1,
      total: 1,
    },
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe('starserve v1 adapter', () => {
  it('converts the internal semantic request to strict snake_case AnalyzeRequest', () => {
    const wire = toAnalyzeRequest(request, { simulations: 4, maxConsidered: 2 });
    expect(wire).toEqual({
      schema_version: 1,
      rules_hash: 'fnv1a64:cdb34fb02be82843',
      rings: 3,
      stones: new Array(30).fill(-1),
      to_move: 0,
      moves_left: 1,
      opening: true,
      pass_streak: 0,
      terminal: false,
      search: {
        simulations: 4,
        max_considered: 2,
        seed: deterministicServerSeed(request.stateHash),
      },
    });
    expect(Number.isSafeInteger(wire.search.seed)).toBe(true);
    expect(JSON.parse(JSON.stringify(wire))).toEqual(wire);
  });

  it('validates a representative AnalyzeResponse and maps its atomic action', () => {
    expect(
      parseAnalyzeResponse(
        request,
        representativeAnalyzeResponse(),
        request.requestId,
      ),
    ).toEqual(makeAiResponse(request, { type: 'place', node: 0 }));
  });

  it('rejects inconsistent or illegal starserve actions', () => {
    const inconsistent = {
      ...representativeAnalyzeResponse(),
      action: { code: 0, kind: 'pass', node: null },
    };
    expect(() => parseAnalyzeResponse(request, inconsistent)).toThrow(
      /code, kind, and node disagree/i,
    );

    const illegalAction = { code: 30, kind: 'place', node: 30 };
    const illegalBase = representativeAnalyzeResponse();
    const illegal = {
      ...illegalBase,
      action: illegalAction,
      root_actions: [illegalAction, ...illegalBase.root_actions.slice(1)],
    };
    expect(() => parseAnalyzeResponse(request, illegal)).toThrow(/illegal action/i);

    expect(() =>
      parseAnalyzeResponse(
        request,
        representativeAnalyzeResponse(),
        'different-request',
      ),
    ).toThrow(/identity/i);
  });

  it('bounds public defaults and rejects malformed explicit budgets', () => {
    expect(
      resolveServerSearchBudget(
        {},
        { simulations: '128', maxConsidered: '8' },
      ),
    ).toEqual({ simulations: 128, maxConsidered: 8 });
    expect(
      resolveServerSearchBudget(
        {},
        { simulations: '999999', maxConsidered: 'invalid' },
      ),
    ).toEqual({
      simulations: DEFAULT_SERVER_AI_SIMULATIONS,
      maxConsidered: DEFAULT_SERVER_AI_MAX_CONSIDERED,
    });
    expect(() =>
      toAnalyzeRequest(request, { simulations: 0, maxConsidered: 8 }),
    ).toThrow(/simulations/i);
    expect(() =>
      toAnalyzeRequest(request, { simulations: 8, maxConsidered: 129 }),
    ).toThrow(/max-considered/i);
  });

  it('accepts a full move endpoint or appends it to a base URL', () => {
    expect(resolveStarAiMoveUrl('https://ai.example')).toBe(
      'https://ai.example/v1/move',
    );
    expect(resolveStarAiMoveUrl('https://ai.example/proxy/')).toBe(
      'https://ai.example/proxy/v1/move',
    );
    expect(resolveStarAiMoveUrl('https://ai.example/v1/move')).toBe(
      'https://ai.example/v1/move',
    );
    expect(resolveStarAiMoveUrl('https://ai.example/v1/analyze')).toBe(
      'https://ai.example/v1/move',
    );
    expect(resolveStarAiMoveUrl('/starserve')).toBe('/starserve/v1/move');
    expect(resolveStarAiHealthUrl('https://ai.example/base')).toBe(
      'https://ai.example/base/v1/health',
    );
  });

  it('defaults browser traffic to the same-origin proxy', () => {
    vi.stubEnv('NEXT_PUBLIC_STAR_AI_URL', '');
    expect(configuredServerAiUrl()).toBe('/v1/move');
    expect(configuredServerHealthUrl()).toBe('/v1/health');
  });

  it('posts the wire request with request identity and no browser bearer secret', async () => {
    const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      expect(String(url)).toBe('https://ai.example/v1/move');
      const headers = new Headers(init?.headers);
      expect(headers.get('X-Request-ID')).toBe(request.requestId);
      expect(headers.has('Authorization')).toBe(false);
      const body = JSON.parse(String(init?.body));
      expect(body).toMatchObject({
        schema_version: 1,
        to_move: 0,
        search: { simulations: 4, max_considered: 2 },
      });
      expect(body.requestId).toBeUndefined();
      return new Response(JSON.stringify(representativeAnalyzeResponse()), {
        status: 200,
        headers: { 'X-Request-ID': request.requestId },
      });
    });
    vi.stubGlobal('fetch', fetchMock);

    await expect(
      requestServerAiAction(request, {
        url: 'https://ai.example',
        search: { simulations: 4, maxConsidered: 2 },
      }),
    ).resolves.toEqual(makeAiResponse(request, { type: 'place', node: 0 }));
    expect(fetchMock).toHaveBeenCalledOnce();
  });
});
