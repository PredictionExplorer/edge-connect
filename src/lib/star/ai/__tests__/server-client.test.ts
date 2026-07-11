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
  rings: 4,
  mode: 'double',
  pieRule: false,
  playerNames: ['A', 'B'],
};
const request = buildAiRequest(config, [], 'starserve-test');

function representativeAnalyzeResponse() {
  const score = new Array<number>(303).fill(0);
  score[151] = 1;
  return {
    schema_version: 2,
    request_id: request.requestId,
    action: { code: 0, kind: 'place', node: 0 },
    root_actions: [
      { code: 0, kind: 'place', node: 0 },
      { code: 1, kind: 'place', node: 1 },
    ],
    root_policy: [0.75, 0.25],
    root_q: [0.2, -0.1],
    root_visits: [3, 1],
    outcome: { loss: 0.2, win: 0.8 },
    value: 0.6,
    search_value: 0.3,
    score_belief: {
      support_min: -151,
      support_max: 151,
      expected_margin: 0,
      probabilities: score,
    },
    model_version: 'fake-v2',
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

describe('starserve v2 adapter', () => {
  it('converts semantic state to strict placement-only snake_case', () => {
    const wire = toAnalyzeRequest(request, {
      simulations: 4,
      maxConsidered: 2,
    });
    expect(wire).toEqual({
      schema_version: 2,
      rules_hash: 'fnv1a64:2da3783519381453',
      rings: 4,
      stones: new Array(50).fill(-1),
      to_move: 0,
      moves_left: 1,
      opening: true,
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

  it('validates binary outcomes and maps a placement response', () => {
    expect(
      parseAnalyzeResponse(
        request,
        representativeAnalyzeResponse(),
        request.requestId,
      ),
    ).toEqual(makeAiResponse(request, { type: 'place', node: 0 }));
  });

  it('rejects removed action and outcome shapes', () => {
    expect(() =>
      parseAnalyzeResponse(request, {
        ...representativeAnalyzeResponse(),
        action: { code: -1, kind: 'pass', node: null },
      }),
    ).toThrow(/valid atomic action|disagree/i);

    const response = representativeAnalyzeResponse();
    expect(() =>
      parseAnalyzeResponse(request, {
        ...response,
        outcome: { ...response.outcome, draw: 0 },
      }),
    ).toThrow(/outcome belief/i);
    expect(() =>
      parseAnalyzeResponse(request, {
        ...response,
        value: 0,
      }),
    ).toThrow(/value belief/i);
  });

  it('rejects inconsistent, illegal, or stale actions', () => {
    const inconsistent = {
      ...representativeAnalyzeResponse(),
      action: { code: 0, kind: 'place', node: 1 },
    };
    expect(() => parseAnalyzeResponse(request, inconsistent)).toThrow(
      /code, kind, and node disagree/i,
    );

    const illegalAction = { code: 50, kind: 'place', node: 50 };
    const illegal = {
      ...representativeAnalyzeResponse(),
      action: illegalAction,
      root_actions: [illegalAction],
      root_policy: [1],
      root_q: [0],
      root_visits: [1],
    };
    expect(() => parseAnalyzeResponse(request, illegal)).toThrow(
      /illegal action/i,
    );
    expect(() =>
      parseAnalyzeResponse(
        request,
        representativeAnalyzeResponse(),
        'different-request',
      ),
    ).toThrow(/identity/i);
  });

  it('bounds defaults and rejects malformed explicit budgets', () => {
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

  it('normalizes v2 move, analyze, base, and health URLs', () => {
    expect(resolveStarAiMoveUrl('https://ai.example')).toBe(
      'https://ai.example/v2/move',
    );
    expect(resolveStarAiMoveUrl('https://ai.example/proxy/')).toBe(
      'https://ai.example/proxy/v2/move',
    );
    expect(resolveStarAiMoveUrl('https://ai.example/v2/move')).toBe(
      'https://ai.example/v2/move',
    );
    expect(resolveStarAiMoveUrl('https://ai.example/v2/analyze')).toBe(
      'https://ai.example/v2/move',
    );
    expect(resolveStarAiMoveUrl('https://ai.example/v2/health')).toBe(
      'https://ai.example/v2/move',
    );
    expect(resolveStarAiMoveUrl('/starserve')).toBe('/starserve/v2/move');
    expect(resolveStarAiHealthUrl('https://ai.example/base')).toBe(
      'https://ai.example/base/v2/health',
    );
    expect(() => resolveStarAiMoveUrl('https://ai.example/v1/move')).toThrow(
      /v2 API/,
    );
  });

  it('defaults browser traffic to the same-origin v2 proxy', () => {
    vi.stubEnv('NEXT_PUBLIC_STAR_AI_URL', '');
    expect(configuredServerAiUrl()).toBe('/v2/move');
    expect(configuredServerHealthUrl()).toBe('/v2/health');
  });

  it('posts schema v2 with request identity and no browser bearer secret', async () => {
    const fetchMock = vi.fn(
      async (url: string | URL | Request, init?: RequestInit) => {
        expect(String(url)).toBe('https://ai.example/v2/move');
        const headers = new Headers(init?.headers);
        expect(headers.get('X-Request-ID')).toBe(request.requestId);
        expect(headers.has('Authorization')).toBe(false);
        const body = JSON.parse(String(init?.body));
        expect(body).toMatchObject({
          schema_version: 2,
          to_move: 0,
          search: { simulations: 4, max_considered: 2 },
        });
        expect(body.pass_streak).toBeUndefined();
        return new Response(JSON.stringify(representativeAnalyzeResponse()), {
          status: 200,
          headers: { 'X-Request-ID': request.requestId },
        });
      },
    );
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
