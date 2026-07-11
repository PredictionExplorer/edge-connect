import { beforeEach, describe, expect, it, vi } from 'vitest';

const proxy = vi.hoisted(() => vi.fn());

vi.mock('@/lib/star/ai/server-proxy', () => ({
  STAR_AI_PROXY_ANALYZE_PATH: '/v2/analyze',
  STAR_AI_PROXY_HEALTH_PATH: '/v2/health',
  STAR_AI_PROXY_MOVE_PATH: '/v2/move',
  proxyStarAiRequest: proxy,
}));

import {
  dynamic as analyzeDynamic,
  maxDuration as analyzeMaxDuration,
  POST as POSTAnalyze,
  runtime as analyzeRuntime,
} from './analyze/route';
import {
  dynamic as healthDynamic,
  GET,
  maxDuration as healthMaxDuration,
  runtime as healthRuntime,
} from './health/route';
import {
  dynamic as moveDynamic,
  maxDuration as moveMaxDuration,
  POST as POSTMove,
  runtime as moveRuntime,
} from './move/route';

describe('same-origin AI v2 route adapters', () => {
  beforeEach(() => {
    proxy.mockReset();
  });

  it('keeps all adapters on the dynamic Node runtime', () => {
    expect([healthRuntime, moveRuntime, analyzeRuntime]).toEqual([
      'nodejs',
      'nodejs',
      'nodejs',
    ]);
    expect([healthDynamic, moveDynamic, analyzeDynamic]).toEqual([
      'force-dynamic',
      'force-dynamic',
      'force-dynamic',
    ]);
    expect(healthMaxDuration).toBeLessThan(moveMaxDuration);
    expect(analyzeMaxDuration).toBe(moveMaxDuration);
  });

  it('forwards requests only to fixed upstream paths', async () => {
    const healthRequest = new Request('https://public.example/v2/health');
    const moveRequest = new Request('https://public.example/v2/move', {
      method: 'POST',
      body: '{}',
    });
    const analyzeRequest = new Request('https://public.example/v2/analyze', {
      method: 'POST',
      body: '{}',
    });
    const responses = [
      Response.json({ status: 'ok' }),
      Response.json({ action: { code: 0 } }),
      Response.json({ action: { code: 1 } }),
    ];
    proxy
      .mockResolvedValueOnce(responses[0])
      .mockResolvedValueOnce(responses[1])
      .mockResolvedValueOnce(responses[2]);

    await expect(GET(healthRequest)).resolves.toBe(responses[0]);
    await expect(POSTMove(moveRequest)).resolves.toBe(responses[1]);
    await expect(POSTAnalyze(analyzeRequest)).resolves.toBe(responses[2]);
    expect(proxy).toHaveBeenNthCalledWith(1, healthRequest, '/v2/health');
    expect(proxy).toHaveBeenNthCalledWith(2, moveRequest, '/v2/move');
    expect(proxy).toHaveBeenNthCalledWith(3, analyzeRequest, '/v2/analyze');
  });
});
