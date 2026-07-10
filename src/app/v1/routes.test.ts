import { beforeEach, describe, expect, it, vi } from 'vitest';

const proxy = vi.hoisted(() => vi.fn());

vi.mock('@/lib/star/ai/server-proxy', () => ({
  STAR_AI_PROXY_HEALTH_PATH: '/v1/health',
  STAR_AI_PROXY_MOVE_PATH: '/v1/move',
  proxyStarAiRequest: proxy,
}));

import {
  dynamic as healthDynamic,
  GET,
  maxDuration as healthMaxDuration,
  runtime as healthRuntime,
} from './health/route';
import {
  dynamic as moveDynamic,
  maxDuration as moveMaxDuration,
  POST,
  runtime as moveRuntime,
} from './move/route';

describe('same-origin AI route adapters', () => {
  beforeEach(() => {
    proxy.mockReset();
  });

  it('keeps both adapters on the dynamic Node runtime', () => {
    expect({ healthRuntime, moveRuntime }).toEqual({
      healthRuntime: 'nodejs',
      moveRuntime: 'nodejs',
    });
    expect({ healthDynamic, moveDynamic }).toEqual({
      healthDynamic: 'force-dynamic',
      moveDynamic: 'force-dynamic',
    });
    expect(healthMaxDuration).toBeLessThan(moveMaxDuration);
  });

  it('forwards health and move requests only to their fixed upstream paths', async () => {
    const healthRequest = new Request('https://public.example/v1/health');
    const moveRequest = new Request('https://public.example/v1/move', {
      method: 'POST',
      body: '{}',
    });
    const healthResponse = Response.json({ status: 'ok' });
    const moveResponse = Response.json({ action: { code: -1 } });
    proxy.mockResolvedValueOnce(healthResponse).mockResolvedValueOnce(moveResponse);

    await expect(GET(healthRequest)).resolves.toBe(healthResponse);
    await expect(POST(moveRequest)).resolves.toBe(moveResponse);
    expect(proxy).toHaveBeenNthCalledWith(1, healthRequest, '/v1/health');
    expect(proxy).toHaveBeenNthCalledWith(2, moveRequest, '/v1/move');
  });
});

