import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  STAR_AI_PROXY_HEALTH_PATH,
  STAR_AI_PROXY_MOVE_PATH,
  STAR_AI_PROXY_REQUEST_BYTES,
  proxyStarAiRequest,
  resolveStarAiUpstreamUrl,
} from '../server-proxy';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('same-origin starserve proxy', () => {
  it('uses only the fixed private target and forwards identity/auth server-side', async () => {
    const upstreamPayload = { schema_version: 1, request_id: 'proxy-request' };
    const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      expect(String(url)).toBe('https://private.example/base/v1/move');
      expect(init?.cache).toBe('no-store');
      const headers = new Headers(init?.headers);
      expect(headers.get('X-Request-ID')).toBe('proxy-request');
      expect(headers.get('Authorization')).toBe('Bearer private-token');
      return Response.json(upstreamPayload, {
        headers: { 'X-Request-ID': 'proxy-request' },
      });
    });
    vi.stubGlobal('fetch', fetchMock);

    const response = await proxyStarAiRequest(
      new Request('https://public.example/v1/move?target=https://evil.example', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Request-ID': 'proxy-request',
        },
        body: JSON.stringify({ schema_version: 1 }),
      }),
      STAR_AI_PROXY_MOVE_PATH,
      {
        serverUrl: 'https://private.example/base',
        bearerToken: 'private-token',
      },
    );

    expect(response.status).toBe(200);
    expect(response.headers.get('Cache-Control')).toContain('no-store');
    await expect(response.json()).resolves.toEqual(upstreamPayload);
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it('proxies health to the fixed v1 endpoint', async () => {
    const fetchMock = vi.fn(async (url: string | URL | Request) => {
      expect(String(url)).toBe('https://private.example/v1/health');
      return Response.json({ status: 'ok' });
    });
    vi.stubGlobal('fetch', fetchMock);
    const response = await proxyStarAiRequest(
      new Request('https://public.example/v1/health'),
      STAR_AI_PROXY_HEALTH_PATH,
      { serverUrl: 'https://private.example/v1/move' },
    );
    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ status: 'ok' });
  });

  it('returns structured unavailable and strict input failures', async () => {
    const unavailable = await proxyStarAiRequest(
      new Request('https://public.example/v1/health'),
      STAR_AI_PROXY_HEALTH_PATH,
      { serverUrl: undefined },
    );
    expect(unavailable.status).toBe(503);
    await expect(unavailable.json()).resolves.toMatchObject({
      error: { code: 'star_ai_unavailable', retryable: false },
    });

    const wrongType = await proxyStarAiRequest(
      new Request('https://public.example/v1/move', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: '{}',
      }),
      STAR_AI_PROXY_MOVE_PATH,
      { serverUrl: 'https://private.example' },
    );
    expect(wrongType.status).toBe(415);

    const tooLarge = await proxyStarAiRequest(
      new Request('https://public.example/v1/move', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': String(STAR_AI_PROXY_REQUEST_BYTES + 1),
        },
        body: '{}',
      }),
      STAR_AI_PROXY_MOVE_PATH,
      { serverUrl: 'https://private.example' },
    );
    expect(tooLarge.status).toBe(413);
  });

  it('normalizes every documented upstream URL form with path prefixes', () => {
    const serverUrls = [
      ['base', 'https://private.example/prefix'],
      ['/v1', 'https://private.example/prefix/v1'],
      ['/v1/move', 'https://private.example/prefix/v1/move'],
      ['/v1/analyze', 'https://private.example/prefix/v1/analyze'],
      ['/v1/health', 'https://private.example/prefix/v1/health'],
      ['/healthz', 'https://private.example/prefix/healthz'],
    ] as const;

    for (const [label, serverUrl] of serverUrls) {
      expect(
        resolveStarAiUpstreamUrl(serverUrl, '/v1/move'),
        `${label} -> move`,
      ).toBe('https://private.example/prefix/v1/move');
      expect(
        resolveStarAiUpstreamUrl(serverUrl, '/v1/health'),
        `${label} -> health`,
      ).toBe('https://private.example/prefix/v1/health');
    }
  });

  it('rejects private endpoint credentials', () => {
    expect(() =>
      resolveStarAiUpstreamUrl('https://user:secret@private.example', '/v1/move'),
    ).toThrow(/without credentials/i);
  });
});
