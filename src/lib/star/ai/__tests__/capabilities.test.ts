import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  checkLocalAiCapability,
  checkServerAiCapability,
  localBrowserCapabilityIssue,
} from '../capabilities';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('AI capability preflight', () => {
  it('validates the server health contract through the same-origin route', async () => {
    const fetchMock = vi.fn(async (url: string | URL | Request) => {
      expect(String(url)).toBe('/v1/health');
      return Response.json({
        status: 'ok',
        service_version: '1.0.0',
        api_schema_version: 1,
        model: { ready: true, model_version: 'test', model_step: 1 },
        rules: {
          schema_id: 'edgeconnect.star.rules.v1',
          version: 1,
          hash: 'fnv1a64:cdb34fb02be82843',
        },
        features: {
          schema_id: 'edgeconnect.star.model-features.external.v1',
          version: 2,
          hash: '59a7da1c00bac4d2',
        },
        actions: {
          schema_id: 'edgeconnect.star.action-layout.nodes-then-pass.v1',
        },
      });
    });
    vi.stubGlobal('fetch', fetchMock);
    await expect(checkServerAiCapability()).resolves.toEqual({
      status: 'available',
      label: 'Server AI',
    });
  });

  it('reports incompatible server health as permanently unavailable', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        Response.json({
          status: 'ok',
          api_schema_version: 1,
          model: { ready: true },
          rules: {
            schema_id: 'edgeconnect.star.rules.v1',
            hash: 'fnv1a64:wrong',
          },
          features: {},
          actions: {},
        }),
      ),
    );
    await expect(checkServerAiCapability()).resolves.toMatchObject({
      status: 'unavailable',
      code: 'server_incompatible',
      retryable: false,
    });
  });

  it('guards local AI before loading BigInt-dependent modules', async () => {
    expect(localBrowserCapabilityIssue()).toMatch(/Web Workers/i);

    vi.stubGlobal('Worker', class {});
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('{}', { status: 404 })),
    );
    await expect(checkLocalAiCapability()).resolves.toMatchObject({
      status: 'unavailable',
      code: 'local_assets_missing',
      retryable: false,
    });
  });
});
