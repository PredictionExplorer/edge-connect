import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  checkLocalAiCapability,
  checkServerAiCapability,
  localBrowserCapabilityIssue,
} from '../capabilities';
import { STAR_FEATURE_SCHEMA_HASH } from '../protocol';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('AI capability preflight', () => {
  it('validates the server health contract through the same-origin route', async () => {
    const fetchMock = vi.fn(async (url: string | URL | Request) => {
      expect(String(url)).toBe('/v2/health');
      return Response.json({
        status: 'ok',
        service_version: '1.0.0',
        api_schema_version: 2,
        model: { ready: true, model_version: 'test', model_step: 1 },
        rules: {
          schema_id: 'edgeconnect.star.rules.v2',
          version: 2,
          hash: 'fnv1a64:2da3783519381453',
        },
        features: {
          schema_id: 'edgeconnect.star.model-features.external.v2',
          version: 3,
          hash: STAR_FEATURE_SCHEMA_HASH,
        },
        actions: {
          schema_id: 'edgeconnect.star.action-layout.nodes-only.v1',
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
          api_schema_version: 2,
          model: { ready: true },
          rules: {
            schema_id: 'edgeconnect.star.rules.v2',
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

  it('preserves optional server device, champion, and actual search metadata', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        Response.json({
          status: 'ok',
          api_schema_version: 2,
          device: 'mps',
          model: {
            ready: true,
            model_version: 'champion-v7',
            model_step: 700,
            model_identity: `sha256-${'a'.repeat(64)}`,
            role: 'champion',
          },
          search: {
            defaults: { simulations: 512, max_considered: 16 },
            maximums: { simulations: 4_096, max_considered: 64 },
            presets: {
              quick: { simulations: 128, max_considered: 8 },
              strong: { simulations: 512, max_considered: 16 },
              maximum: { simulations: 4_096, max_considered: 64 },
            },
          },
          rules: {
            schema_id: 'edgeconnect.star.rules.v2',
            hash: 'fnv1a64:2da3783519381453',
          },
          features: {
            schema_id: 'edgeconnect.star.model-features.external.v2',
            version: 3,
            hash: STAR_FEATURE_SCHEMA_HASH,
          },
          actions: {
            schema_id: 'edgeconnect.star.action-layout.nodes-only.v1',
          },
        }),
      ),
    );
    await expect(checkServerAiCapability()).resolves.toEqual({
      status: 'available',
      label: 'Server AI',
      device: 'mps',
      champion: {
        role: 'champion',
        modelVersion: 'champion-v7',
        modelStep: 700,
        modelIdentity: `sha256-${'a'.repeat(64)}`,
      },
      search: {
        default: { simulations: 512, maxConsidered: 16 },
        maximum: { simulations: 4_096, maxConsidered: 64 },
        presets: {
          quick: { simulations: 128, maxConsidered: 8 },
          strong: { simulations: 512, maxConsidered: 16 },
          maximum: { simulations: 4_096, maxConsidered: 64 },
        },
      },
    });
  });

  it('rejects malformed optional server limits instead of clamping them', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        Response.json({
          status: 'ok',
          api_schema_version: 2,
          device: 'mps',
          model: { ready: true },
          search: {
            defaults: { simulations: 512, max_considered: 16 },
            maximums: { simulations: 128, max_considered: 8 },
            presets: {
              strong: { simulations: 512, max_considered: 16 },
            },
          },
          rules: {
            schema_id: 'edgeconnect.star.rules.v2',
            hash: 'fnv1a64:2da3783519381453',
          },
          features: {
            schema_id: 'edgeconnect.star.model-features.external.v2',
            version: 3,
            hash: STAR_FEATURE_SCHEMA_HASH,
          },
          actions: {
            schema_id: 'edgeconnect.star.action-layout.nodes-only.v1',
          },
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
