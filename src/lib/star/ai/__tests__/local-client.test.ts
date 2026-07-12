import { afterEach, describe, expect, it, vi } from 'vitest';
import type { GameConfig } from '../../game';
import { LocalStarAiClient } from '../local-client';
import {
  buildAiRequest,
  makeAiResponse,
  type StarAiRequest,
} from '../protocol';

const config: GameConfig = {
  rings: 4,
  mode: 'double',
  pieRule: false,
  playerNames: ['A', 'B'],
};

function decisionFor(
  request: StarAiRequest,
  search = { simulations: 1, maxConsidered: 1 },
) {
  return {
    response: makeAiResponse(request, { type: 'place' as const, node: 0 }),
    analysis: {
      perspective: request.state.toMove,
      stateHash: request.stateHash,
      outcome: { loss: 0.5, win: 0.5 },
      modelValue: 0,
      searchValue: 0,
      expectedMargin: 0,
      rootActions: [{ type: 'place' as const, node: 0 }],
      rootPolicy: [1],
      rootQ: [0],
      rootVisits: [search.simulations],
      modelVersion: 'browser-test',
      modelStep: null,
      modelIdentity: 'browser-test',
      simulations: search.simulations,
      maxConsidered: search.maxConsidered,
      timingMs: {
        queue: 0,
        modelLoad: 0,
        inferenceSearch: 1,
        total: 1,
      },
    },
  };
}

class FakeWorker extends EventTarget {
  readonly messages: unknown[] = [];
  terminated = false;

  postMessage(message: unknown): void {
    this.messages.push(message);
  }

  terminate(): void {
    this.terminated = true;
  }

  emit(message: unknown): void {
    this.dispatchEvent(new MessageEvent('message', { data: message }));
  }
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('local worker construction lifecycle', () => {
  it('waits for the worker handshake before choosing and disposes cleanly', async () => {
    vi.stubGlobal('Worker', class {});
    const worker = new FakeWorker();
    const client = new LocalStarAiClient(() => worker as unknown as Worker);
    const request = buildAiRequest(config, [], 'local-handshake');

    const result = client.request(request);
    expect(worker.messages).toEqual([]);
    worker.emit({ type: 'ready', protocolVersion: 2 });
    await Promise.resolve();
    expect(worker.messages).toEqual([
      { type: 'choose', taskId: request.requestId, request, search: null },
    ]);

    worker.emit({
      type: 'result',
      taskId: request.requestId,
      decision: decisionFor(request),
    });
    await expect(result).resolves.toEqual(decisionFor(request));
    client.dispose();
    expect(worker.terminated).toBe(true);
  });

  it('terminates a blocked worker on cancellation and recreates it', async () => {
    vi.stubGlobal('Worker', class {});
    const workers = [new FakeWorker(), new FakeWorker()];
    let nextWorker = 0;
    const client = new LocalStarAiClient(
      () => workers[nextWorker++] as unknown as Worker,
    );
    const abort = new AbortController();
    const firstRequest = buildAiRequest(config, [], 'local-cancel');
    const first = client.request(firstRequest, { signal: abort.signal });
    const active = workers[0];
    active.emit({ type: 'ready', protocolVersion: 2 });
    await Promise.resolve();
    abort.abort();
    await expect(first).rejects.toMatchObject({ code: 'cancelled' });
    expect(active.terminated).toBe(true);

    const secondRequest = buildAiRequest(config, [], 'local-recreated');
    const second = client.request(secondRequest);
    const replacement = workers[1];
    expect(replacement).not.toBe(active);
    replacement.emit({ type: 'ready', protocolVersion: 2 });
    await Promise.resolve();
    replacement.emit({
      type: 'result',
      taskId: secondRequest.requestId,
      decision: decisionFor(secondRequest),
    });
    await expect(second).resolves.toEqual(decisionFor(secondRequest));
    client.dispose();
  });

  it('terminates a worker when local inference times out', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('Worker', class {});
    const worker = new FakeWorker();
    const client = new LocalStarAiClient(() => worker as unknown as Worker);
    const request = buildAiRequest(config, [], 'local-timeout');
    const result = client.request(request, { timeoutMs: 100 });
    worker.emit({ type: 'ready', protocolVersion: 2 });
    await Promise.resolve();

    const rejection = expect(result).rejects.toMatchObject({
      code: 'timeout',
      retryable: true,
    });
    await vi.advanceTimersByTimeAsync(100);
    await rejection;
    expect(worker.terminated).toBe(true);
  });

  it('sends validated per-request budgets and returns compact worker analysis', async () => {
    vi.stubGlobal('Worker', class {});
    const worker = new FakeWorker();
    const client = new LocalStarAiClient(() => worker as unknown as Worker);
    const request = buildAiRequest(config, [], 'local-budget');
    const search = { simulations: 32, maxConsidered: 8 };
    const result = client.request(request, { search });
    worker.emit({ type: 'ready', protocolVersion: 2 });
    await Promise.resolve();
    expect(worker.messages).toEqual([
      { type: 'choose', taskId: request.requestId, request, search },
    ]);
    worker.emit({
      type: 'result',
      taskId: request.requestId,
      decision: decisionFor(request, search),
    });
    await expect(result).resolves.toMatchObject({
      analysis: {
        simulations: 32,
        maxConsidered: 8,
        modelIdentity: 'browser-test',
      },
    });
    await expect(
      client.request(buildAiRequest(config, [], 'invalid-budget'), {
        search: { simulations: 1_025, maxConsidered: 8 },
      }),
    ).rejects.toMatchObject({ code: 'protocol' });
    client.dispose();
  });

  it('rejects a worker decision with stale nested state identity', async () => {
    vi.stubGlobal('Worker', class {});
    const worker = new FakeWorker();
    const client = new LocalStarAiClient(() => worker as unknown as Worker);
    const request = buildAiRequest(config, [], 'local-stale');
    const result = client.request(request);
    worker.emit({ type: 'ready', protocolVersion: 2 });
    await Promise.resolve();
    const stale = decisionFor(request);
    stale.response.stateHash = 'zobrist64:0000000000000000';
    stale.analysis.stateHash = 'zobrist64:0000000000000000';
    worker.emit({
      type: 'result',
      taskId: request.requestId,
      decision: stale,
    });
    await expect(result).rejects.toMatchObject({ code: 'stale' });
    client.dispose();
  });
});
