import { afterEach, describe, expect, it, vi } from 'vitest';
import type { GameConfig } from '../../game';
import { LocalStarAiClient } from '../local-client';
import { buildAiRequest, makeAiResponse } from '../protocol';

const config: GameConfig = {
  rings: 4,
  mode: 'double',
  pieRule: false,
  playerNames: ['A', 'B'],
};

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
      { type: 'choose', taskId: request.requestId, request },
    ]);

    worker.emit({
      type: 'result',
      taskId: request.requestId,
      response: makeAiResponse(request, { type: 'place', node: 0 }),
    });
    await expect(result).resolves.toEqual(
      makeAiResponse(request, { type: 'place', node: 0 }),
    );
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
      response: makeAiResponse(secondRequest, { type: 'place', node: 0 }),
    });
    await expect(second).resolves.toEqual(
      makeAiResponse(secondRequest, { type: 'place', node: 0 }),
    );
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
});
