import { StarAiError } from './errors';
import {
  parseStarAiDecision,
  responseFromStarAiDecision,
  type StarAiDecision,
  type StarAiSearchBudget,
} from './decision';
import type { StarAiRequest, StarAiResponse } from './protocol';
import {
  parseBrowserSearchBudget,
  parseWorkerEvent,
  type StarAiWorkerCommand,
  type StarAiWorkerEvent,
} from './worker-protocol';

interface PendingRequest {
  request: StarAiRequest;
  resolve: (decision: StarAiDecision) => void;
  reject: (error: StarAiError) => void;
  signal?: AbortSignal;
  abortListener?: () => void;
  timeout?: ReturnType<typeof setTimeout>;
}

type WorkerFactory = () => Worker;

export interface LocalAiRequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
  search?: StarAiSearchBudget;
}

export const DEFAULT_LOCAL_AI_TIMEOUT_MS = 90_000;
export const LOCAL_AI_HANDSHAKE_TIMEOUT_MS = 5_000;
export const LOCAL_AI_IDLE_TIMEOUT_MS = 60_000;

function createLocalWorker(): Worker {
  return new Worker(new URL('../../../workers/star-ai.worker.ts', import.meta.url), {
    type: 'module',
    name: 'star-local-ai',
  });
}

export class LocalStarAiClient {
  private worker: Worker | null = null;
  private readonly pending = new Map<string, PendingRequest>();
  private readyPromise: Promise<Worker> | null = null;
  private readyResolve: ((worker: Worker) => void) | null = null;
  private readyReject: ((error: StarAiError) => void) | null = null;
  private handshakeTimeout: ReturnType<typeof setTimeout> | null = null;
  private idleTimeout: ReturnType<typeof setTimeout> | null = null;

  constructor(private readonly workerFactory: WorkerFactory = createLocalWorker) {}

  request(
    request: StarAiRequest,
    options: LocalAiRequestOptions = {},
  ): Promise<StarAiDecision> {
    const { signal } = options;
    if (signal?.aborted) {
      return Promise.reject(new StarAiError('cancelled', 'AI request cancelled.'));
    }
    if (this.pending.has(request.requestId)) {
      return Promise.reject(new StarAiError('protocol', 'Duplicate local AI request id.'));
    }

    const timeoutMs = options.timeoutMs ?? DEFAULT_LOCAL_AI_TIMEOUT_MS;
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      return Promise.reject(new StarAiError('protocol', 'Local AI timeout is invalid.'));
    }
    let search: StarAiSearchBudget | null = null;
    try {
      if (options.search !== undefined) {
        search = parseBrowserSearchBudget(options.search);
      }
    } catch (error) {
      return Promise.reject(
        error instanceof StarAiError
          ? error
          : new StarAiError('protocol', 'Local AI search budget is invalid.', false, error),
      );
    }

    return new Promise<StarAiDecision>((resolve, reject) => {
      const pending: PendingRequest = {
        request,
        resolve,
        reject: (error) => reject(error),
        signal,
      };
      if (signal) {
        pending.abortListener = () => {
          if (!this.pending.has(request.requestId)) return;
          this.resetWorker(new StarAiError('cancelled', 'AI request cancelled.'));
        };
        signal.addEventListener('abort', pending.abortListener, { once: true });
      }
      pending.timeout = setTimeout(() => {
        if (!this.pending.has(request.requestId)) return;
        this.resetWorker(new StarAiError('timeout', 'Local AI timed out.', true));
      }, timeoutMs);
      this.pending.set(request.requestId, pending);
      this.clearIdleTimeout();
      let ready: Promise<Worker>;
      try {
        ready = this.ensureWorker();
      } catch (error) {
        this.resetWorker(
          error instanceof StarAiError
            ? error
            : new StarAiError(
                'unavailable',
                'Local AI worker is unavailable.',
                true,
                error,
              ),
        );
        return;
      }
      void ready
        .then((worker) => {
          if (!this.pending.has(request.requestId)) return;
          const command: StarAiWorkerCommand = {
            type: 'choose',
            taskId: request.requestId,
            request,
            search,
          };
          worker.postMessage(command);
        })
        .catch((error) => {
          if (!this.pending.has(request.requestId)) return;
          this.resetWorker(
            error instanceof StarAiError
              ? error
              : new StarAiError(
                  'unavailable',
                  'Local AI worker is unavailable.',
                  true,
                  error,
                ),
          );
        });
    });
  }

  dispose(): void {
    this.resetWorker(new StarAiError('cancelled', 'Local AI disposed.'));
  }

  private ensureWorker(): Promise<Worker> {
    if (this.worker && this.readyPromise) return this.readyPromise;
    if (typeof Worker === 'undefined') {
      throw new StarAiError('unavailable', 'Local AI requires browser Web Worker support.');
    }
    const worker = this.workerFactory();
    worker.addEventListener('message', this.onMessage);
    worker.addEventListener('error', this.onWorkerError);
    this.worker = worker;
    this.readyPromise = new Promise<Worker>((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });
    this.handshakeTimeout = setTimeout(() => {
      this.resetWorker(
        new StarAiError('unavailable', 'Local AI worker failed to start.', true),
      );
    }, LOCAL_AI_HANDSHAKE_TIMEOUT_MS);
    return this.readyPromise;
  }

  private readonly onMessage = (event: MessageEvent<unknown>) => {
    let message: StarAiWorkerEvent;
    try {
      message = parseWorkerEvent(event.data);
    } catch (error) {
      this.resetWorker(
        error instanceof StarAiError
          ? error
          : new StarAiError('protocol', 'Local AI worker protocol failed.', false, error),
      );
      return;
    }
    if (message.type === 'ready') {
      if (!this.worker || !this.readyResolve) return;
      if (this.handshakeTimeout) clearTimeout(this.handshakeTimeout);
      this.handshakeTimeout = null;
      const resolve = this.readyResolve;
      this.readyResolve = null;
      this.readyReject = null;
      resolve(this.worker);
      return;
    }
    const pending = this.pending.get(message.taskId);
    if (!pending) return;
    this.pending.delete(message.taskId);
    this.cleanPending(pending);

    if (message.type === 'error') {
      pending.reject(
        new StarAiError(
          message.error.code,
          message.error.message,
          message.error.retryable,
        ),
      );
      this.scheduleIdleDisposal();
      return;
    }
    try {
      pending.resolve(parseStarAiDecision(pending.request, message.decision));
    } catch (error) {
      pending.reject(
        error instanceof StarAiError
          ? error
          : new StarAiError('protocol', 'Local AI returned an invalid action.', false, error),
      );
    }
    this.scheduleIdleDisposal();
  };

  private readonly onWorkerError = (event: ErrorEvent) => {
    this.resetWorker(
      new StarAiError(
        'internal',
        event.message || 'Local AI worker crashed.',
        true,
        event.error,
      ),
    );
  };

  private removeAbortListener(pending: PendingRequest): void {
    if (pending.signal && pending.abortListener) {
      pending.signal.removeEventListener('abort', pending.abortListener);
    }
  }

  private cleanPending(pending: PendingRequest): void {
    this.removeAbortListener(pending);
    if (pending.timeout) clearTimeout(pending.timeout);
  }

  private clearIdleTimeout(): void {
    if (this.idleTimeout) clearTimeout(this.idleTimeout);
    this.idleTimeout = null;
  }

  private scheduleIdleDisposal(): void {
    if (this.pending.size > 0 || !this.worker) return;
    this.clearIdleTimeout();
    this.idleTimeout = setTimeout(() => this.dispose(), LOCAL_AI_IDLE_TIMEOUT_MS);
  }

  private resetWorker(error: StarAiError): void {
    this.clearIdleTimeout();
    if (this.handshakeTimeout) clearTimeout(this.handshakeTimeout);
    this.handshakeTimeout = null;
    this.readyReject?.(error);
    this.readyResolve = null;
    this.readyReject = null;
    this.readyPromise = null;
    for (const pending of this.pending.values()) {
      this.cleanPending(pending);
      pending.reject(error);
    }
    this.pending.clear();
    if (this.worker) {
      this.worker.removeEventListener('message', this.onMessage);
      this.worker.removeEventListener('error', this.onWorkerError);
      this.worker.terminate();
      this.worker = null;
    }
  }
}

const localClient = new LocalStarAiClient();

export function requestLocalAiAction(
  request: StarAiRequest,
  options: LocalAiRequestOptions = {},
): Promise<StarAiResponse> {
  return requestLocalAiDecision(request, options).then(responseFromStarAiDecision);
}

export function requestLocalAiDecision(
  request: StarAiRequest,
  options: LocalAiRequestOptions = {},
): Promise<StarAiDecision> {
  return localClient.request(request, options);
}

export function disposeLocalAiClient(): void {
  localClient.dispose();
}
