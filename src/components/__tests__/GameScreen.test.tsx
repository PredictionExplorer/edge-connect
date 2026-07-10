import { StrictMode } from 'react';
import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { StarAiError } from '@/lib/star/ai/errors';
import {
  acceptAiResponse,
  makeAiResponse,
  type StarAiResponse,
} from '@/lib/star/ai/protocol';
import { requestServerAiAction } from '@/lib/star/ai/server-client';
import {
  useAppStore,
  type AppState,
} from '@/lib/store';
import { GameScreen } from '../GameScreen';

vi.mock('@/lib/star/ai/protocol', async () => {
  const actual = await vi.importActual<typeof import('@/lib/star/ai/protocol')>(
    '@/lib/star/ai/protocol',
  );
  return { ...actual, acceptAiResponse: vi.fn(actual.acceptAiResponse) };
});

vi.mock('@/lib/star/ai/server-client', async () => {
  const actual = await vi.importActual<typeof import('@/lib/star/ai/server-client')>(
    '@/lib/star/ai/server-client',
  );
  return { ...actual, requestServerAiAction: vi.fn() };
});

const config = {
  rings: 3,
  mode: 'double',
  pieRule: false,
  playerNames: ['Ada', 'Grace'],
} as const;

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason: unknown) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function resetPlayingStore(overrides: Partial<AppState> = {}) {
  useAppStore.setState({
    phase: 'playing',
    config: { ...config, playerNames: [...config.playerNames] },
    controllers: ['server', 'human'],
    aiPaused: false,
    log: [],
    redoStack: [],
    reviewing: false,
    ...overrides,
  });
}

function PhaseHarness() {
  const phase = useAppStore((state) => state.phase);
  return phase === 'playing' ? <GameScreen /> : <p>Setup is ready</p>;
}

beforeEach(() => {
  localStorage.clear();
  resetPlayingStore();
  vi.mocked(requestServerAiAction).mockReset();
});

afterEach(() => {
  cleanup();
  localStorage.clear();
});

describe('GameScreen AI lifecycle', () => {
  it('reuses one logical request when Strict Mode replays effects', async () => {
    const flight = deferred<StarAiResponse>();
    vi.mocked(requestServerAiAction).mockReturnValue(flight.promise);

    render(
      <StrictMode>
        <GameScreen />
      </StrictMode>,
    );

    expect(await screen.findByText('Server AI is thinking…')).toBeInTheDocument();
    expect(requestServerAiAction).toHaveBeenCalledOnce();
    const [request, options] = vi.mocked(requestServerAiAction).mock.calls[0];
    expect(options?.signal?.aborted).toBe(false);

    flight.resolve(makeAiResponse(request, { type: 'place', node: 0 }));
    await waitFor(() =>
      expect(useAppStore.getState().log).toEqual([{ type: 'place', node: 0 }]),
    );
    expect(requestServerAiAction).toHaveBeenCalledOnce();
  });

  it('aborts on exit and ignores a response that arrives after cancellation', async () => {
    const flight = deferred<StarAiResponse>();
    vi.mocked(requestServerAiAction).mockReturnValue(flight.promise);
    const user = userEvent.setup();
    render(<PhaseHarness />);

    await screen.findByText('Server AI is thinking…');
    const [request, options] = vi.mocked(requestServerAiAction).mock.calls[0];

    await user.click(screen.getByRole('button', { name: 'New game' }));
    expect(options?.signal?.aborted).toBe(true);
    expect(screen.getByText('Setup is ready')).toBeInTheDocument();

    await act(async () => {
      flight.resolve(makeAiResponse(request, { type: 'place', node: 0 }));
      await flight.promise;
    });
    expect(acceptAiResponse).not.toHaveBeenCalled();
    expect(useAppStore.getState().log).toEqual([]);
  });

  it('rejects a stale response without mutating the action log', async () => {
    const flight = deferred<StarAiResponse>();
    vi.mocked(requestServerAiAction).mockReturnValue(flight.promise);
    render(<GameScreen />);

    await screen.findByText('Server AI is thinking…');
    const [request] = vi.mocked(requestServerAiAction).mock.calls[0];
    flight.resolve({
      ...makeAiResponse(request, { type: 'place', node: 0 }),
      requestId: 'obsolete-request',
    });

    await waitFor(() => expect(acceptAiResponse).toHaveBeenCalledOnce());
    expect(useAppStore.getState().log).toEqual([]);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('retries a recoverable error and applies the successful response', async () => {
    const retryFlight = deferred<StarAiResponse>();
    vi.mocked(requestServerAiAction)
      .mockRejectedValueOnce(new StarAiError('network', 'Server AI is offline.', true))
      .mockReturnValueOnce(retryFlight.promise);
    const user = userEvent.setup();
    render(<GameScreen />);

    expect(await screen.findByRole('alert')).toHaveTextContent('Server AI is offline.');
    await user.click(screen.getByRole('button', { name: 'Retry' }));
    await waitFor(() => expect(requestServerAiAction).toHaveBeenCalledTimes(2));

    const [retryRequest] = vi.mocked(requestServerAiAction).mock.calls[1];
    retryFlight.resolve(makeAiResponse(retryRequest, { type: 'place', node: 0 }));
    await waitFor(() =>
      expect(useAppStore.getState().log).toEqual([{ type: 'place', node: 0 }]),
    );
  });

  it('lets the current player take over after an AI error', async () => {
    vi.mocked(requestServerAiAction).mockRejectedValue(
      new StarAiError('network', 'Server AI is offline.', true),
    );
    const user = userEvent.setup();
    render(<GameScreen />);

    await screen.findByRole('alert');
    await user.click(screen.getByRole('button', { name: 'Take over as human' }));

    expect(useAppStore.getState().controllers).toEqual(['human', 'human']);
    const firstNode = screen.getByRole('button', {
      name: /node \*10, empty interior node; ada may place here/i,
    });
    await user.click(firstNode);
    expect(useAppStore.getState().log).toEqual([{ type: 'place', node: 0 }]);
  });

  it('has no detectable accessibility violations for a human turn', async () => {
    resetPlayingStore({ controllers: ['human', 'human'] });
    const { container } = render(<GameScreen />);

    expect((await axe(container)).violations).toEqual([]);
    expect(requestServerAiAction).not.toHaveBeenCalled();
  });
});
