import { StrictMode } from 'react';
import {
  act,
  cleanup,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type {
  StarAiAnalysis,
  StarAiDecision,
} from '@/lib/star/ai/decision';
import { StarAiError } from '@/lib/star/ai/errors';
import {
  acceptAiResponse,
  makeAiResponse,
  type AtomicGameAction,
  type StarAiRequest,
} from '@/lib/star/ai/protocol';
import { requestLocalAiDecision } from '@/lib/star/ai/local-client';
import { requestServerAiDecision } from '@/lib/star/ai/server-client';
import { getBoard, parseLabel } from '@/lib/star/board';
import {
  DEFAULT_AI_SEARCH_SETTINGS,
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
  return { ...actual, requestServerAiDecision: vi.fn() };
});

vi.mock('@/lib/star/ai/local-client', async () => {
  const actual = await vi.importActual<typeof import('@/lib/star/ai/local-client')>(
    '@/lib/star/ai/local-client',
  );
  return { ...actual, requestLocalAiDecision: vi.fn() };
});

const config = {
  rings: 4,
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

function makeDecision(
  request: StarAiRequest,
  action: AtomicGameAction = { type: 'place', node: 0 },
  overrides: Partial<StarAiAnalysis> = {},
): StarAiDecision {
  const rootActions = overrides.rootActions ?? [
    action,
    { type: 'place', node: action.node + 1 },
    { type: 'place', node: action.node + 2 },
  ];
  const rootVisits = overrides.rootVisits ?? [6, 3, 1];
  const simulations =
    overrides.simulations ??
    rootVisits.reduce((total, visits) => total + visits, 0);
  return {
    response: makeAiResponse(request, action),
    analysis: {
      perspective: request.state.toMove,
      stateHash: request.stateHash,
      outcome: { loss: 0.25, win: 0.75 },
      modelValue: 0.5,
      searchValue: 0.4,
      expectedMargin: 2.5,
      rootActions,
      rootPolicy: rootActions.map((_, index) => (index === 0 ? 0.6 : 0.2)),
      rootQ: rootActions.map((_, index) => 0.3 - index * 0.2),
      rootVisits,
      modelVersion: 'test-champion',
      modelStep: 700,
      modelIdentity: 'sha256-test',
      simulations,
      maxConsidered: 16,
      timingMs: {
        queue: 1,
        modelLoad: 2,
        inferenceSearch: 9,
        total: 12,
      },
      ...overrides,
    },
  };
}

function resetPlayingStore(overrides: Partial<AppState> = {}) {
  useAppStore.setState({
    phase: 'playing',
    config: { ...config, playerNames: [...config.playerNames] },
    controllers: ['server', 'human'],
    aiSearchSettings: {
      server: { ...DEFAULT_AI_SEARCH_SETTINGS.server },
      local: { ...DEFAULT_AI_SEARCH_SETTINGS.local },
    },
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
  vi.stubEnv('NEXT_PUBLIC_STAR_AI_DEVTOOLS', '0');
  resetPlayingStore();
  vi.mocked(requestServerAiDecision).mockReset();
  vi.mocked(requestLocalAiDecision).mockReset();
});

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.unstubAllEnvs();
});

describe('GameScreen AI lifecycle', () => {
  it('reuses one logical request when Strict Mode replays effects', async () => {
    const flight = deferred<StarAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(flight.promise);

    render(
      <StrictMode>
        <GameScreen />
      </StrictMode>,
    );

    expect(await screen.findByText('Server AI is thinking…')).toBeInTheDocument();
    expect(requestServerAiDecision).toHaveBeenCalledOnce();
    const [request, options] = vi.mocked(requestServerAiDecision).mock.calls[0];
    expect(options?.signal?.aborted).toBe(false);
    expect(options?.search).toBeUndefined();

    flight.resolve(makeDecision(request));
    await waitFor(() =>
      expect(useAppStore.getState().log).toEqual([{ type: 'place', node: 0 }]),
    );
    expect(requestServerAiDecision).toHaveBeenCalledOnce();
    expect(
      screen.queryByRole('status', { name: 'Engine estimate' }),
    ).not.toBeInTheDocument();
  });

  it('aborts on exit and ignores a response that arrives after cancellation', async () => {
    vi.stubEnv('NEXT_PUBLIC_STAR_AI_DEVTOOLS', '1');
    const flight = deferred<StarAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(flight.promise);
    const user = userEvent.setup();
    render(<PhaseHarness />);

    await screen.findByText('Mac engine — current champion is thinking…');
    const [request, options] = vi.mocked(requestServerAiDecision).mock.calls[0];

    await user.click(screen.getByRole('button', { name: 'New game' }));
    expect(options?.signal?.aborted).toBe(true);
    expect(screen.getByText('Setup is ready')).toBeInTheDocument();

    await act(async () => {
      flight.resolve(makeDecision(request));
      await flight.promise;
    });
    expect(acceptAiResponse).not.toHaveBeenCalled();
    expect(useAppStore.getState().log).toEqual([]);
    expect(
      screen.queryByRole('status', { name: 'Engine estimate' }),
    ).not.toBeInTheDocument();
  });

  it('rejects a stale response without mutating the action log', async () => {
    vi.stubEnv('NEXT_PUBLIC_STAR_AI_DEVTOOLS', '1');
    const flight = deferred<StarAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(flight.promise);
    render(<GameScreen />);

    await screen.findByText('Mac engine — current champion is thinking…');
    const [request] = vi.mocked(requestServerAiDecision).mock.calls[0];
    const stale = makeDecision(request);
    flight.resolve({
      ...stale,
      response: { ...stale.response, requestId: 'obsolete-request' },
    });

    await waitFor(() => expect(acceptAiResponse).toHaveBeenCalledOnce());
    expect(useAppStore.getState().log).toEqual([]);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(
      screen.queryByRole('status', { name: 'Engine estimate' }),
    ).not.toBeInTheDocument();
  });

  it('retries a recoverable error and applies the successful response', async () => {
    const retryFlight = deferred<StarAiDecision>();
    vi.mocked(requestServerAiDecision)
      .mockRejectedValueOnce(new StarAiError('network', 'Server AI is offline.', true))
      .mockReturnValueOnce(retryFlight.promise);
    const user = userEvent.setup();
    render(<GameScreen />);

    expect(await screen.findByRole('alert')).toHaveTextContent('Server AI is offline.');
    await user.click(screen.getByRole('button', { name: 'Retry' }));
    await waitFor(() => expect(requestServerAiDecision).toHaveBeenCalledTimes(2));

    const [retryRequest] = vi.mocked(requestServerAiDecision).mock.calls[1];
    retryFlight.resolve(makeDecision(retryRequest));
    await waitFor(() =>
      expect(useAppStore.getState().log).toEqual([{ type: 'place', node: 0 }]),
    );
  });

  it('lets the current player take over after an AI error', async () => {
    vi.mocked(requestServerAiDecision).mockRejectedValue(
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

  it('passes the selected server and local budgets only in developer mode', async () => {
    vi.stubEnv('NEXT_PUBLIC_STAR_AI_DEVTOOLS', '1');
    resetPlayingStore({
      aiSearchSettings: {
        server: { simulations: 777, maxConsidered: 21 },
        local: { simulations: 123, maxConsidered: 9 },
      },
    });
    const serverFlight = deferred<StarAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(serverFlight.promise);
    const first = render(<GameScreen />);

    await screen.findByText('Mac engine — current champion is thinking…');
    expect(requestServerAiDecision).toHaveBeenCalledOnce();
    expect(vi.mocked(requestServerAiDecision).mock.calls[0][1]?.search).toEqual({
      simulations: 777,
      maxConsidered: 21,
    });
    first.unmount();

    resetPlayingStore({
      controllers: ['local', 'human'],
      aiSearchSettings: {
        server: { simulations: 777, maxConsidered: 21 },
        local: { simulations: 123, maxConsidered: 9 },
      },
    });
    const localFlight = deferred<StarAiDecision>();
    vi.mocked(requestLocalAiDecision).mockReturnValue(localFlight.promise);
    render(<GameScreen />);

    await screen.findByText('Browser AI — lightweight is thinking…');
    expect(requestLocalAiDecision).toHaveBeenCalledOnce();
    expect(vi.mocked(requestLocalAiDecision).mock.calls[0][1]?.search).toEqual({
      simulations: 123,
      maxConsidered: 9,
    });
  });

  it('renders named estimates and top board labels from the accepted perspective', async () => {
    vi.stubEnv('NEXT_PUBLIC_STAR_AI_DEVTOOLS', '1');
    const flight = deferred<StarAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(flight.promise);
    const { container } = render(<GameScreen />);

    await screen.findByText('Mac engine — current champion is thinking…');
    const [request] = vi.mocked(requestServerAiDecision).mock.calls[0];
    flight.resolve(
      makeDecision(request, { type: 'place', node: 0 }, {
        outcome: { loss: 0.2, win: 0.8 },
        modelValue: 0.6,
        searchValue: -0.25,
        expectedMargin: 3.5,
        rootActions: [
          { type: 'place', node: 0 },
          { type: 'place', node: 1 },
          { type: 'place', node: 2 },
        ],
        rootPolicy: [0.2, 0.5, 0.3],
        rootQ: [-0.1, 0.7, 0.3],
        rootVisits: [2, 7, 5],
        simulations: 14,
        modelStep: 12_345,
        timingMs: {
          queue: 1,
          modelLoad: 2,
          inferenceSearch: 15,
          total: 18,
        },
      }),
    );

    const panel = await screen.findByRole('status', { name: 'Engine estimate' });
    expect(within(panel).getByText('Ada 80.0%')).toBeInTheDocument();
    expect(within(panel).getByText('Grace 20.0%')).toBeInTheDocument();
    expect(within(panel).getByText('Ada +3.5 points')).toBeInTheDocument();
    expect(within(panel).getByText('-0.250')).toBeInTheDocument();
    expect(within(panel).queryByText('37.5%')).not.toBeInTheDocument();
    expect(within(panel).getByText('14')).toBeInTheDocument();
    expect(within(panel).getByText('18 ms')).toBeInTheDocument();
    expect(within(panel).getByText('12,345')).toBeInTheDocument();
    const candidates = within(panel).getAllByRole('listitem');
    expect(candidates[0]).toHaveTextContent('S10');
    expect(candidates[1]).toHaveTextContent('T10');
    expect(candidates[2]).toHaveTextContent('*10');
    expect((await axe(container)).violations).toEqual([]);
  });

  it('maps a second-player analysis to the correct named win estimates', async () => {
    vi.stubEnv('NEXT_PUBLIC_STAR_AI_DEVTOOLS', '1');
    resetPlayingStore({
      controllers: ['human', 'server'],
      log: [
        { type: 'place', node: 0 },
        { type: 'place', node: 1 },
      ],
    });
    const flight = deferred<StarAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(flight.promise);
    render(<GameScreen />);

    await screen.findByText('Mac engine — current champion is thinking…');
    const [request] = vi.mocked(requestServerAiDecision).mock.calls[0];
    expect(request.state.toMove).toBe(1);
    flight.resolve(
      makeDecision(request, { type: 'place', node: 2 }, {
        outcome: { loss: 0.3, win: 0.7 },
        modelValue: 0.4,
        expectedMargin: -4,
      }),
    );

    const panel = await screen.findByRole('status', { name: 'Engine estimate' });
    expect(within(panel).getByText('Ada 30.0%')).toBeInTheDocument();
    expect(within(panel).getByText('Grace 70.0%')).toBeInTheDocument();
    expect(within(panel).getByText('Ada +4.0 points')).toBeInTheDocument();
    expect(within(panel).getByText(/from Grace's turn/i)).toBeInTheDocument();
  });

  it('clears a published estimate on history navigation', async () => {
    vi.stubEnv('NEXT_PUBLIC_STAR_AI_DEVTOOLS', '1');
    const flight = deferred<StarAiDecision>();
    vi.mocked(requestServerAiDecision).mockReturnValue(flight.promise);
    const user = userEvent.setup();
    render(<GameScreen />);

    await screen.findByText('Mac engine — current champion is thinking…');
    const [request] = vi.mocked(requestServerAiDecision).mock.calls[0];
    flight.resolve(makeDecision(request));
    await screen.findByRole('status', { name: 'Engine estimate' });

    await user.click(screen.getByRole('button', { name: 'Undo' }));
    expect(
      screen.queryByRole('status', { name: 'Engine estimate' }),
    ).not.toBeInTheDocument();
  });

  it('clears the previous estimate when the next engine request errors', async () => {
    vi.stubEnv('NEXT_PUBLIC_STAR_AI_DEVTOOLS', '1');
    resetPlayingStore({ controllers: ['server', 'server'] });
    const first = deferred<StarAiDecision>();
    const second = deferred<StarAiDecision>();
    vi.mocked(requestServerAiDecision)
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    render(<GameScreen />);

    await screen.findByText('Mac engine — current champion is thinking…');
    const [firstRequest] = vi.mocked(requestServerAiDecision).mock.calls[0];
    first.resolve(makeDecision(firstRequest));
    await screen.findByRole('status', { name: 'Engine estimate' });
    await waitFor(() =>
      expect(requestServerAiDecision).toHaveBeenCalledTimes(2),
    );

    second.reject(new StarAiError('timeout', 'Engine timed out.', true));
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Engine timed out.',
    );
    expect(
      screen.queryByRole('status', { name: 'Engine estimate' }),
    ).not.toBeInTheDocument();
  });

  it('retries a timeout with an explicitly reduced search budget', async () => {
    vi.stubEnv('NEXT_PUBLIC_STAR_AI_DEVTOOLS', '1');
    const retryFlight = deferred<StarAiDecision>();
    vi.mocked(requestServerAiDecision)
      .mockRejectedValueOnce(new StarAiError('timeout', 'Engine timed out.', true))
      .mockReturnValueOnce(retryFlight.promise);
    const user = userEvent.setup();
    render(<GameScreen />);

    await screen.findByRole('alert');
    await user.click(screen.getByRole('button', { name: 'Use less effort' }));
    await waitFor(() =>
      expect(requestServerAiDecision).toHaveBeenCalledTimes(2),
    );
    expect(useAppStore.getState().aiSearchSettings.server).toEqual({
      simulations: 256,
      maxConsidered: 8,
    });
    expect(vi.mocked(requestServerAiDecision).mock.calls[1][1]?.search).toEqual({
      simulations: 256,
      maxConsidered: 8,
    });
  });

  it('has no detectable accessibility violations for a human turn', async () => {
    resetPlayingStore({ controllers: ['human', 'human'] });
    const { container } = render(<GameScreen />);

    expect(screen.queryByRole('button', { name: 'Pass' })).not.toBeInTheDocument();
    expect((await axe(container)).violations).toEqual([]);
    expect(requestServerAiDecision).not.toHaveBeenCalled();
  });
});

describe('GameScreen score guidance', () => {
  it('shows both extreme completion scores throughout play', () => {
    resetPlayingStore({ controllers: ['human', 'human'] });
    render(<GameScreen />);

    expect(
      screen.getByRole('heading', { name: 'Completion bounds' }),
    ).toBeInTheDocument();
    expect(screen.getByText('All open → Ada')).toBeInTheDocument();
    expect(screen.getByText('All open → Grace')).toBeInTheDocument();
    expect(
      screen.getByText('The final winner is not clinched yet.'),
    ).toBeInTheDocument();
  });

  it('announces a clinched result without ending the game', () => {
    resetPlayingStore({
      controllers: ['human', 'human'],
      log: Array.from({ length: 49 }, (_, node) => ({
        type: 'place' as const,
        node,
      })),
    });
    render(<GameScreen />);

    expect(
      screen.getByText('Grace has clinched the game', { exact: true }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/even if every open node went to ada/i),
    ).toBeInTheDocument();
    expect(screen.getByLabelText('Ada scores 10')).toBeInTheDocument();
    expect(screen.getByLabelText('Grace scores 11')).toBeInTheDocument();
    expect(screen.getByText(/Grace to play/)).toBeInTheDocument();
    expect(screen.queryByRole('dialog', { name: 'Game over' })).not.toBeInTheDocument();
  });

  it('suppresses completion guidance while a pie swap can recolor the opening', () => {
    resetPlayingStore({
      controllers: ['human', 'human'],
      config: {
        ...config,
        pieRule: true,
        playerNames: [...config.playerNames],
      },
      log: [{ type: 'place', node: 0 }],
    });
    render(<GameScreen />);

    expect(screen.getByText(/may steal the opening stone/i)).toBeInTheDocument();
    expect(
      screen.queryByRole('heading', { name: 'Completion bounds' }),
    ).not.toBeInTheDocument();
  });

  it('does not cross rescuable stones in projected opponent territory', () => {
    const board = getBoard(4);
    resetPlayingStore({
      controllers: ['human', 'human'],
      log: ['S10', '*40', '*41'].map((label) => ({
        type: 'place' as const,
        node: parseLabel(board, label),
      })),
    });
    const { container } = render(<GameScreen />);

    expect(
      screen.getByText('Current scoring projection'),
    ).toBeInTheDocument();
    expect(
      container.querySelectorAll('[data-provably-dead-stone]'),
    ).toHaveLength(0);
    expect(
      screen.getByRole('button', {
        name: /node s10, ada stone.*not currently part of a living star/i,
      }),
    ).toBeInTheDocument();
  });

  it('removes and restores a provably dead marker through undo and redo', async () => {
    const user = userEvent.setup();
    const board = getBoard(4);
    const dead = parseLabel(board, '*43');
    resetPlayingStore({
      controllers: ['human', 'human'],
      log: ['*43', '*42', '*32', 'T42', 'T43', 'S30', 'S40'].map(
        (label) => ({
          type: 'place' as const,
          node: parseLabel(board, label),
        }),
      ),
    });
    const { container } = render(<GameScreen />);

    expect(
      container.querySelector(`[data-provably-dead-stone="${dead}"]`),
    ).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Undo' }));
    expect(
      container.querySelector(`[data-provably-dead-stone="${dead}"]`),
    ).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Redo' }));
    expect(
      container.querySelector(`[data-provably-dead-stone="${dead}"]`),
    ).toBeInTheDocument();
  });
});
