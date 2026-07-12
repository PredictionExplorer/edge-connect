import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  checkAiCapabilities,
  type AiCapabilities,
} from '@/lib/star/ai/capabilities';
import {
  DEFAULT_AI_SEARCH_SETTINGS,
  DEFAULT_CONFIG,
  useAppStore,
  type AppState,
} from '@/lib/store';
import { SetupScreen } from '../SetupScreen';

vi.mock('@/lib/star/ai/capabilities', async () => {
  const actual = await vi.importActual<typeof import('@/lib/star/ai/capabilities')>(
    '@/lib/star/ai/capabilities',
  );
  return { ...actual, checkAiCapabilities: vi.fn() };
});

const availableCapabilities: AiCapabilities = {
  server: { status: 'available', label: 'Server AI' },
  local: { status: 'available', label: 'Local AI' },
};

const developerCapabilities: AiCapabilities = {
  server: {
    status: 'available',
    label: 'Server AI',
    search: {
      default: { simulations: 512, maxConsidered: 16 },
      maximum: { simulations: 4_096, maxConsidered: 64 },
      presets: {
        quick: { simulations: 128, maxConsidered: 8 },
        strong: { simulations: 512, maxConsidered: 16 },
        maximum: { simulations: 4_096, maxConsidered: 64 },
      },
    },
  },
  local: {
    status: 'available',
    label: 'Local AI',
    search: {
      default: { simulations: 64, maxConsidered: 16 },
      maximum: { simulations: 1_024, maxConsidered: 128 },
      presets: {
        quick: { simulations: 64, maxConsidered: 8 },
        strong: { simulations: 64, maxConsidered: 16 },
        maximum: { simulations: 1_024, maxConsidered: 128 },
      },
    },
  },
};

function resetStore(overrides: Partial<AppState> = {}) {
  const config = overrides.config ?? DEFAULT_CONFIG;
  useAppStore.setState({
    phase: 'setup',
    config: { ...config, playerNames: [...config.playerNames] },
    controllers: ['human', 'human'],
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

beforeEach(() => {
  localStorage.clear();
  vi.stubEnv('NEXT_PUBLIC_STAR_AI_DEVTOOLS', '0');
  resetStore();
  vi.mocked(checkAiCapabilities).mockReset();
  vi.mocked(checkAiCapabilities).mockResolvedValue(availableCapabilities);
});

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.unstubAllEnvs();
});

describe('SetupScreen', () => {
  it('stores trimmed fallback player names when fields are blank', async () => {
    const user = userEvent.setup();
    render(<SetupScreen />);

    const playerOne = screen.getByRole('textbox', { name: 'Player 1 name' });
    const playerTwo = screen.getByRole('textbox', { name: 'Player 2 name' });
    expect(playerOne).toHaveValue('Player 1');
    expect(playerTwo).toHaveValue('Player 2');
    expect(screen.getByRole('slider', { name: 'Custom' })).toHaveAttribute(
      'min',
      '4',
    );
    expect(screen.getByRole('slider', { name: 'Custom' })).toHaveAttribute(
      'max',
      '10',
    );
    expect(screen.getByRole('slider', { name: 'Custom' })).toHaveAttribute(
      'step',
      '2',
    );
    fireEvent.change(screen.getByRole('slider', { name: 'Custom' }), {
      target: { value: '9' },
    });
    expect(screen.getByRole('slider', { name: 'Custom' })).toHaveValue('6');

    await user.clear(playerOne);
    await user.clear(playerTwo);
    await user.click(screen.getByRole('button', { name: /begin the game/i }));

    expect(useAppStore.getState()).toMatchObject({
      phase: 'playing',
      controllers: ['human', 'human'],
      config: { playerNames: ['Player 1', 'Player 2'] },
    });
  });

  it('shows controllers only for a supported mode and resets them for the pie rule', async () => {
    const user = userEvent.setup();
    render(<SetupScreen />);

    const classic = screen.getByRole('button', { name: '*Star, 1 stone per turn' });
    const double = screen.getByRole('button', { name: /Double \*Star, 2 stones per turn/i });
    expect(classic).toHaveAttribute('aria-pressed', 'true');
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument();

    await user.click(double);
    const playerOneController = await screen.findByRole('combobox', {
      name: 'Player 1 controller',
    });
    expect(double).toHaveAttribute('aria-pressed', 'true');

    await waitFor(() =>
      expect(
        within(playerOneController).getByRole('option', { name: 'Server AI' }),
      ).toBeEnabled(),
    );
    await user.selectOptions(playerOneController, 'server');
    expect(playerOneController).toHaveValue('server');
    expect(
      screen.queryByText('Engine developer settings'),
    ).not.toBeInTheDocument();

    const pieRule = screen.getByRole('checkbox', { name: /pie rule/i });
    await user.click(pieRule);
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument();

    await user.click(pieRule);
    expect(
      screen.getByRole('combobox', { name: 'Player 1 controller' }),
    ).toHaveValue('human');
  });

  it('blocks setup until a selected controller is ready and supports rechecking', async () => {
    resetStore({
      config: {
        rings: 4,
        mode: 'double',
        pieRule: false,
        playerNames: ['Ada', 'Grace'],
      },
      controllers: ['server', 'human'],
    });
    vi.mocked(checkAiCapabilities).mockResolvedValueOnce({
      server: {
        status: 'unavailable',
        label: 'Server AI',
        code: 'server_unavailable',
        reason: 'Server AI is offline.',
        retryable: true,
      },
      local: { status: 'available', label: 'Local AI' },
    });

    const user = userEvent.setup();
    render(<SetupScreen />);

    const begin = screen.getByRole('button', { name: /begin the game/i });
    expect(begin).toBeDisabled();
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Server AI: Server AI is offline.',
    );
    expect(begin).toBeDisabled();

    await user.click(
      screen.getByRole('button', { name: /check ai availability again/i }),
    );
    await waitFor(() => expect(begin).toBeEnabled());
    expect(checkAiCapabilities).toHaveBeenCalledTimes(2);
  });

  it('offers exact validated developer budgets from capability presets', async () => {
    vi.stubEnv('NEXT_PUBLIC_STAR_AI_DEVTOOLS', '1');
    vi.mocked(checkAiCapabilities).mockResolvedValue(developerCapabilities);
    const user = userEvent.setup();
    render(<SetupScreen />);

    await user.click(
      screen.getByRole('button', { name: /Double \*Star, 2 stones per turn/i }),
    );
    expect(
      screen.queryByText('Engine developer settings'),
    ).not.toBeInTheDocument();

    const playerOneController = screen.getByRole('combobox', {
      name: 'Player 1 controller',
    });
    await waitFor(() =>
      expect(
        within(playerOneController).getByRole('option', {
          name: 'Mac engine — current champion',
        }),
      ).toBeEnabled(),
    );
    expect(
      within(playerOneController).getByRole('option', {
        name: 'Browser AI — lightweight',
      }),
    ).toBeEnabled();
    await user.selectOptions(playerOneController, 'server');

    const settingsSummary = screen.getByText('Engine developer settings');
    const settingsDetails = settingsSummary.closest('details');
    expect(settingsDetails).not.toHaveAttribute('open');
    await user.click(settingsSummary);

    const quick = screen.getByRole('button', {
      name: /Quick, 128 simulations.*8 candidates/i,
    });
    const strong = screen.getByRole('button', {
      name: /Strong, 512 simulations.*16 candidates/i,
    });
    const maximum = screen.getByRole('button', {
      name: /Maximum, 4,096 simulations.*64 candidates/i,
    });
    expect(strong).toHaveAttribute('aria-pressed', 'true');
    expect(quick).toHaveAttribute('aria-pressed', 'false');
    expect(maximum).toHaveAttribute('aria-pressed', 'false');

    await user.click(quick);
    expect(useAppStore.getState().aiSearchSettings.server).toEqual({
      simulations: 128,
      maxConsidered: 8,
    });
    expect(screen.getByText(/Runs exactly 128 simulations/i)).toBeInTheDocument();

    await user.click(screen.getByText('Advanced search budget'));
    const simulations = screen.getByRole('spinbutton', {
      name: 'Simulations',
    });
    fireEvent.change(simulations, { target: { value: '4097' } });
    expect(simulations).toHaveAttribute('aria-invalid', 'true');
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Simulations must be a whole number from 1 to 4,096.',
    );
    expect(useAppStore.getState().aiSearchSettings.server.simulations).toBe(128);
    await waitFor(() =>
      expect(
        screen.getByRole('button', { name: /begin the game/i }),
      ).toBeDisabled(),
    );

    fireEvent.change(simulations, { target: { value: '777' } });
    await waitFor(() =>
      expect(useAppStore.getState().aiSearchSettings.server).toEqual({
        simulations: 777,
        maxConsidered: 8,
      }),
    );
    expect(simulations).toHaveAttribute('aria-invalid', 'false');
    await waitFor(() =>
      expect(
        screen.getByRole('button', { name: /begin the game/i }),
      ).toBeEnabled(),
    );
  });

  it('keeps developer engine controls accessible when expanded', async () => {
    vi.stubEnv('NEXT_PUBLIC_STAR_AI_DEVTOOLS', '1');
    vi.mocked(checkAiCapabilities).mockResolvedValue(developerCapabilities);
    const user = userEvent.setup();
    const { container } = render(<SetupScreen />);

    await user.click(
      screen.getByRole('button', { name: /Double \*Star, 2 stones per turn/i }),
    );
    const controller = screen.getByRole('combobox', {
      name: 'Player 1 controller',
    });
    await waitFor(() =>
      expect(
        within(controller).getByRole('option', {
          name: 'Mac engine — current champion',
        }),
      ).toBeEnabled(),
    );
    await user.selectOptions(controller, 'server');
    await user.click(screen.getByText('Engine developer settings'));
    await user.click(screen.getByText('Advanced search budget'));

    expect((await axe(container)).violations).toEqual([]);
  });

  it('has no detectable accessibility violations', async () => {
    const { container } = render(<SetupScreen />);
    await waitFor(() => expect(checkAiCapabilities).toHaveBeenCalledOnce());

    expect((await axe(container)).violations).toEqual([]);
  });
});
