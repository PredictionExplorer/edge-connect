import { afterEach, describe, expect, it } from 'vitest';
import {
  APP_STORE_VERSION,
  DEFAULT_AI_SEARCH_SETTINGS,
  DEFAULT_CONFIG,
  migratePersistedState,
  normalizeAiSearchSettings,
  normalizeGameConfig,
  parseAiSearchBudget,
  parseGameAction,
  sanitizePersistedState,
  useAppStore,
} from '../../store';
import type { GameConfig } from '../game';

const double: GameConfig = {
  rings: 6,
  mode: 'double',
  pieRule: false,
  playerNames: ['A', 'B'],
};

afterEach(() => {
  useAppStore.setState({
    phase: 'setup',
    config: DEFAULT_CONFIG,
    controllers: ['human', 'human'],
    aiSearchSettings: {
      server: { ...DEFAULT_AI_SEARCH_SETTINGS.server },
      local: { ...DEFAULT_AI_SEARCH_SETTINGS.local },
    },
    aiPaused: false,
    log: [],
    redoStack: [],
    reviewing: false,
  });
});

describe('persisted app-state validation', () => {
  it('rehydrates only replayable strict action logs and redo history', () => {
    expect(
      sanitizePersistedState({
        phase: 'playing',
        config: double,
        controllers: ['server', 'local'],
        aiSearchSettings: {
          server: { simulations: 128, maxConsidered: 8 },
          local: { simulations: 32, maxConsidered: 4 },
        },
        aiPaused: true,
        log: [{ type: 'place', node: 0 }],
        redoStack: [{ type: 'place', node: 1 }],
      }),
    ).toMatchObject({
      phase: 'playing',
      controllers: ['server', 'local'],
      aiSearchSettings: {
        server: { simulations: 128, maxConsidered: 8 },
        local: { simulations: 32, maxConsidered: 4 },
      },
      aiPaused: true,
      log: [{ type: 'place', node: 0 }],
      redoStack: [{ type: 'place', node: 1 }],
    });
  });

  it('returns to setup and clears malformed or illegal history', () => {
    const malformed = sanitizePersistedState({
      phase: 'playing',
      config: double,
      controllers: ['server', 'local'],
      log: [{ type: 'pass', unexpected: true }],
      redoStack: [],
    });
    expect(malformed).toMatchObject({
      phase: 'setup',
      log: [],
      redoStack: [],
      aiPaused: false,
    });

    const illegal = sanitizePersistedState({
      phase: 'playing',
      config: double,
      controllers: ['server', 'local'],
      log: [
        { type: 'place', node: 0 },
        { type: 'place', node: 0 },
      ],
      redoStack: [],
    });
    expect(illegal.phase).toBe('setup');
    expect(illegal.log).toEqual([]);
  });

  it('has no legacy pass parser path', () => {
    expect(parseGameAction({ type: 'pass' })).toBeNull();
    expect(parseGameAction({ type: 'place', node: 2 })).toEqual({
      type: 'place',
      node: 2,
    });
    expect(parseGameAction({ type: 'swap' })).toEqual({ type: 'swap' });
  });

  it('rejects incompatible configs instead of casting persisted values', () => {
    const result = sanitizePersistedState({
      phase: 'playing',
      config: { ...double, rings: '3' },
      controllers: ['server', 'local'],
      log: [],
      redoStack: [],
    });
    expect(result.phase).toBe('setup');
    expect(result.config).toEqual(DEFAULT_CONFIG);
    expect(result.controllers).toEqual(['human', 'human']);
  });

  it('resets old games without replay migration and preserves valid preferences', () => {
    const migrated = migratePersistedState(
      {
        phase: 'playing',
        config: {
          rings: 8,
          mode: 'double',
          pieRule: false,
          playerNames: ['Ada', 'Grace'],
        },
        controllers: ['server', 'local'],
        aiPaused: true,
        log: [{ type: 'place', node: 0 }],
        redoStack: [{ type: 'place', node: 1 }],
      },
      APP_STORE_VERSION - 1,
    );
    expect(migrated).toEqual({
      phase: 'setup',
      config: {
        rings: 8,
        mode: 'double',
        pieRule: false,
        playerNames: ['Ada', 'Grace'],
      },
      controllers: ['server', 'local'],
      aiSearchSettings: DEFAULT_AI_SEARCH_SETTINGS,
      aiPaused: false,
      log: [],
      redoStack: [],
    });
  });

  it('normalizes preferences independently and defaults an invalid ring to 6', () => {
    expect(
      normalizeGameConfig({
        rings: 5,
        mode: 'double',
        pieRule: true,
        playerNames: ['Ada', 17],
      }),
    ).toEqual({
      rings: 6,
      mode: 'double',
      pieRule: true,
      playerNames: ['Ada', 'Player 2'],
    });
  });

  it('validates each runtime search budget without clamping', () => {
    expect(parseAiSearchBudget('server', {
      simulations: 4_096,
      maxConsidered: 64,
    })).toEqual({ simulations: 4_096, maxConsidered: 64 });
    expect(parseAiSearchBudget('server', {
      simulations: 16_385,
      maxConsidered: 64,
    })).toBeNull();
    expect(parseAiSearchBudget('local', {
      simulations: 1_025,
      maxConsidered: 8,
    })).toBeNull();

    expect(
      normalizeAiSearchSettings({
        server: { simulations: 256, maxConsidered: 12 },
        local: { simulations: 0, maxConsidered: 8 },
      }),
    ).toEqual({
      server: { simulations: 256, maxConsidered: 12 },
      local: DEFAULT_AI_SEARCH_SETTINGS.local,
    });
  });

  it('updates valid runtime settings and ignores invalid setter values', () => {
    useAppStore.getState().setAiSearchBudget('local', {
      simulations: 128,
      maxConsidered: 8,
    });
    expect(useAppStore.getState().aiSearchSettings.local).toEqual({
      simulations: 128,
      maxConsidered: 8,
    });

    useAppStore.getState().setAiSearchBudget('local', {
      simulations: 2_048,
      maxConsidered: 8,
    });
    expect(useAppStore.getState().aiSearchSettings.local).toEqual({
      simulations: 128,
      maxConsidered: 8,
    });
  });
});

describe('history navigation AI pause', () => {
  it('pauses after undo and preserves redo when a player takes over', () => {
    useAppStore.setState({
      phase: 'playing',
      config: double,
      controllers: ['server', 'local'],
      aiPaused: false,
      log: [
        { type: 'place', node: 0 },
        { type: 'place', node: 1 },
      ],
      redoStack: [],
    });

    useAppStore.getState().undo();
    expect(useAppStore.getState()).toMatchObject({
      aiPaused: true,
      log: [{ type: 'place', node: 0 }],
      redoStack: [{ type: 'place', node: 1 }],
    });

    useAppStore.getState().setPlayerController(1, 'human');
    expect(useAppStore.getState()).toMatchObject({
      controllers: ['server', 'human'],
      aiPaused: false,
      redoStack: [{ type: 'place', node: 1 }],
    });
  });
});

describe('gameplay store actions', () => {
  it('starts, acts, redoes, reviews, rematches, and leaves without stale state', () => {
    const store = useAppStore.getState();
    store.startGame(double, ['server', 'local']);
    expect(useAppStore.getState()).toMatchObject({
      phase: 'playing',
      config: double,
      controllers: ['server', 'local'],
      log: [],
      reviewing: false,
    });

    useAppStore.getState().act({ type: 'place', node: 0 });
    useAppStore.getState().act({ type: 'place', node: 1 });
    useAppStore.getState().undo();
    useAppStore.getState().redo();
    expect(useAppStore.getState()).toMatchObject({
      log: [
        { type: 'place', node: 0 },
        { type: 'place', node: 1 },
      ],
      redoStack: [],
      aiPaused: true,
    });

    useAppStore.getState().setReviewing(true);
    expect(useAppStore.getState().reviewing).toBe(true);
    useAppStore.getState().rematch();
    expect(useAppStore.getState()).toMatchObject({
      phase: 'playing',
      log: [],
      redoStack: [],
      aiPaused: false,
      reviewing: false,
    });

    useAppStore.getState().toSetup();
    expect(useAppStore.getState()).toMatchObject({
      phase: 'setup',
      log: [],
      redoStack: [],
      aiPaused: false,
      reviewing: false,
    });
  });

  it('refuses to start unsupported board sizes instead of coercing them', () => {
    const before = useAppStore.getState();
    expect(() =>
      before.startGame({ ...double, rings: 9 }, ['human', 'human']),
    ).toThrow(/unsupported configuration/i);
    expect(useAppStore.getState().phase).toBe('setup');
  });

  it('normalizes unsupported AI controllers and ignores invalid controller slots', () => {
    const classic: GameConfig = { ...double, mode: 'classic' };
    useAppStore.getState().startGame(classic, ['server', 'local']);
    expect(useAppStore.getState().controllers).toEqual(['human', 'human']);
    const before = useAppStore.getState();
    before.setPlayerController(2 as 0, 'server');
    expect(useAppStore.getState().controllers).toEqual(['human', 'human']);
    useAppStore.getState().resumeAi();
    expect(useAppStore.getState().aiPaused).toBe(false);
  });
});
