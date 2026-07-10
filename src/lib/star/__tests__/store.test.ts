import { afterEach, describe, expect, it } from 'vitest';
import {
  DEFAULT_CONFIG,
  sanitizePersistedState,
  useAppStore,
} from '../../store';
import type { GameConfig } from '../game';

const double: GameConfig = {
  rings: 3,
  mode: 'double',
  pieRule: false,
  playerNames: ['A', 'B'],
};

afterEach(() => {
  useAppStore.setState({
    phase: 'setup',
    config: DEFAULT_CONFIG,
    controllers: ['human', 'human'],
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
        aiPaused: true,
        log: [{ type: 'place', node: 0 }],
        redoStack: [{ type: 'place', node: 1 }],
      }),
    ).toMatchObject({
      phase: 'playing',
      controllers: ['server', 'local'],
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
