'use client';

import { useSyncExternalStore } from 'react';
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { isSupportedRings } from './star/board';
import {
  HUMAN_CONTROLLERS,
  isControllerType,
  normalizeControllers,
  type ControllerType,
  type PlayerControllers,
} from './star/ai/controllers';
import { replay, type GameAction, type GameConfig } from './star/game';

export type Phase = 'setup' | 'playing';

export interface AppState {
  phase: Phase;
  config: GameConfig;
  controllers: PlayerControllers;
  aiPaused: boolean;
  log: GameAction[];
  redoStack: GameAction[];
  /** Dismissed the game-over overlay to review the final board. */
  reviewing: boolean;

  startGame: (config: GameConfig, controllers: PlayerControllers) => void;
  act: (action: GameAction) => void;
  undo: () => void;
  redo: () => void;
  rematch: () => void;
  toSetup: () => void;
  resumeAi: () => void;
  setPlayerController: (player: 0 | 1, controller: ControllerType) => void;
  setReviewing: (reviewing: boolean) => void;
}

export interface PersistedAppState {
  phase: Phase;
  config: GameConfig;
  controllers: PlayerControllers;
  aiPaused: boolean;
  log: GameAction[];
  redoStack: GameAction[];
}

export const DEFAULT_CONFIG: GameConfig = {
  rings: 6,
  mode: 'classic',
  pieRule: false,
  playerNames: ['Player 1', 'Player 2'],
};

export const DEFAULT_CONTROLLERS: PlayerControllers = [...HUMAN_CONTROLLERS];
export const APP_STORE_VERSION = 4;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return actual.length === expected.length && actual.every((key, index) => key === expected[index]);
}

export function parseGameConfig(value: unknown): GameConfig | null {
  if (!isRecord(value) || !hasExactKeys(value, ['rings', 'mode', 'pieRule', 'playerNames'])) {
    return null;
  }
  const names = value.playerNames;
  if (
    typeof value.rings !== 'number' ||
    !isSupportedRings(value.rings) ||
    (value.mode !== 'classic' && value.mode !== 'double') ||
    typeof value.pieRule !== 'boolean' ||
    !Array.isArray(names) ||
    names.length !== 2 ||
    typeof names[0] !== 'string' ||
    typeof names[1] !== 'string'
  ) {
    return null;
  }
  return {
    rings: value.rings,
    mode: value.mode,
    pieRule: value.pieRule,
    playerNames: [names[0], names[1]],
  };
}

export function normalizeGameConfig(value: unknown): GameConfig {
  const record = isRecord(value) ? value : {};
  const names = Array.isArray(record.playerNames) ? record.playerNames : [];
  return {
    rings: isSupportedRings(record.rings) ? record.rings : DEFAULT_CONFIG.rings,
    mode:
      record.mode === 'classic' || record.mode === 'double'
        ? record.mode
        : DEFAULT_CONFIG.mode,
    pieRule:
      typeof record.pieRule === 'boolean' ? record.pieRule : DEFAULT_CONFIG.pieRule,
    playerNames: [
      typeof names[0] === 'string' ? names[0] : DEFAULT_CONFIG.playerNames[0],
      typeof names[1] === 'string' ? names[1] : DEFAULT_CONFIG.playerNames[1],
    ],
  };
}

export function parseGameAction(value: unknown): GameAction | null {
  if (!isRecord(value) || typeof value.type !== 'string') return null;
  if (value.type === 'swap') {
    return hasExactKeys(value, ['type']) ? { type: 'swap' } : null;
  }
  if (
    value.type === 'place' &&
    hasExactKeys(value, ['type', 'node']) &&
    typeof value.node === 'number' &&
    Number.isInteger(value.node) &&
    value.node >= 0
  ) {
    return { type: 'place', node: value.node };
  }
  return null;
}

function allGameActions(values: Array<GameAction | null>): values is GameAction[] {
  return values.every((action) => action !== null);
}

function setupSnapshot(
  config: GameConfig = DEFAULT_CONFIG,
  controllers: PlayerControllers = DEFAULT_CONTROLLERS,
): PersistedAppState {
  return {
    phase: 'setup',
    config: { ...config, playerNames: [...config.playerNames] },
    controllers: normalizeControllers(config, controllers),
    aiPaused: false,
    log: [],
    redoStack: [],
  };
}

export function sanitizePersistedState(value: unknown): PersistedAppState {
  if (!isRecord(value)) return setupSnapshot();
  const config = parseGameConfig(value.config);
  if (!config) return setupSnapshot();
  const controllers = normalizeControllers(config, value.controllers);
  if (value.phase !== 'playing') return setupSnapshot(config, controllers);
  if (!Array.isArray(value.log) || !Array.isArray(value.redoStack)) {
    return setupSnapshot(config, controllers);
  }
  const log = value.log.map(parseGameAction);
  const redoStack = value.redoStack.map(parseGameAction);
  if (!allGameActions(log) || !allGameActions(redoStack)) {
    return setupSnapshot(config, controllers);
  }
  try {
    replay(config, log);
    replay(config, [...log, ...[...redoStack].reverse()]);
  } catch {
    return setupSnapshot(config, controllers);
  }
  return {
    phase: 'playing',
    config,
    controllers,
    aiPaused: value.aiPaused === true,
    log,
    redoStack,
  };
}

export function migratePersistedState(
  value: unknown,
  persistedVersion: number,
): PersistedAppState {
  if (persistedVersion >= APP_STORE_VERSION) {
    return sanitizePersistedState(value);
  }
  const record = isRecord(value) ? value : {};
  const config = normalizeGameConfig(record.config);
  const controllers = normalizeControllers(config, record.controllers);
  return setupSnapshot(config, controllers);
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      phase: 'setup',
      config: DEFAULT_CONFIG,
      controllers: DEFAULT_CONTROLLERS,
      aiPaused: false,
      log: [],
      redoStack: [],
      reviewing: false,

      startGame: (config, controllers) => {
        const validConfig = parseGameConfig(config);
        if (!validConfig) {
          throw new Error('cannot start a game with an unsupported configuration');
        }
        set({
          phase: 'playing',
          config: validConfig,
          controllers: normalizeControllers(validConfig, controllers),
          aiPaused: false,
          log: [],
          redoStack: [],
          reviewing: false,
        });
      },
      act: (action) =>
        set((s) => ({
          log: [...s.log, action],
          redoStack: [],
          aiPaused: false,
          reviewing: false,
        })),
      undo: () =>
        set((s) =>
          s.log.length === 0
            ? s
            : {
                log: s.log.slice(0, -1),
                redoStack: [...s.redoStack, s.log[s.log.length - 1]],
                aiPaused: true,
                reviewing: false,
              },
        ),
      redo: () =>
        set((s) =>
          s.redoStack.length === 0
            ? s
            : {
                log: [...s.log, s.redoStack[s.redoStack.length - 1]],
                redoStack: s.redoStack.slice(0, -1),
                aiPaused: true,
              },
        ),
      rematch: () =>
        set({ log: [], redoStack: [], aiPaused: false, reviewing: false }),
      toSetup: () =>
        set({
          phase: 'setup',
          log: [],
          redoStack: [],
          aiPaused: false,
          reviewing: false,
        }),
      resumeAi: () => set({ aiPaused: false }),
      setPlayerController: (player, controller) =>
        set((state) => {
          if ((player !== 0 && player !== 1) || !isControllerType(controller)) return state;
          const requested: PlayerControllers = [...state.controllers];
          requested[player] = controller;
          return {
            controllers: normalizeControllers(state.config, requested),
            aiPaused: false,
          };
        }),
      setReviewing: (reviewing) => set({ reviewing }),
    }),
    {
      name: 'edgeconnect-star-v1',
      version: APP_STORE_VERSION,
      migrate: migratePersistedState,
      merge: (persisted, current) => {
        const valid = sanitizePersistedState(persisted);
        return {
          ...current,
          ...valid,
          reviewing: false,
        };
      },
      partialize: (s) => ({
        phase: s.phase,
        config: s.config,
        controllers: s.controllers,
        aiPaused: s.aiPaused,
        log: s.log,
        redoStack: s.redoStack,
      }),
    },
  ),
);

const emptySubscribe = () => () => {};

/**
 * True once mounted on the client. localStorage hydration is synchronous, so
 * after mount the persisted game is loaded; gating on this avoids SSR
 * hydration mismatches.
 */
export function useMounted(): boolean {
  return useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false,
  );
}
