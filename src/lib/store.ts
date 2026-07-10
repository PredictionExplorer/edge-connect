'use client';

import { useSyncExternalStore } from 'react';
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { GameAction, GameConfig } from './star/game';

export type Phase = 'setup' | 'playing';

interface AppState {
  phase: Phase;
  config: GameConfig;
  log: GameAction[];
  redoStack: GameAction[];
  /** Dismissed the game-over overlay to review the final board. */
  reviewing: boolean;

  startGame: (config: GameConfig) => void;
  act: (action: GameAction) => void;
  undo: () => void;
  redo: () => void;
  rematch: () => void;
  toSetup: () => void;
  setReviewing: (reviewing: boolean) => void;
}

export const DEFAULT_CONFIG: GameConfig = {
  rings: 6,
  mode: 'classic',
  pieRule: false,
  playerNames: ['Player 1', 'Player 2'],
};

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      phase: 'setup',
      config: DEFAULT_CONFIG,
      log: [],
      redoStack: [],
      reviewing: false,

      startGame: (config) =>
        set({ phase: 'playing', config, log: [], redoStack: [], reviewing: false }),
      act: (action) =>
        set((s) => ({ log: [...s.log, action], redoStack: [], reviewing: false })),
      undo: () =>
        set((s) =>
          s.log.length === 0
            ? s
            : {
                log: s.log.slice(0, -1),
                redoStack: [...s.redoStack, s.log[s.log.length - 1]],
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
              },
        ),
      rematch: () => set({ log: [], redoStack: [], reviewing: false }),
      toSetup: () => set({ phase: 'setup', log: [], redoStack: [], reviewing: false }),
      setReviewing: (reviewing) => set({ reviewing }),
    }),
    {
      name: 'edgeconnect-star-v1',
      partialize: (s) => ({
        phase: s.phase,
        config: s.config,
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
