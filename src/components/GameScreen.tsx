'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  BookOpenText,
  Eye,
  Flag,
  Redo2,
  Replace,
  Settings2,
  Trophy,
  Undo2,
} from 'lucide-react';
import { controllerLabel, type ControllerType } from '@/lib/star/ai/controllers';
import {
  StarAiError,
  asStarAiError,
  type StarAiErrorCode,
} from '@/lib/star/ai/errors';
import {
  acceptAiResponse,
  buildAiRequest,
  semanticStateFromGame,
  semanticStateHash,
  type StarAiRequest,
} from '@/lib/star/ai/protocol';
import { requestServerAiAction } from '@/lib/star/ai/server-client';
import { replay } from '@/lib/star/game';
import { scorePosition } from '@/lib/star/scoring';
import { useAppStore } from '@/lib/store';
import { GameOverOverlay } from './GameOverOverlay';
import { RulesDialog } from './RulesDialog';
import { ScorePanel } from './ScorePanel';
import { StarBoard } from './StarBoard';
import { PLAYER_COLORS } from './theme';

type AiStatus =
  | { kind: 'idle' }
  | { kind: 'thinking'; controller: Exclude<ControllerType, 'human'> }
  | {
      kind: 'error';
      controller: Exclude<ControllerType, 'human'>;
      code: StarAiErrorCode;
      message: string;
      retryable: boolean;
    };

interface AiFlight {
  key: string;
  request: StarAiRequest;
  controller: Exclude<ControllerType, 'human'>;
  abortController: AbortController;
  cancelScheduled: boolean;
  cancelled: boolean;
}

export function GameScreen() {
  const { config, controllers, aiPaused, log, redoStack, reviewing } = useAppStore();
  const {
    act,
    undo,
    redo,
    rematch,
    toSetup,
    resumeAi,
    setPlayerController,
    setReviewing,
  } = useAppStore();

  const [rulesOpen, setRulesOpen] = useState(false);
  const [showInfluence, setShowInfluence] = useState(false);
  const [hoverNode, setHoverNode] = useState(-1);
  const [aiStatus, setAiStatus] = useState<AiStatus>({ kind: 'idle' });
  const [retryNonce, setRetryNonce] = useState(0);
  const flightRef = useRef<AiFlight | null>(null);

  const game = useMemo(() => {
    try {
      return replay(config, log);
    } catch {
      return null; // corrupted persisted log
    }
  }, [config, log]);

  useEffect(() => {
    if (game === null) toSetup();
  }, [game, toSetup]);

  const aiPositionKey = useMemo(() => {
    if (!game || game.over) return null;
    const controller = controllers[game.toMove];
    if (controller === 'human') return null;
    if (typeof BigInt !== 'function') return `${controller}:bigint-unavailable`;
    return `${controller}:${semanticStateHash(semanticStateFromGame(game))}`;
  }, [controllers, game]);

  const cancelActiveAi = useCallback(() => {
    const flight = flightRef.current;
    if (!flight) return;
    flight.cancelScheduled = false;
    flight.cancelled = true;
    flightRef.current = null;
    flight.abortController.abort();
  }, []);

  useEffect(() => {
    if (!game || game.over || aiPaused) return;
    const controller = controllers[game.toMove];
    if (controller === 'human' || !aiPositionKey) return;

    const key = `${aiPositionKey}:${retryNonce}`;
    const scheduleCancellation = (flight: AiFlight) => {
      flight.cancelScheduled = true;
      queueMicrotask(() => {
        if (flightRef.current !== flight || !flight.cancelScheduled) return;
        flight.cancelled = true;
        flightRef.current = null;
        flight.abortController.abort();
      });
    };

    const existing = flightRef.current;
    if (existing?.key === key) {
      // React Strict Mode replays effects. Reattach to the same logical
      // request before the cleanup microtask can cancel it.
      existing.cancelScheduled = false;
      return () => scheduleCancellation(existing);
    }
    if (existing) {
      existing.cancelled = true;
      existing.abortController.abort();
      flightRef.current = null;
    }

    let request: StarAiRequest;
    try {
      request = buildAiRequest(config, log);
    } catch (error) {
      const aiError = asStarAiError(error);
      queueMicrotask(() => {
        setAiStatus({
          kind: 'error',
          controller,
          code: aiError.code,
          message: aiError.message,
          retryable: aiError.retryable,
        });
      });
      return;
    }

    const flight: AiFlight = {
      key,
      request,
      controller,
      abortController: new AbortController(),
      cancelScheduled: false,
      cancelled: false,
    };
    flightRef.current = flight;
    queueMicrotask(() => {
      if (flightRef.current === flight && !flight.cancelled) {
        setAiStatus({ kind: 'thinking', controller });
      }
    });

    const response =
      controller === 'server'
        ? requestServerAiAction(request, { signal: flight.abortController.signal })
        : import('@/lib/star/ai/local-client').then(({ requestLocalAiAction }) =>
            requestLocalAiAction(request, { signal: flight.abortController.signal }),
          );

    void response
      .then((payload) => {
        if (flight.cancelled || flightRef.current !== flight) return;
        const current = useAppStore.getState();
        if (current.phase !== 'playing') return;
        const accepted = acceptAiResponse(request, payload, current.config, current.log);
        if (!accepted.ok) {
          if (accepted.code === 'stale') return;
          throw new StarAiError(accepted.code, accepted.message, false);
        }
        const currentGame = replay(current.config, current.log);
        if (
          currentGame.over ||
          current.controllers[currentGame.toMove] !== flight.controller
        ) {
          return;
        }
        current.act(accepted.action);
      })
      .catch((error) => {
        if (flight.cancelled || flightRef.current !== flight) return;
        const aiError = asStarAiError(error);
        if (aiError.code === 'cancelled') return;
        setAiStatus({
          kind: 'error',
          controller: flight.controller,
          code: aiError.code,
          message: aiError.message,
          retryable: aiError.retryable,
        });
      })
      .finally(() => {
        if (flightRef.current === flight) flightRef.current = null;
      });

    return () => scheduleCancellation(flight);
  }, [aiPaused, aiPositionKey, config, controllers, game, log, retryNonce]);

  const disposeLocalAi = useCallback(() => {
    void import('@/lib/star/ai/local-client').then(({ disposeLocalAiClient }) => {
      disposeLocalAiClient();
    });
  }, []);

  const leaveGame = useCallback(() => {
    cancelActiveAi();
    if (controllers.includes('local')) disposeLocalAi();
    toSetup();
  }, [cancelActiveAi, controllers, disposeLocalAi, toSetup]);

  const undoAction = useCallback(() => {
    cancelActiveAi();
    undo();
  }, [cancelActiveAi, undo]);

  const redoAction = useCallback(() => {
    cancelActiveAi();
    redo();
  }, [cancelActiveAi, redo]);

  const rematchAction = useCallback(() => {
    cancelActiveAi();
    rematch();
  }, [cancelActiveAi, rematch]);

  const takeOverAsHuman = useCallback(
    (player: 0 | 1, controller: ControllerType) => {
      cancelActiveAi();
      if (controller === 'local') disposeLocalAi();
      setAiStatus({ kind: 'idle' });
      setPlayerController(player, 'human');
    },
    [cancelActiveAi, disposeLocalAi, setPlayerController],
  );

  const resumeAiAction = useCallback(() => {
    setAiStatus({ kind: 'idle' });
    resumeAi();
  }, [resumeAi]);

  const score = useMemo(
    () => (game ? scorePosition(game.board, game.stones) : null),
    [game],
  );

  if (!game || !score) return null;

  const { board } = game;
  const showTerritory = game.over || showInfluence;
  const activeColor = PLAYER_COLORS[game.toMove];
  const currentController = controllers[game.toMove];
  const thinking =
    currentController !== 'human' &&
    aiStatus.kind === 'thinking' &&
    aiStatus.controller === currentController;
  const activeAiError =
    currentController !== 'human' &&
    aiStatus.kind === 'error' &&
    aiStatus.controller === currentController
      ? aiStatus
      : null;
  const humanCanAct = currentController === 'human' && !thinking;

  return (
    <main className="relative z-10 mx-auto flex w-full max-w-7xl flex-1 flex-col px-4 py-5 sm:px-6">
      <header className="mb-4 flex items-center justify-between gap-3">
        <button type="button" onClick={leaveGame} className="group flex items-baseline gap-3 text-left">
          <span className="font-display text-shimmer text-3xl font-semibold leading-none">
            ✳Star
          </span>
          <span className="hidden text-xs text-muted sm:block">
            {config.mode === 'double' ? 'Double *Star' : 'Classic'} · {config.rings} rings ·{' '}
            {board.periCount + 1} points in the sky
          </span>
        </button>
        <nav className="flex items-center gap-2">
          {game.over && reviewing && (
            <button
              type="button"
              onClick={() => setReviewing(false)}
              className="flex items-center gap-2 rounded-xl border border-gold/60 bg-gold-faint px-3.5 py-2 text-sm text-gold-strong transition-colors hover:bg-gold/25"
            >
              <Trophy className="h-4 w-4" aria-hidden /> Result
            </button>
          )}
          <button
            type="button"
            onClick={() => setRulesOpen(true)}
            className="flex items-center gap-2 rounded-xl border border-white/15 px-3.5 py-2 text-sm text-ink transition-colors hover:border-gold/50"
          >
            <BookOpenText className="h-4 w-4" aria-hidden />
            <span className="hidden sm:inline">Rules</span>
          </button>
          <button
            type="button"
            onClick={leaveGame}
            className="flex items-center gap-2 rounded-xl border border-white/15 px-3.5 py-2 text-sm text-ink transition-colors hover:border-gold/50"
          >
            <Settings2 className="h-4 w-4" aria-hidden />
            <span className="hidden sm:inline">New game</span>
          </button>
        </nav>
      </header>

      <div className="grid flex-1 gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        {/* Board */}
        <div className="relative flex items-center justify-center">
          <StarBoard
            board={board}
            stones={game.stones}
            nodeOwner={score.nodeOwner}
            aliveStone={score.aliveStone}
            showTerritory={showTerritory}
            lastMove={game.lastMove}
            currentTurnMoves={game.currentTurnMoves}
            toMove={game.toMove}
            interactive={!game.over && humanCanAct}
            onPlace={(node) => {
              if (humanCanAct) act({ type: 'place', node });
            }}
            onHover={setHoverNode}
            className="max-h-[82dvh] w-full max-w-[860px]"
          />
          {/* Node readout */}
          <div
            aria-live="polite"
            className="pointer-events-none absolute bottom-1 left-2 rounded-lg border border-white/10 bg-black/30 px-2.5 py-1 font-mono text-xs text-muted backdrop-blur-sm"
          >
            {hoverNode >= 0 ? (
              <>
                node <span className="text-gold">{board.labels[hoverNode]}</span>
                {board.isQuark[hoverNode]
                  ? ' · quark'
                  : board.isPeri[hoverNode]
                    ? ' · peri'
                    : ''}
              </>
            ) : (
              `${game.stonesPlaced} / ${board.n} nodes filled`
            )}
          </div>
        </div>

        {/* Side panel */}
        <div className="flex min-w-0 flex-col gap-4">
          {/* Turn banner */}
          <div
            key={game.over ? 'over' : `${game.toMove}-${game.movesLeft}-${game.passStreak}`}
            className="fade-in rounded-2xl border px-4 py-3 text-center text-sm"
            style={{
              borderColor: game.over ? 'rgba(232,196,139,0.5)' : activeColor.base + '66',
              background: game.over ? 'rgba(232,196,139,0.10)' : activeColor.soft,
            }}
          >
            {game.over ? (
              <span className="text-gold-strong">Game over — the sky is settled</span>
            ) : (
              <span style={{ color: activeColor.bright }}>
                {config.playerNames[game.toMove]} to play
                {config.mode === 'double' &&
                  ` — ${game.movesLeft} stone${game.movesLeft > 1 ? 's' : ''} left`}
                {game.passStreak === 1 && ' · a pass now ends the game'}
              </span>
            )}
          </div>

          {!game.over && currentController !== 'human' && (
            <div
              aria-live="polite"
              className="rounded-xl border border-white/10 bg-white/[0.03] px-3.5 py-2.5 text-xs text-muted"
            >
              {aiPaused ? (
                <div>
                  <p>AI is paused after history navigation.</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={resumeAiAction}
                      className="rounded-lg border border-gold/50 px-2.5 py-1 text-gold-strong transition-colors hover:bg-gold/15"
                    >
                      Resume AI
                    </button>
                    <button
                      type="button"
                      onClick={() => takeOverAsHuman(game.toMove, currentController)}
                      className="rounded-lg border border-white/20 px-2.5 py-1 text-ink transition-colors hover:border-gold/40"
                    >
                      Take over as human
                    </button>
                  </div>
                </div>
              ) : activeAiError ? (
                <div>
                  <p role="alert">
                    {activeAiError.message}{' '}
                    <span className="font-mono text-[10px] opacity-70">
                      ({activeAiError.code})
                    </span>
                  </p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {activeAiError.retryable && (
                      <button
                        type="button"
                        onClick={() => setRetryNonce((value) => value + 1)}
                        className="rounded-lg border border-gold/50 px-2.5 py-1 text-gold-strong transition-colors hover:bg-gold/15"
                      >
                        Retry
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => takeOverAsHuman(game.toMove, currentController)}
                      className="rounded-lg border border-white/20 px-2.5 py-1 text-ink transition-colors hover:border-gold/40"
                    >
                      Take over as human
                    </button>
                  </div>
                </div>
              ) : (
                <span>{controllerLabel(currentController)} is thinking…</span>
              )}
            </div>
          )}

          {/* Pie rule offer */}
          {game.canSwap && (
            <div className="fade-up rounded-2xl border border-dashed border-gold/50 bg-gold-faint px-4 py-3 text-sm">
              <p className="text-ink">
                Pie rule — {config.playerNames[1]} may steal the opening stone.
              </p>
              <button
                type="button"
                disabled={!humanCanAct}
                onClick={() => {
                  if (humanCanAct) act({ type: 'swap' });
                }}
                className="mt-2 flex items-center gap-2 rounded-lg border border-gold/60 px-3 py-1.5 text-xs font-medium text-gold-strong transition-colors hover:bg-gold/20"
              >
                <Replace className="h-3.5 w-3.5" aria-hidden /> Steal it (swap sides)
              </button>
            </div>
          )}

          <ScorePanel game={game} score={score} />

          {/* Actions */}
          <div className="grid grid-cols-3 gap-2">
            <button
              type="button"
              disabled={game.over || !humanCanAct}
              onClick={() => {
                if (humanCanAct) act({ type: 'pass' });
              }}
              className="flex items-center justify-center gap-2 rounded-xl border border-white/15 px-3 py-2.5 text-sm text-ink transition-colors enabled:hover:border-danger/60 enabled:hover:text-danger disabled:opacity-35"
            >
              <Flag className="h-4 w-4" aria-hidden /> Pass
            </button>
            <button
              type="button"
              disabled={log.length === 0}
              onClick={undoAction}
              className="flex items-center justify-center gap-2 rounded-xl border border-white/15 px-3 py-2.5 text-sm text-ink transition-colors enabled:hover:border-gold/50 disabled:opacity-35"
            >
              <Undo2 className="h-4 w-4" aria-hidden /> Undo
            </button>
            <button
              type="button"
              disabled={redoStack.length === 0}
              onClick={redoAction}
              className="flex items-center justify-center gap-2 rounded-xl border border-white/15 px-3 py-2.5 text-sm text-ink transition-colors enabled:hover:border-gold/50 disabled:opacity-35"
            >
              <Redo2 className="h-4 w-4" aria-hidden /> Redo
            </button>
          </div>

          <label className="flex cursor-pointer items-center justify-between rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3">
            <span className="flex items-center gap-2 text-sm text-ink">
              <Eye className="h-4 w-4 text-muted" aria-hidden /> Show influence
            </span>
            <input
              type="checkbox"
              checked={showTerritory}
              disabled={game.over}
              onChange={(e) => setShowInfluence(e.target.checked)}
              className="h-4 w-4 accent-[#e8c48b]"
            />
          </label>

          <p className="px-1 text-[11px] leading-relaxed text-muted">
            Influence shows who currently claims each peri — occupied or walled off — and dims
            stones that do not yet belong to a star. It is always on once the game ends.
          </p>
        </div>
      </div>

      <RulesDialog open={rulesOpen} onClose={() => setRulesOpen(false)} />
      <GameOverOverlay
        open={game.over && !reviewing}
        game={game}
        score={score}
        onReview={() => setReviewing(true)}
        onRematch={rematchAction}
        onSetup={leaveGame}
      />
    </main>
  );
}
