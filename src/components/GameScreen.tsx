'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  BookOpenText,
  Eye,
  Redo2,
  Replace,
  Settings2,
  ShieldCheck,
  Trophy,
  Undo2,
} from 'lucide-react';
import { controllerLabel, type ControllerType } from '@/lib/star/ai/controllers';
import {
  StarAiError,
  asStarAiError,
  type StarAiErrorCode,
} from '@/lib/star/ai/errors';
import type {
  StarAiAnalysis,
  StarAiSearchBudget,
} from '@/lib/star/ai/decision';
import {
  acceptAiResponse,
  buildAiRequest,
  semanticStateFromGame,
  semanticStateHash,
  type StarAiRequest,
} from '@/lib/star/ai/protocol';
import { requestServerAiDecision } from '@/lib/star/ai/server-client';
import { scoreCompletionBounds } from '@/lib/star/completion-bounds';
import { replay } from '@/lib/star/game';
import { scorePosition, validateTerminalWinner } from '@/lib/star/scoring';
import { useAppStore } from '@/lib/store';
import { EngineEstimatePanel } from './EngineEstimatePanel';
import { GameOverOverlay } from './GameOverOverlay';
import { RulesDialog } from './RulesDialog';
import { ScorePanel } from './ScorePanel';
import { StarBoard } from './StarBoard';
import {
  engineControllerLabel,
  starAiDevtoolsEnabled,
} from './starAiDevtools';
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
  settled: boolean;
}

interface PublishedAiAnalysis {
  key: string;
  analysis: StarAiAnalysis;
}

function reducedSearchBudget(
  budget: StarAiSearchBudget,
): StarAiSearchBudget | null {
  if (budget.simulations === 1 && budget.maxConsidered === 1) return null;
  return {
    simulations: Math.max(1, Math.floor(budget.simulations / 2)),
    maxConsidered: Math.max(1, Math.floor(budget.maxConsidered / 2)),
  };
}

export function GameScreen() {
  const {
    config,
    controllers,
    aiSearchSettings,
    aiPaused,
    log,
    redoStack,
    reviewing,
  } = useAppStore();
  const {
    act,
    undo,
    redo,
    rematch,
    toSetup,
    resumeAi,
    setPlayerController,
    setAiSearchBudget,
    setReviewing,
  } = useAppStore();
  const devtools = starAiDevtoolsEnabled();

  const [rulesOpen, setRulesOpen] = useState(false);
  const [showInfluence, setShowInfluence] = useState(false);
  const [hoverNode, setHoverNode] = useState(-1);
  const [aiStatus, setAiStatus] = useState<AiStatus>({ kind: 'idle' });
  const [publishedAnalysis, setPublishedAnalysis] =
    useState<PublishedAiAnalysis | null>(null);
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
    setPublishedAnalysis(null);
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

    const selectedSearch = aiSearchSettings[controller];
    const key = `${aiPositionKey}:${retryNonce}${
      devtools
        ? `:${selectedSearch.simulations}:${selectedSearch.maxConsidered}`
        : ''
    }`;
    const scheduleCancellation = (flight: AiFlight) => {
      if (flight.settled) return;
      flight.cancelScheduled = true;
      queueMicrotask(() => {
        if (flightRef.current !== flight || !flight.cancelScheduled) return;
        flight.cancelled = true;
        flightRef.current = null;
        flight.abortController.abort();
        setPublishedAnalysis(null);
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
      if (!existing.settled) setPublishedAnalysis(null);
    }

    let request: StarAiRequest;
    try {
      request = buildAiRequest(config, log);
    } catch (error) {
      const aiError = asStarAiError(error);
      queueMicrotask(() => {
        setPublishedAnalysis(null);
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
      settled: false,
    };
    flightRef.current = flight;
    queueMicrotask(() => {
      if (flightRef.current === flight && !flight.cancelled) {
        setAiStatus({ kind: 'thinking', controller });
      }
    });

    const options = {
      signal: flight.abortController.signal,
      ...(devtools ? { search: selectedSearch } : {}),
    };
    const response =
      controller === 'server'
        ? requestServerAiDecision(request, options)
        : import('@/lib/star/ai/local-client').then(({ requestLocalAiDecision }) =>
            requestLocalAiDecision(request, options),
          );

    void response
      .then((decision) => {
        if (flight.cancelled || flightRef.current !== flight) return;
        const current = useAppStore.getState();
        if (current.phase !== 'playing') {
          flight.settled = true;
          setPublishedAnalysis(null);
          return;
        }
        const accepted = acceptAiResponse(
          request,
          decision.response,
          current.config,
          current.log,
        );
        if (!accepted.ok) {
          if (accepted.code === 'stale') {
            flight.settled = true;
            setPublishedAnalysis(null);
            setAiStatus({ kind: 'idle' });
            return;
          }
          throw new StarAiError(accepted.code, accepted.message, false);
        }
        const currentGame = replay(current.config, current.log);
        if (
          currentGame.over ||
          current.controllers[currentGame.toMove] !== flight.controller
        ) {
          flight.settled = true;
          setPublishedAnalysis(null);
          return;
        }
        flight.settled = true;
        setAiStatus({ kind: 'idle' });
        current.act(accepted.action);
        if (devtools) {
          setPublishedAnalysis({
            key: `${decision.analysis.stateHash}:${decision.analysis.perspective}`,
            analysis: decision.analysis,
          });
        }
      })
      .catch((error) => {
        if (flight.cancelled || flightRef.current !== flight) return;
        flight.settled = true;
        setPublishedAnalysis(null);
        const aiError = asStarAiError(error);
        if (aiError.code === 'cancelled' || aiError.code === 'stale') {
          setAiStatus({ kind: 'idle' });
          return;
        }
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
  }, [
    aiPaused,
    aiPositionKey,
    aiSearchSettings,
    config,
    controllers,
    devtools,
    game,
    log,
    retryNonce,
  ]);

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

  const retryWithLessEffort = useCallback(
    (controller: Exclude<ControllerType, 'human'>) => {
      const reduced = reducedSearchBudget(
        useAppStore.getState().aiSearchSettings[controller],
      );
      if (!reduced) return;
      setPublishedAnalysis(null);
      setAiStatus({ kind: 'idle' });
      setAiSearchBudget(controller, reduced);
    },
    [setAiSearchBudget],
  );

  const score = useMemo(
    () => (game ? scorePosition(game.board, game.stones) : null),
    [game],
  );
  const completionBounds = useMemo(
    () =>
      game && !game.canSwap
        ? scoreCompletionBounds(game.board, game.stones)
        : null,
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
  const lowerBudget =
    currentController === 'human'
      ? null
      : reducedSearchBudget(aiSearchSettings[currentController]);
  const canUseLessEffort =
    devtools &&
    activeAiError?.retryable === true &&
    activeAiError.code === 'timeout' &&
    lowerBudget !== null;
  const humanCanAct = currentController === 'human' && !thinking;

  return (
    <main className="relative z-10 mx-auto flex w-full max-w-7xl flex-1 flex-col px-4 py-5 sm:px-6">
      <h1 className="sr-only">*Star game</h1>
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
            provablyDeadStone={completionBounds?.provablyDeadStone}
            showTerritory={showTerritory}
            lastMove={game.lastMove}
            currentTurnMoves={game.currentTurnMoves}
            toMove={game.toMove}
            interactive={!game.over && humanCanAct}
            playerNames={config.playerNames}
            onPlace={(node) => {
              if (humanCanAct) {
                setPublishedAnalysis(null);
                act({ type: 'place', node });
              }
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
            key={game.over ? 'over' : `${game.toMove}-${game.movesLeft}`}
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
              </span>
            )}
          </div>

          {!game.over &&
            completionBounds &&
            completionBounds.guaranteedWinner !== null && (
              <div
                role="status"
                aria-live="polite"
                className="fade-in flex items-start gap-3 rounded-2xl border px-4 py-3"
                style={{
                  borderColor:
                    PLAYER_COLORS[completionBounds.guaranteedWinner].base + '88',
                  background: PLAYER_COLORS[completionBounds.guaranteedWinner].soft,
                }}
              >
                <ShieldCheck
                  className="mt-0.5 h-5 w-5 shrink-0"
                  style={{
                    color: PLAYER_COLORS[completionBounds.guaranteedWinner].bright,
                  }}
                  aria-hidden
                />
                <div>
                  <p
                    className="text-sm font-medium"
                    style={{
                      color: PLAYER_COLORS[completionBounds.guaranteedWinner].bright,
                    }}
                  >
                    {config.playerNames[completionBounds.guaranteedWinner]} has clinched the game
                  </p>
                  <p className="mt-0.5 text-[11px] leading-relaxed text-muted">
                    Even if every open node went to{' '}
                    {config.playerNames[1 - completionBounds.guaranteedWinner]}, the result would
                    not change. Play may continue.
                  </p>
                </div>
              </div>
            )}

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
                        onClick={() => {
                          setPublishedAnalysis(null);
                          setAiStatus({ kind: 'idle' });
                          setRetryNonce((value) => value + 1);
                        }}
                        className="rounded-lg border border-gold/50 px-2.5 py-1 text-gold-strong transition-colors hover:bg-gold/15"
                      >
                        Retry
                      </button>
                    )}
                    {canUseLessEffort && (
                      <button
                        type="button"
                        onClick={() => retryWithLessEffort(currentController)}
                        className="rounded-lg border border-gold/50 px-2.5 py-1 text-gold-strong transition-colors hover:bg-gold/15"
                      >
                        Use less effort
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
                <span>
                  {devtools
                    ? engineControllerLabel(currentController)
                    : controllerLabel(currentController)}{' '}
                  is thinking…
                </span>
              )}
            </div>
          )}

          {devtools && publishedAnalysis && (
            <EngineEstimatePanel
              key={publishedAnalysis.key}
              analysis={publishedAnalysis.analysis}
              board={board}
              playerNames={config.playerNames}
            />
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
                  if (humanCanAct) {
                    setPublishedAnalysis(null);
                    act({ type: 'swap' });
                  }
                }}
                className="mt-2 flex items-center gap-2 rounded-lg border border-gold/60 px-3 py-1.5 text-xs font-medium text-gold-strong transition-colors hover:bg-gold/20"
              >
                <Replace className="h-3.5 w-3.5" aria-hidden /> Steal it (swap sides)
              </button>
            </div>
          )}

          <ScorePanel
            game={game}
            score={score}
            completionBounds={completionBounds}
          />

          {/* Actions */}
          <div className="grid grid-cols-2 gap-2">
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
            Influence shows the current scoring projection and dims groups that are not yet
            stars. Crosses mark only groups that cannot become a star even if they received every
            open node. Influence is always on once the game ends.
          </p>
        </div>
      </div>

      <RulesDialog open={rulesOpen} onClose={() => setRulesOpen(false)} />
      {game.over && (
        <GameOverOverlay
          open={!reviewing}
          game={game}
          score={score}
          winner={validateTerminalWinner(game.board, game.stones).winner}
          onReview={() => setReviewing(true)}
          onRematch={rematchAction}
          onSetup={leaveGame}
        />
      )}
    </main>
  );
}
