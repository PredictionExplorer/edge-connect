'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  BookOpenText,
  CircleAlert,
  Eye,
  Flag,
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
import {
  EMPTY,
  scorePosition,
  validateTerminalWinner,
} from '@/lib/star/scoring';
import { useAppStore } from '@/lib/store';
import { BoardStage } from './BoardStage';
import { EngineEstimatePanel } from './EngineEstimatePanel';
import {
  ClinchDialog,
  EndGameConfirmDialog,
  ResignDialog,
} from './EndGameDialogs';
import { GameOverOverlay, type GameResult } from './GameOverOverlay';
import { GameStatus, type GameStatusState } from './GameStatus';
import { RulesDialog } from './RulesDialog';
import { ScorePanel } from './ScorePanel';
import {
  engineControllerLabel,
  starAiDevtoolsEnabled,
} from './starAiDevtools';
import styles from './GameScreen.module.css';
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
  const config = useAppStore((state) => state.config);
  const controllers = useAppStore((state) => state.controllers);
  const aiSearchSettings = useAppStore((state) => state.aiSearchSettings);
  const aiPaused = useAppStore((state) => state.aiPaused);
  const log = useAppStore((state) => state.log);
  const redoStack = useAppStore((state) => state.redoStack);
  const reviewing = useAppStore((state) => state.reviewing);
  const earlyOutcome = useAppStore((state) => state.earlyOutcome);
  const clinchAcknowledgement = useAppStore(
    (state) => state.clinchAcknowledgement,
  );
  const act = useAppStore((state) => state.act);
  const undo = useAppStore((state) => state.undo);
  const redo = useAppStore((state) => state.redo);
  const rematch = useAppStore((state) => state.rematch);
  const toSetup = useAppStore((state) => state.toSetup);
  const resumeAi = useAppStore((state) => state.resumeAi);
  const setPlayerController = useAppStore((state) => state.setPlayerController);
  const setAiSearchBudget = useAppStore((state) => state.setAiSearchBudget);
  const setReviewing = useAppStore((state) => state.setReviewing);
  const acknowledgeClinch = useAppStore((state) => state.acknowledgeClinch);
  const endClinchedGame = useAppStore((state) => state.endClinchedGame);
  const resign = useAppStore((state) => state.resign);
  const devtools = starAiDevtoolsEnabled();

  const [rulesOpen, setRulesOpen] = useState(false);
  const [showInfluence, setShowInfluence] = useState(false);
  const [proofMode, setProofMode] = useState(false);
  const [endConfirmOpen, setEndConfirmOpen] = useState(false);
  const [resignConfirmOpen, setResignConfirmOpen] = useState(false);
  const [aiStatus, setAiStatus] = useState<AiStatus>({ kind: 'idle' });
  const [publishedAnalysis, setPublishedAnalysis] =
    useState<PublishedAiAnalysis | null>(null);
  const [retryNonce, setRetryNonce] = useState(0);
  const flightRef = useRef<AiFlight | null>(null);
  const boardStageRef = useRef<HTMLElement>(null);

  const game = useMemo(() => {
    try {
      return replay(config, log);
    } catch {
      return null; // corrupted persisted log
    }
  }, [config, log]);

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
  const guaranteedWinner = completionBounds?.guaranteedWinner ?? null;
  const clinchAcknowledged =
    guaranteedWinner !== null &&
    clinchAcknowledgement?.winner === guaranteedWinner &&
    log.length >= clinchAcknowledgement.atLogLength;
  const clinchPending =
    Boolean(game) &&
    !game?.over &&
    earlyOutcome === null &&
    guaranteedWinner !== null &&
    !clinchAcknowledged;
  const effectiveOver = Boolean(game?.over || earlyOutcome);
  const proofEligible =
    Boolean(game) &&
    !game?.over &&
    guaranteedWinner !== null &&
    earlyOutcome?.reason !== 'resignation';
  const validProofMode = proofMode && proofEligible;
  const validEndConfirmOpen =
    endConfirmOpen &&
    guaranteedWinner !== null &&
    !effectiveOver &&
    !clinchPending;
  const validResignConfirmOpen = resignConfirmOpen && !effectiveOver;
  const uiBlocksPlay =
    effectiveOver ||
    clinchPending ||
    validProofMode ||
    validEndConfirmOpen ||
    validResignConfirmOpen;

  useEffect(() => {
    if (game === null) toSetup();
  }, [game, toSetup]);

  const aiPositionKey = useMemo(() => {
    if (!game || game.over || uiBlocksPlay) return null;
    const controller = controllers[game.toMove];
    if (controller === 'human') return null;
    if (typeof BigInt !== 'function') return `${controller}:bigint-unavailable`;
    return `${controller}:${semanticStateHash(semanticStateFromGame(game))}`;
  }, [controllers, game, uiBlocksPlay]);

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
    if (!game || game.over || aiPaused || uiBlocksPlay) return;
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
          current.earlyOutcome ||
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
    uiBlocksPlay,
  ]);

  const disposeLocalAi = useCallback(() => {
    void import('@/lib/star/ai/local-client').then(({ disposeLocalAiClient }) => {
      disposeLocalAiClient();
    });
  }, []);

  const leaveGame = useCallback(() => {
    cancelActiveAi();
    setProofMode(false);
    setEndConfirmOpen(false);
    setResignConfirmOpen(false);
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
    setRulesOpen(false);
    setProofMode(false);
    setEndConfirmOpen(false);
    setResignConfirmOpen(false);
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

  const currentController = game ? controllers[game.toMove] : 'human';
  const thinking =
    Boolean(game) &&
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
  const humanCanAct =
    Boolean(game) &&
    currentController === 'human' &&
    !thinking &&
    !uiBlocksPlay;
  const placeStone = useCallback(
    (node: number) => {
      if (!humanCanAct) return;
      setPublishedAnalysis(null);
      act({ type: 'place', node });
    },
    [act, humanCanAct],
  );

  const proofWinner =
    earlyOutcome?.reason === 'resignation'
      ? null
      : earlyOutcome?.reason === 'clinch'
        ? earlyOutcome.winner
        : guaranteedWinner;
  const proofLoser =
    proofWinner === null ? null : ((1 - proofWinner) as 0 | 1);
  const proofScenario =
    proofLoser === null ? null : (completionBounds?.scenarios[proofLoser] ?? null);
  const proofActive = validProofMode && proofScenario !== null;
  const proofMask = useMemo(() => {
    if (!game || !proofActive) return null;
    return Uint8Array.from(
      { length: game.board.n },
      (_, node) => (game.stones[node] === EMPTY ? 1 : 0),
    );
  }, [game, proofActive]);
  const humanPlayers = ([0, 1] as const).filter(
    (player) => controllers[player] === 'human',
  );
  const resigningPlayer: 0 | 1 | null =
    humanPlayers.length === 1
      ? humanPlayers[0]
      : humanPlayers.length === 2 && game
        ? game.toMove
        : null;

  const showProofBoard = useCallback(() => {
    if (proofScenario === null) return;
    cancelActiveAi();
    setRulesOpen(false);
    setEndConfirmOpen(false);
    setProofMode(true);
  }, [cancelActiveAi, proofScenario]);

  const continueAfterClinch = useCallback(() => {
    if (proofWinner === null) return;
    cancelActiveAi();
    acknowledgeClinch(proofWinner);
    setRulesOpen(false);
    setProofMode(false);
    setEndConfirmOpen(false);
  }, [acknowledgeClinch, cancelActiveAi, proofWinner]);

  const endClinchNow = useCallback(() => {
    if (proofWinner === null) return;
    cancelActiveAi();
    setRulesOpen(false);
    setProofMode(false);
    setEndConfirmOpen(false);
    endClinchedGame(proofWinner);
  }, [cancelActiveAi, endClinchedGame, proofWinner]);

  const confirmResignation = useCallback(() => {
    if (resigningPlayer === null) return;
    cancelActiveAi();
    setResignConfirmOpen(false);
    resign(resigningPlayer);
  }, [cancelActiveAi, resign, resigningPlayer]);

  if (!game || !score) return null;

  const { board } = game;
  const displayedScore = proofActive ? proofScenario.score : score;
  const showTerritory = game.over || showInfluence || proofActive;
  const gameResult: GameResult | null = game.over
    ? {
        reason: 'full-board',
        winner: validateTerminalWinner(game.board, game.stones).winner,
        score,
      }
    : earlyOutcome;
  const statusPlayer =
    gameResult?.winner ??
    ((clinchPending || proofActive) && proofWinner !== null
      ? proofWinner
      : game.toMove);
  const activeColor = PLAYER_COLORS[statusPlayer];
  const gameStatusState: GameStatusState = effectiveOver
    ? 'over'
    : proofActive
      ? 'proof'
      : clinchPending
        ? 'clinch'
        : validEndConfirmOpen || validResignConfirmOpen
          ? 'confirming'
          : currentController === 'human'
            ? 'human'
            : aiPaused
              ? 'paused'
              : activeAiError
                ? 'error'
                : 'thinking';
  const currentControllerName =
    currentController === 'human'
      ? 'Human'
      : devtools
        ? engineControllerLabel(currentController)
        : controllerLabel(currentController);
  const proofDescription =
    proofActive && proofScenario && proofWinner !== null && proofLoser !== null
      ? `Proof scenario—not actual moves. Every remaining open node is hypothetically assigned to ${config.playerNames[proofLoser]}; ${config.playerNames[proofWinner]} still wins ${proofScenario.score.players[proofWinner].total} to ${proofScenario.score.players[proofLoser].total}.`
      : null;
  const scoreView = proofActive && proofLoser !== null
    ? ({ kind: 'proof', fillPlayer: proofLoser } as const)
    : earlyOutcome
      ? ({ kind: 'ended' } as const)
      : ({ kind: 'live' } as const);

  return (
    <main
      className={`${styles.screen} relative z-10 mx-auto flex w-full max-w-[100rem] flex-col`}
    >
      <h1 className="sr-only">*Star game</h1>
      <header className="mb-3 flex shrink-0 items-center justify-between gap-3">
        <button
          type="button"
          onClick={leaveGame}
          aria-label="Return to setup"
          className="group flex min-h-11 min-w-0 items-center gap-3 text-left"
        >
          <span className="font-display text-shimmer text-3xl font-semibold leading-none">
            ✳Star
          </span>
          <span className="hidden truncate text-xs text-muted sm:block">
            {config.mode === 'double' ? 'Double *Star' : 'Classic'} · {config.rings} rings ·{' '}
            {board.periCount + 1} points in the sky
          </span>
        </button>
        <nav className="flex shrink-0 items-center gap-1.5 sm:gap-2" aria-label="Game">
          {effectiveOver && reviewing && (
            <button
              type="button"
              onClick={() => setReviewing(false)}
              aria-label="Result"
              className={`min-h-11 items-center gap-2 rounded-xl border border-gold/60 bg-gold-faint px-3 text-sm text-gold-strong transition-colors hover:bg-gold/25 ${
                earlyOutcome?.reason === 'clinch'
                  ? 'hidden lg:flex'
                  : 'flex'
              }`}
            >
              <Trophy className="h-4 w-4" aria-hidden />
              <span className="hidden sm:inline">Result</span>
            </button>
          )}
          <button
            type="button"
            onClick={() => setRulesOpen(true)}
            aria-label="Rules"
            className="flex min-h-11 items-center gap-2 rounded-xl border border-white/15 px-3 text-sm text-ink transition-colors hover:border-gold/50"
          >
            <BookOpenText className="h-4 w-4" aria-hidden />
            <span className="hidden sm:inline">Rules</span>
          </button>
          <button
            type="button"
            onClick={leaveGame}
            aria-label="New game"
            className="flex min-h-11 items-center gap-2 rounded-xl border border-white/15 px-3 text-sm text-ink transition-colors hover:border-gold/50"
          >
            <Settings2 className="h-4 w-4" aria-hidden />
            <span className="hidden sm:inline">New game</span>
          </button>
        </nav>
      </header>

      <div className={`${styles.workspace} w-full`}>
        <BoardStage
          className={styles.boardArea}
          board={board}
          stones={proofActive ? proofScenario.stones : game.stones}
          nodeOwner={displayedScore.nodeOwner}
          aliveStone={displayedScore.aliveStone}
          provablyDeadStone={
            proofActive ? null : completionBounds?.provablyDeadStone
          }
          syntheticStone={proofMask}
          showTerritory={showTerritory}
          lastMove={proofActive ? -1 : game.lastMove}
          currentTurnMoves={proofActive ? [] : game.currentTurnMoves}
          toMove={game.toMove}
          interactive={!effectiveOver && humanCanAct}
          playerNames={config.playerNames}
          onPlace={placeStone}
          filledCount={proofActive ? board.n : game.stonesPlaced}
          focusRef={boardStageRef}
          proof={
            proofDescription && proofScenario && proofLoser !== null
              ? {
                  label: 'Clinch proof',
                  detail: `${completionBounds?.emptyNodes ?? 0} hypothetical ${
                    config.playerNames[proofLoser]
                  } stone${
                    (completionBounds?.emptyNodes ?? 0) === 1 ? '' : 's'
                  }`,
                  description: proofDescription,
                }
              : null
          }
        />

        {/* Side panel */}
        <div className={styles.sidePanel}>
          <GameStatus
            className={styles.status}
            state={gameStatusState}
            playerName={config.playerNames[statusPlayer]}
            controllerName={currentControllerName}
            mode={config.mode}
            movesLeft={game.movesLeft}
            color={activeColor}
          />
          <div className={`${styles.rail} thin-scroll flex min-w-0 flex-col gap-3`}>
            {!effectiveOver &&
              currentController !== 'human' &&
              (aiPaused || activeAiError) && (
                <section
                  aria-live="polite"
                  className="rounded-2xl border border-danger/35 bg-danger/[0.06] px-4 py-3 text-xs text-muted"
                >
                  <div className="flex items-start gap-3">
                    <CircleAlert
                      className="mt-0.5 h-4 w-4 shrink-0 text-danger"
                      aria-hidden
                    />
                    <div className="min-w-0 flex-1">
                      {aiPaused ? (
                        <p>AI is paused after history navigation.</p>
                      ) : activeAiError ? (
                        <p role="alert">
                          {activeAiError.message}{' '}
                          <span className="font-mono text-xs opacity-70">
                            ({activeAiError.code})
                          </span>
                        </p>
                      ) : null}
                    </div>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2 pl-7">
                    {aiPaused && (
                      <button
                        type="button"
                        onClick={resumeAiAction}
                        className="min-h-9 rounded-lg border border-gold/50 px-3 py-1 text-gold-strong transition-colors hover:bg-gold/15"
                      >
                        Resume AI
                      </button>
                    )}
                    {activeAiError?.retryable && (
                      <button
                        type="button"
                        onClick={() => {
                          setPublishedAnalysis(null);
                          setAiStatus({ kind: 'idle' });
                          setRetryNonce((value) => value + 1);
                        }}
                        className="min-h-9 rounded-lg border border-gold/50 px-3 py-1 text-gold-strong transition-colors hover:bg-gold/15"
                      >
                        Retry
                      </button>
                    )}
                    {canUseLessEffort && (
                      <button
                        type="button"
                        onClick={() => retryWithLessEffort(currentController)}
                        className="min-h-9 rounded-lg border border-gold/50 px-3 py-1 text-gold-strong transition-colors hover:bg-gold/15"
                      >
                        Use less effort
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => takeOverAsHuman(game.toMove, currentController)}
                      className="min-h-9 rounded-lg border border-white/20 px-3 py-1 text-ink transition-colors hover:border-gold/40"
                    >
                      Take over as human
                    </button>
                  </div>
                </section>
              )}

          {/* Pie rule offer */}
          {game.canSwap && (
            <div className="rounded-2xl border border-dashed border-gold/50 bg-gold-faint px-4 py-3 text-sm">
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
                className="mt-2 flex min-h-10 items-center gap-2 rounded-lg border border-gold/60 px-3 py-1.5 text-xs font-medium text-gold-strong transition-colors hover:bg-gold/20"
              >
                <Replace className="h-3.5 w-3.5" aria-hidden /> Steal it (swap sides)
              </button>
            </div>
          )}

          {proofScenario &&
            proofWinner !== null &&
            proofLoser !== null &&
            !game.over &&
            earlyOutcome?.reason !== 'resignation' && (
              <section
                aria-labelledby="clinch-status-heading"
                data-clinch-banner
                className="rounded-2xl border px-4 py-3.5"
                style={{
                  borderColor: `${PLAYER_COLORS[proofWinner].base}66`,
                  background: `linear-gradient(145deg, ${PLAYER_COLORS[proofWinner].soft}, rgba(16,21,42,0.9))`,
                }}
              >
                <div className="flex items-start gap-3">
                  <span
                    aria-hidden
                    className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border"
                    style={{
                      borderColor: `${PLAYER_COLORS[proofWinner].base}66`,
                      color: PLAYER_COLORS[proofWinner].bright,
                    }}
                  >
                    <ShieldCheck className="h-4.5 w-4.5" />
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <h2
                        id="clinch-status-heading"
                        className="text-sm font-semibold"
                        style={{ color: PLAYER_COLORS[proofWinner].bright }}
                      >
                        {config.playerNames[proofWinner]} has clinched
                      </h2>
                      <span className="rounded-full border border-gold/30 bg-gold-faint px-2 py-0.5 text-[0.65rem] font-semibold uppercase tracking-[0.12em] text-gold">
                        Result locked
                      </span>
                    </div>
                    <p className="mt-1 text-xs leading-relaxed text-muted">
                      Give all {completionBounds?.emptyNodes ?? 0} open node
                      {(completionBounds?.emptyNodes ?? 0) === 1 ? '' : 's'} to{' '}
                      {config.playerNames[proofLoser]} and{' '}
                      {config.playerNames[proofWinner]} still wins.
                    </p>
                  </div>
                </div>
                <div className="mt-3 flex items-center justify-between gap-3 rounded-xl border border-white/10 bg-black/15 px-3 py-2 text-xs">
                  <span className="flex min-w-0 items-center gap-2 text-muted">
                    <span
                      aria-hidden
                      className="h-4 w-4 shrink-0 rounded-full border border-dashed border-white/80 opacity-70"
                      style={{
                        background: `repeating-linear-gradient(45deg, ${PLAYER_COLORS[proofLoser].base}99 0 2px, transparent 2px 4px)`,
                      }}
                    />
                    <span className="truncate">
                      All open → {config.playerNames[proofLoser]}
                    </span>
                  </span>
                  <span className="shrink-0 font-mono tabular-nums text-ink">
                    {proofScenario.score.players[proofWinner].total}–{proofScenario.score.players[proofLoser].total}
                  </span>
                </div>
                {proofActive && (
                  <p className="mt-2 text-center text-xs font-medium text-gold">
                    Proof board active · striped stones are hypothetical
                  </p>
                )}
              </section>
            )}

          <ScorePanel
            game={game}
            score={displayedScore}
            completionBounds={completionBounds}
            view={scoreView}
          />

            {devtools && publishedAnalysis && (
              <EngineEstimatePanel
                analysis={publishedAnalysis.analysis}
                board={board}
                playerNames={config.playerNames}
              />
            )}

            <details className="rounded-xl border border-white/10 bg-white/[0.025] px-3">
              <summary className="flex min-h-11 cursor-pointer items-center text-xs font-medium text-muted">
                About the scoring display
              </summary>
              <p className="pb-3 text-xs leading-relaxed text-muted">
                Influence shows the current scoring projection and dims groups that are not yet
                stars. Crosses mark only groups that cannot become a star even if they received
                every open node. A striped proof stone is hypothetical and never changes the
                actual move history.
              </p>
            </details>
          </div>

          {/* Persistent actions */}
          <div
            className={`${styles.actions} panel-surface grid grid-cols-2 gap-2 rounded-2xl p-2`}
            data-action-dock
          >
            <button
              type="button"
              disabled={
                log.length === 0 ||
                earlyOutcome !== null ||
                proofActive ||
                clinchPending
              }
              onClick={undoAction}
              className="flex min-h-11 items-center justify-center gap-2 rounded-xl border border-white/15 px-3 py-2 text-sm text-ink transition-colors enabled:hover:border-gold/50 disabled:opacity-35"
            >
              <Undo2 className="h-4 w-4" aria-hidden /> Undo
            </button>
            <button
              type="button"
              disabled={
                redoStack.length === 0 ||
                earlyOutcome !== null ||
                proofActive ||
                clinchPending
              }
              onClick={redoAction}
              className="flex min-h-11 items-center justify-center gap-2 rounded-xl border border-white/15 px-3 py-2 text-sm text-ink transition-colors enabled:hover:border-gold/50 disabled:opacity-35"
            >
              <Redo2 className="h-4 w-4" aria-hidden /> Redo
            </button>

            <label className="col-span-2 flex min-h-11 cursor-pointer items-center justify-between rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2">
              <span className="flex items-center gap-2 text-sm text-ink">
                <Eye className="h-4 w-4 text-muted" aria-hidden /> Show influence
              </span>
              <input
                type="checkbox"
                checked={showTerritory}
                disabled={game.over || proofActive}
                onChange={(e) => setShowInfluence(e.target.checked)}
                className="h-4 w-4 accent-[#e8c48b]"
              />
            </label>

            {proofScenario &&
              proofWinner !== null &&
              proofLoser !== null &&
              !game.over &&
              earlyOutcome?.reason !== 'resignation' && (
                <button
                  type="button"
                  aria-pressed={proofActive}
                  onClick={() => {
                    if (proofActive) setProofMode(false);
                    else showProofBoard();
                  }}
                  className={`col-span-2 min-h-11 items-center justify-between gap-3 rounded-xl border px-3 py-2 text-left text-sm transition-colors ${
                    proofActive
                      ? 'border-gold/60 bg-gold-faint text-gold-strong'
                      : 'border-white/15 bg-white/[0.03] text-ink hover:border-gold/50'
                  } hidden lg:flex`}
                >
                  <span className="flex items-center gap-2">
                    <ShieldCheck className="h-4 w-4" aria-hidden />
                    {proofActive ? 'Return to live board' : 'Show proof board'}
                  </span>
                  <span className="text-xs text-muted">
                    all open → {config.playerNames[proofLoser]}
                  </span>
                </button>
              )}

            {clinchPending && proofActive ? (
              <>
                <button
                  type="button"
                  onClick={continueAfterClinch}
                  className="hidden min-h-11 items-center justify-center rounded-xl border border-white/15 px-3 py-2 text-sm text-ink transition-colors hover:border-gold/50 lg:flex"
                >
                  Continue playing
                </button>
                <button
                  type="button"
                  onClick={endClinchNow}
                  className="hidden min-h-11 items-center justify-center gap-2 rounded-xl border border-gold/60 bg-gold-faint px-3 py-2 text-sm font-medium text-gold-strong transition-colors hover:bg-gold/25 lg:flex"
                >
                  <Trophy className="h-4 w-4" aria-hidden /> End now
                </button>
              </>
            ) : guaranteedWinner !== null && earlyOutcome === null ? (
              <button
                type="button"
                onClick={() => {
                  cancelActiveAi();
                  setEndConfirmOpen(true);
                }}
                className="col-span-2 hidden min-h-11 items-center justify-center gap-2 rounded-xl border border-gold/60 bg-gold-faint px-3 py-2 text-sm font-medium text-gold-strong transition-colors hover:bg-gold/25 lg:flex"
              >
                <Trophy className="h-4 w-4" aria-hidden /> End game
              </button>
            ) : (
              resigningPlayer !== null &&
              !effectiveOver && (
                <button
                  type="button"
                  onClick={() => {
                    cancelActiveAi();
                    setResignConfirmOpen(true);
                  }}
                  className="col-span-2 flex min-h-11 items-center justify-center gap-2 rounded-xl border border-danger/35 px-3 py-2 text-sm text-danger transition-colors hover:border-danger/60 hover:bg-danger/[0.07]"
                >
                  <Flag className="h-4 w-4" aria-hidden /> Resign{' '}
                  {config.playerNames[resigningPlayer]}
                </button>
              )
            )}
          </div>
        </div>
      </div>

      {proofScenario &&
        proofWinner !== null &&
        proofLoser !== null &&
        !game.over &&
        earlyOutcome?.reason !== 'resignation' &&
        (!gameResult || reviewing) &&
        (!clinchPending || proofActive) && (
          <div
            className={`panel-surface fixed z-30 grid gap-2 rounded-2xl p-2 shadow-[0_18px_70px_rgba(0,0,0,0.55)] lg:hidden ${
              clinchPending && proofActive ? 'grid-cols-3' : 'grid-cols-2'
            }`}
            style={{
              left: 'max(0.5rem, env(safe-area-inset-left))',
              right: 'max(0.5rem, env(safe-area-inset-right))',
              bottom: 'max(0.5rem, env(safe-area-inset-bottom))',
            }}
            data-mobile-clinch-controls
            role="group"
            aria-label="Clinched game controls"
          >
            <button
              type="button"
              aria-pressed={proofActive}
              onClick={() => {
                if (proofActive) setProofMode(false);
                else showProofBoard();
              }}
              className="flex min-h-11 items-center justify-center gap-1.5 rounded-xl border border-white/15 px-2 py-2 text-xs text-ink transition-colors hover:border-gold/50"
            >
              <ShieldCheck className="h-4 w-4" aria-hidden />
              {proofActive ? 'Live board' : 'Show proof'}
            </button>
            {clinchPending && proofActive ? (
              <>
                <button
                  type="button"
                  onClick={continueAfterClinch}
                  className="min-h-11 rounded-xl border border-white/15 px-2 py-2 text-xs text-ink transition-colors hover:border-gold/50"
                >
                  Continue
                </button>
                <button
                  type="button"
                  onClick={endClinchNow}
                  className="min-h-11 rounded-xl border border-gold/60 bg-gold-faint px-2 py-2 text-xs font-medium text-gold-strong"
                >
                  End now
                </button>
              </>
            ) : earlyOutcome?.reason === 'clinch' ? (
              <button
                type="button"
                onClick={() => setReviewing(false)}
                className="flex min-h-11 items-center justify-center gap-1.5 rounded-xl border border-gold/60 bg-gold-faint px-2 py-2 text-xs font-medium text-gold-strong"
              >
                <Trophy className="h-4 w-4" aria-hidden /> Result
              </button>
            ) : (
              <button
                type="button"
                onClick={() => {
                  cancelActiveAi();
                  setEndConfirmOpen(true);
                }}
                className="flex min-h-11 items-center justify-center gap-1.5 rounded-xl border border-gold/60 bg-gold-faint px-2 py-2 text-xs font-medium text-gold-strong"
              >
                <Trophy className="h-4 w-4" aria-hidden /> End game
              </button>
            )}
          </div>
        )}

      <RulesDialog
        open={
          rulesOpen &&
          !clinchPending &&
          !validEndConfirmOpen &&
          !validResignConfirmOpen &&
          (!gameResult || reviewing)
        }
        onClose={() => setRulesOpen(false)}
      />
      {proofScenario && proofWinner !== null && proofLoser !== null && (
        <ClinchDialog
          open={clinchPending && !proofActive}
          winner={proofWinner}
          winnerName={config.playerNames[proofWinner]}
          loserName={config.playerNames[proofLoser]}
          emptyNodes={completionBounds?.emptyNodes ?? 0}
          returnFocusRef={boardStageRef}
          proofScores={[
            proofScenario.score.players[0].total,
            proofScenario.score.players[1].total,
          ]}
          onContinue={continueAfterClinch}
          onProof={showProofBoard}
          onEnd={endClinchNow}
        />
      )}
      {proofWinner !== null && (
        <EndGameConfirmDialog
          open={validEndConfirmOpen}
          winnerName={config.playerNames[proofWinner]}
          emptyNodes={completionBounds?.emptyNodes ?? 0}
          onCancel={() => setEndConfirmOpen(false)}
          onConfirm={endClinchNow}
        />
      )}
      {resigningPlayer !== null && (
        <ResignDialog
          open={validResignConfirmOpen}
          loserName={config.playerNames[resigningPlayer]}
          winnerName={config.playerNames[1 - resigningPlayer]}
          onCancel={() => setResignConfirmOpen(false)}
          onConfirm={confirmResignation}
        />
      )}
      {gameResult && (
        <GameOverOverlay
          open={!reviewing}
          game={game}
          result={gameResult}
          returnFocusRef={boardStageRef}
          onReview={() => {
            setRulesOpen(false);
            setReviewing(true);
            if (gameResult.reason === 'clinch') showProofBoard();
            else setProofMode(false);
          }}
          onRematch={rematchAction}
          onSetup={leaveGame}
        />
      )}
    </main>
  );
}
