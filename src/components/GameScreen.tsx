'use client';

import { useEffect, useMemo, useState } from 'react';
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
import { replay } from '@/lib/star/game';
import { scorePosition } from '@/lib/star/scoring';
import { useAppStore } from '@/lib/store';
import { GameOverOverlay } from './GameOverOverlay';
import { RulesDialog } from './RulesDialog';
import { ScorePanel } from './ScorePanel';
import { StarBoard } from './StarBoard';
import { PLAYER_COLORS } from './theme';

export function GameScreen() {
  const { config, log, redoStack, reviewing } = useAppStore();
  const { act, undo, redo, rematch, toSetup, setReviewing } = useAppStore();

  const [rulesOpen, setRulesOpen] = useState(false);
  const [showInfluence, setShowInfluence] = useState(false);
  const [hoverNode, setHoverNode] = useState(-1);

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

  const score = useMemo(
    () => (game ? scorePosition(game.board, game.stones) : null),
    [game],
  );

  if (!game || !score) return null;

  const { board } = game;
  const showTerritory = game.over || showInfluence;
  const activeColor = PLAYER_COLORS[game.toMove];

  return (
    <main className="relative z-10 mx-auto flex w-full max-w-7xl flex-1 flex-col px-4 py-5 sm:px-6">
      <header className="mb-4 flex items-center justify-between gap-3">
        <button type="button" onClick={toSetup} className="group flex items-baseline gap-3 text-left">
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
            onClick={toSetup}
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
            interactive={!game.over}
            onPlace={(node) => act({ type: 'place', node })}
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

          {/* Pie rule offer */}
          {game.canSwap && (
            <div className="fade-up rounded-2xl border border-dashed border-gold/50 bg-gold-faint px-4 py-3 text-sm">
              <p className="text-ink">
                Pie rule — {config.playerNames[1]} may steal the opening stone.
              </p>
              <button
                type="button"
                onClick={() => act({ type: 'swap' })}
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
              disabled={game.over}
              onClick={() => act({ type: 'pass' })}
              className="flex items-center justify-center gap-2 rounded-xl border border-white/15 px-3 py-2.5 text-sm text-ink transition-colors enabled:hover:border-danger/60 enabled:hover:text-danger disabled:opacity-35"
            >
              <Flag className="h-4 w-4" aria-hidden /> Pass
            </button>
            <button
              type="button"
              disabled={log.length === 0}
              onClick={undo}
              className="flex items-center justify-center gap-2 rounded-xl border border-white/15 px-3 py-2.5 text-sm text-ink transition-colors enabled:hover:border-gold/50 disabled:opacity-35"
            >
              <Undo2 className="h-4 w-4" aria-hidden /> Undo
            </button>
            <button
              type="button"
              disabled={redoStack.length === 0}
              onClick={redo}
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
        onRematch={rematch}
        onSetup={toSetup}
      />
    </main>
  );
}
