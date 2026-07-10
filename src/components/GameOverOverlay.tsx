'use client';

import { Eye, RotateCcw, Settings2, Trophy } from 'lucide-react';
import type { GameState } from '@/lib/star/game';
import type { ScoreResult } from '@/lib/star/scoring';
import { PLAYER_COLORS } from './theme';

interface GameOverOverlayProps {
  open: boolean;
  game: GameState;
  score: ScoreResult;
  onReview: () => void;
  onRematch: () => void;
  onSetup: () => void;
}

const delay = (step: number) => ({ animationDelay: `${0.25 + step * 0.18}s` });

export function GameOverOverlay({
  open,
  game,
  score,
  onReview,
  onRematch,
  onSetup,
}: GameOverOverlayProps) {
  if (!open) return null;

  const winner = score.leader;
  // Winner despite equal totals: the quark tie-break decided it.
  const quarkTieBreak =
    winner !== -1 && score.players[0].total === score.players[1].total
      ? ([score.players[winner].quarks, score.players[winner === 0 ? 1 : 0].quarks] as const)
      : null;

  return (
    <div
      className="fade-in fixed inset-0 z-40 flex items-center justify-center bg-black/70 p-4 backdrop-blur-md"
      role="dialog"
      aria-modal
      aria-label="Game over"
    >
      <div className="fade-up w-full max-w-lg rounded-3xl border border-gold/40 bg-night-raise p-8 text-center shadow-[0_0_80px_rgba(232,196,139,0.18)]">
        <div
          className="pop-in mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-full border border-gold/50 bg-gold-faint"
          style={delay(0)}
        >
          <Trophy className="h-7 w-7 text-gold-strong" aria-hidden />
        </div>

        <p className="fade-in text-xs uppercase tracking-[0.35em] text-muted" style={delay(1)}>
          the sky is settled
        </p>
        <h2 className="font-display fade-up mt-1 text-4xl text-ink" style={delay(1)}>
          {winner === -1 ? (
            'A perfect tie'
          ) : (
            <>
              <span style={{ color: PLAYER_COLORS[winner].base }}>
                {game.config.playerNames[winner]}
              </span>{' '}
              wins
            </>
          )}
        </h2>
        {quarkTieBreak && (
          <p className="fade-in mt-1 text-sm text-muted" style={delay(2)}>
            equal totals — decided on quarks, {quarkTieBreak[0]} to {quarkTieBreak[1]}
          </p>
        )}

        <div className="mt-6 grid grid-cols-2 gap-3">
          {([0, 1] as const).map((p) => {
            const s = score.players[p];
            const c = PLAYER_COLORS[p];
            const won = winner === p;
            return (
              <div
                key={p}
                className="fade-up rounded-2xl border p-4 text-left"
                style={{
                  ...delay(2),
                  borderColor: won ? c.base : 'rgba(255,255,255,0.10)',
                  background: won ? c.soft : 'rgba(255,255,255,0.03)',
                }}
              >
                <div className="truncate text-xs text-muted">{game.config.playerNames[p]}</div>
                <div
                  className="font-display pop-in text-5xl tabular-nums"
                  style={{ ...delay(3), color: c.bright }}
                >
                  {s.total}
                </div>
                <dl className="fade-in mt-2 space-y-0.5 text-[11px] text-muted" style={delay(4)}>
                  <div className="flex justify-between">
                    <dt>peries</dt>
                    <dd className="text-ink/90">{s.peries}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt>quark peri</dt>
                    <dd className="text-ink/90">{s.quarkPeri ? '+1' : '0'}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt>
                      award ({s.stars} star{s.stars === 1 ? '' : 's'})
                    </dt>
                    <dd className="text-ink/90">{s.award > 0 ? `+${s.award}` : s.award}</dd>
                  </div>
                </dl>
              </div>
            );
          })}
        </div>

        <div className="fade-in mt-7 flex flex-wrap justify-center gap-2.5" style={delay(5)}>
          <button
            type="button"
            onClick={onReview}
            className="flex items-center gap-2 rounded-xl border border-white/15 px-4 py-2.5 text-sm text-ink transition-colors hover:border-gold/50"
          >
            <Eye className="h-4 w-4" aria-hidden /> Review board
          </button>
          <button
            type="button"
            onClick={onRematch}
            className="flex items-center gap-2 rounded-xl border border-gold/60 bg-gold-faint px-4 py-2.5 text-sm font-medium text-gold-strong transition-colors hover:bg-gold/25"
          >
            <RotateCcw className="h-4 w-4" aria-hidden /> Rematch
          </button>
          <button
            type="button"
            onClick={onSetup}
            className="flex items-center gap-2 rounded-xl border border-white/15 px-4 py-2.5 text-sm text-ink transition-colors hover:border-gold/50"
          >
            <Settings2 className="h-4 w-4" aria-hidden /> New setup
          </button>
        </div>
      </div>
    </div>
  );
}
