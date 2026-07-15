'use client';

import { useRef, type RefObject } from 'react';
import {
  Eye,
  Flag,
  RotateCcw,
  Settings2,
  ShieldCheck,
  Trophy,
} from 'lucide-react';
import type { GameState } from '@/lib/star/game';
import type { ScoreResult } from '@/lib/star/scoring';
import type { EarlyGameOutcome } from '@/lib/store';
import { ModalDialog } from './ModalDialog';
import { PLAYER_COLORS } from './theme';

export type GameResult =
  | {
      reason: 'full-board';
      winner: 0 | 1;
      score: ScoreResult;
    }
  | EarlyGameOutcome;

interface GameOverOverlayProps {
  open: boolean;
  game: GameState;
  result: GameResult;
  returnFocusRef?: RefObject<HTMLElement | null>;
  onReview: () => void;
  onRematch: () => void;
  onSetup: () => void;
}

const delay = () => ({ animationDelay: '0s' });

export function GameOverOverlay({
  open,
  game,
  result,
  returnFocusRef,
  onReview,
  onRematch,
  onSetup,
}: GameOverOverlayProps) {
  const rematchButton = useRef<HTMLButtonElement>(null);
  const { winner } = result;
  const eyebrow =
    result.reason === 'full-board'
      ? 'the sky is settled'
      : result.reason === 'clinch'
        ? 'result clinched'
        : 'by resignation';

  return (
    <ModalDialog
      open={open}
      onClose={onReview}
      ariaLabel="Game over"
      initialFocusRef={rematchButton}
      returnFocusRef={returnFocusRef}
      closeOnBackdrop={false}
      className="max-w-lg"
    >
      <div className="thin-scroll panel-surface max-h-[calc(100dvh-2rem)] w-full overflow-y-auto rounded-3xl p-5 text-center shadow-[0_0_80px_rgba(232,196,139,0.18)] sm:p-8">
        <div
          className="pop-in mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-full border border-gold/50 bg-gold-faint"
          style={delay()}
        >
          <Trophy className="h-7 w-7 text-gold-strong" aria-hidden />
        </div>

        <p className="fade-in text-xs uppercase tracking-[0.16em] text-muted" style={delay()}>
          {eyebrow}
        </p>
        <h2 className="font-display fade-up mt-1 text-4xl text-ink" style={delay()}>
          <span style={{ color: PLAYER_COLORS[winner].base }}>
            {game.config.playerNames[winner]}
          </span>{' '}
          wins
        </h2>

        {result.reason === 'full-board' ? (
          <div className="mt-6 grid grid-cols-2 gap-3">
            {([0, 1] as const).map((p) => {
              const s = result.score.players[p];
              const c = PLAYER_COLORS[p];
              const won = winner === p;
              return (
                <div
                  key={p}
                  className="fade-up rounded-2xl border p-4 text-left"
                  style={{
                    ...delay(),
                    borderColor: won ? c.base : 'rgba(255,255,255,0.10)',
                    background: won ? c.soft : 'rgba(255,255,255,0.03)',
                  }}
                >
                  <div className="truncate text-xs text-muted">
                    {game.config.playerNames[p]}
                  </div>
                  <div
                    className="font-display pop-in text-5xl tabular-nums"
                    style={{ ...delay(), color: c.bright }}
                  >
                    {s.total}
                  </div>
                  <dl
                    className="fade-in mt-2 space-y-0.5 text-xs text-muted"
                    style={delay()}
                  >
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
                      <dd className="text-ink/90">
                        {s.award > 0 ? `+${s.award}` : s.award}
                      </dd>
                    </div>
                  </dl>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="fade-up mt-6 rounded-2xl border border-white/10 bg-white/[0.035] p-5 text-left">
            <div className="flex items-start gap-3">
              <span
                className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-gold/35 bg-gold-faint text-gold-strong"
                aria-hidden
              >
                {result.reason === 'clinch' ? (
                  <ShieldCheck className="h-5 w-5" />
                ) : (
                  <Flag className="h-5 w-5" />
                )}
              </span>
              <div>
                <p className="text-sm font-medium text-ink">
                  {result.reason === 'clinch'
                    ? `The result was guaranteed with ${result.emptyNodes} open node${
                        result.emptyNodes === 1 ? '' : 's'
                      }.`
                    : `${game.config.playerNames[result.loser]} resigned.`}
                </p>
                <p className="mt-1 text-xs leading-relaxed text-muted">
                  No final score was recorded because the board was not filled.
                  {result.reason === 'clinch'
                    ? ' Review the proof to see the strongest possible completion for the losing player.'
                    : ''}
                </p>
              </div>
            </div>
          </div>
        )}

        <div
          className="sticky bottom-0 z-10 -mx-2 mt-5 flex flex-wrap justify-center gap-2.5 rounded-2xl bg-night-surface-strong/95 px-2 py-3 backdrop-blur-md"
          style={delay()}
        >
          <button
            type="button"
            onClick={onReview}
            className="flex min-h-11 items-center gap-2 rounded-xl border border-white/15 px-4 py-2 text-sm text-ink transition-colors hover:border-gold/50"
          >
            <Eye className="h-4 w-4" aria-hidden />{' '}
            {result.reason === 'clinch' ? 'Review proof' : 'Review board'}
          </button>
          <button
            ref={rematchButton}
            type="button"
            onClick={onRematch}
            className="flex min-h-11 items-center gap-2 rounded-xl border border-gold/60 bg-gold-faint px-4 py-2 text-sm font-medium text-gold-strong transition-colors hover:bg-gold/25"
          >
            <RotateCcw className="h-4 w-4" aria-hidden /> Rematch
          </button>
          <button
            type="button"
            onClick={onSetup}
            className="flex min-h-11 items-center gap-2 rounded-xl border border-white/15 px-4 py-2 text-sm text-ink transition-colors hover:border-gold/50"
          >
            <Settings2 className="h-4 w-4" aria-hidden /> New setup
          </button>
        </div>
      </div>
    </ModalDialog>
  );
}
