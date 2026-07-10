'use client';

import type { GameState } from '@/lib/star/game';
import type { ScoreResult } from '@/lib/star/scoring';
import { PLAYER_COLORS } from './theme';

interface ScorePanelProps {
  game: GameState;
  score: ScoreResult;
}

function Row({ label, value, accent }: { label: string; value: string | number; accent?: boolean }) {
  return (
    <div className="flex items-baseline justify-between text-sm">
      <span className="text-muted">{label}</span>
      <span className={accent ? 'text-gold' : 'text-ink'}>{value}</span>
    </div>
  );
}

export function ScorePanel({ game, score }: ScorePanelProps) {
  const { config } = game;

  return (
    <div className="flex flex-col gap-3">
      {([0, 1] as const).map((p) => {
        const s = score.players[p];
        const active = !game.over && game.toMove === p;
        const c = PLAYER_COLORS[p];
        return (
          <section
            key={p}
            className="rounded-2xl border bg-white/[0.03] px-4 py-3.5 backdrop-blur-sm transition-all duration-300"
            style={{
              borderColor: active ? c.base : 'rgba(255,255,255,0.10)',
              boxShadow: active ? `0 0 32px ${c.soft}` : '0 0 0 rgba(0,0,0,0)',
            }}
          >
            <header className="mb-2.5 flex items-center gap-2.5">
              <span
                aria-hidden
                className="h-5 w-5 rounded-full"
                style={{
                  background: `radial-gradient(circle at 35% 30%, ${c.bright}, ${c.base} 55%, ${c.deep})`,
                }}
              />
              <h3 className="min-w-0 flex-1 truncate text-sm font-medium text-ink">
                {config.playerNames[p]}
              </h3>
              {active && (
                <span
                  className="rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-widest"
                  style={{ background: c.soft, color: c.bright }}
                >
                  {config.mode === 'double' && !game.over
                    ? `to play · ${game.movesLeft} stone${game.movesLeft > 1 ? 's' : ''}`
                    : 'to play'}
                </span>
              )}
              <span className="font-display text-3xl leading-none text-ink tabular-nums">
                {s.total}
              </span>
            </header>
            <div className="grid grid-cols-2 gap-x-5 gap-y-0.5">
              <Row label="Peries" value={s.peries} />
              <Row label="Quarks" value={`${s.quarks} / 5`} />
              <Row label="Stars" value={s.stars} />
              <Row label="Quark peri" value={s.quarkPeri ? '+1' : '—'} accent={s.quarkPeri === 1} />
              <Row
                label="Star award"
                value={s.award > 0 ? `+${s.award}` : s.award}
                accent={s.award > 0}
              />
            </div>
          </section>
        );
      })}

      <p className="px-1 text-center text-[11px] leading-relaxed text-muted">
        {score.contestedPeries > 0 ? (
          <>
            <span className="text-gold">{score.contestedPeries}</span> per
            {score.contestedPeries === 1 ? 'i is' : 'ies are'} still contested · totals reach{' '}
            <span className="text-gold">{game.board.periCount + 1}</span> when decided
          </>
        ) : (
          <>
            every peri is claimed — totals sum to{' '}
            <span className="text-gold">{game.board.periCount + 1}</span>
          </>
        )}
      </p>
    </div>
  );
}
