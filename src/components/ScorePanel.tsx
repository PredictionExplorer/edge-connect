'use client';

import type { GameState } from '@/lib/star/game';
import type { ScoreResult } from '@/lib/star/scoring';
import type { CompletionBounds } from '@/lib/star/completion-bounds';
import { PLAYER_COLORS } from './theme';

interface ScorePanelProps {
  game: GameState;
  score: ScoreResult;
  completionBounds?: CompletionBounds | null;
}

function Row({ label, value, accent }: { label: string; value: string | number; accent?: boolean }) {
  return (
    <div className="flex items-baseline justify-between text-sm">
      <span className="text-muted">{label}</span>
      <span className={accent ? 'text-gold' : 'text-ink'}>{value}</span>
    </div>
  );
}

function CompletionForecast({
  game,
  bounds,
}: {
  game: GameState;
  bounds: CompletionBounds;
}) {
  const guaranteed = bounds.guaranteedWinner;

  return (
    <section
      aria-labelledby="completion-forecast-heading"
      className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3.5"
    >
      <header className="flex items-center justify-between gap-3">
        <h3
          id="completion-forecast-heading"
          className="text-xs font-semibold uppercase tracking-[0.18em] text-muted"
        >
          Completion bounds
        </h3>
        <span className="text-[10px] tabular-nums text-muted">
          {bounds.emptyNodes} open
        </span>
      </header>
      <p className="mt-1 text-[11px] leading-relaxed text-muted">
        Final scores if every open node went to one side.
      </p>

      <div className="mt-3 grid grid-cols-[minmax(0,1fr)_3.6rem_3.6rem] items-center gap-x-2 gap-y-1.5 text-xs">
        <span aria-hidden />
        {([0, 1] as const).map((player) => (
          <span
            key={player}
            className="truncate text-center text-[10px] font-medium"
            style={{ color: PLAYER_COLORS[player].bright }}
            title={game.config.playerNames[player]}
          >
            {game.config.playerNames[player]}
          </span>
        ))}
        {bounds.scenarios.map((scenario) => (
          <div className="contents" key={scenario.fillPlayer}>
            <span className="flex min-w-0 items-center gap-1.5 text-muted">
              <span
                aria-hidden
                className="h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ background: PLAYER_COLORS[scenario.fillPlayer].base }}
              />
              <span className="truncate">
                All open → {game.config.playerNames[scenario.fillPlayer]}
              </span>
            </span>
            {([0, 1] as const).map((player) => (
              <span
                key={player}
                className="rounded-md border py-1 text-center font-mono tabular-nums"
                style={{
                  borderColor:
                    scenario.winner === player
                      ? PLAYER_COLORS[player].base
                      : 'rgba(255,255,255,0.08)',
                  color:
                    scenario.winner === player
                      ? PLAYER_COLORS[player].bright
                      : 'var(--muted)',
                  background:
                    scenario.winner === player
                      ? PLAYER_COLORS[player].soft
                      : 'rgba(255,255,255,0.02)',
                }}
                aria-label={`${game.config.playerNames[player]} scores ${scenario.score.players[player].total}`}
              >
                {scenario.score.players[player].total}
              </span>
            ))}
          </div>
        ))}
      </div>

      <p
        className="mt-3 border-t border-white/10 pt-2 text-center text-[11px] leading-relaxed"
        style={{
          color:
            guaranteed === null
              ? 'var(--muted)'
              : PLAYER_COLORS[guaranteed].bright,
        }}
      >
        {guaranteed === null
          ? 'The final winner is not clinched yet.'
          : `${game.config.playerNames[guaranteed]} has clinched the game.`}
      </p>
    </section>
  );
}

export function ScorePanel({ game, score, completionBounds }: ScorePanelProps) {
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
              <h2 className="min-w-0 flex-1 truncate text-sm font-medium text-ink">
                {config.playerNames[p]}
              </h2>
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

      {!game.over && completionBounds && (
        <CompletionForecast game={game} bounds={completionBounds} />
      )}
    </div>
  );
}
