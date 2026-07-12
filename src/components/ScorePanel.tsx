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

interface ScoreRow {
  label: string;
  values: readonly (string | number)[];
  accents?: readonly boolean[];
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
      className="rounded-2xl border border-white/10 bg-white/[0.025] px-4 py-3.5"
    >
      <header className="flex items-center justify-between gap-3">
        <h3
          id="completion-forecast-heading"
          className="text-xs font-semibold uppercase tracking-[0.14em] text-muted"
        >
          Completion bounds
        </h3>
        <span className="text-xs tabular-nums text-muted">
          {bounds.emptyNodes} open
        </span>
      </header>
      <p className="mt-1 text-xs leading-relaxed text-muted">
        Final scores if every open node went to one side.
      </p>

      <div className="mt-3 grid grid-cols-[minmax(0,1fr)_3.75rem_3.75rem] items-center gap-x-2 gap-y-1.5 text-xs">
        <span aria-hidden />
        {([0, 1] as const).map((player) => (
          <span
            key={player}
            className="truncate text-center text-xs font-medium"
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
        className="mt-3 min-h-10 border-t border-white/10 pt-2 text-center text-xs leading-relaxed"
        style={{
          color:
            guaranteed === null
              ? 'var(--muted)'
              : PLAYER_COLORS[guaranteed].bright,
        }}
      >
        {guaranteed === null ? (
          'The final winner is not clinched yet.'
        ) : (
          <>
            <span className="block font-medium">
              {game.config.playerNames[guaranteed]} has clinched the game
            </span>
            <span className="mt-0.5 block text-muted">
              Even if every open node went to{' '}
              {game.config.playerNames[1 - guaranteed]}, the result would not change.
            </span>
          </>
        )}
      </p>
    </section>
  );
}

export function ScorePanel({ game, score, completionBounds }: ScorePanelProps) {
  const { config } = game;
  const rows: ScoreRow[] = [
    {
      label: 'Peries',
      values: [score.players[0].peries, score.players[1].peries],
    },
    {
      label: 'Quarks',
      values: [`${score.players[0].quarks} / 5`, `${score.players[1].quarks} / 5`],
    },
    {
      label: 'Stars',
      values: [score.players[0].stars, score.players[1].stars],
    },
    {
      label: 'Quark peri',
      values: [
        score.players[0].quarkPeri ? '+1' : '—',
        score.players[1].quarkPeri ? '+1' : '—',
      ],
      accents: [score.players[0].quarkPeri === 1, score.players[1].quarkPeri === 1],
    },
    {
      label: 'Star award',
      values: [
        score.players[0].award > 0
          ? `+${score.players[0].award}`
          : score.players[0].award,
        score.players[1].award > 0
          ? `+${score.players[1].award}`
          : score.players[1].award,
      ],
      accents: [score.players[0].award > 0, score.players[1].award > 0],
    },
  ];

  return (
    <div className="flex flex-col gap-3">
      <div className="flex min-h-5 items-center justify-between px-1 text-xs font-semibold uppercase tracking-[0.14em] text-muted">
        <span>{game.over ? 'Final score' : 'Current scoring projection'}</span>
        {!game.over && <span className="normal-case tracking-normal">can change</span>}
      </div>

      <section
        aria-label={game.over ? 'Final player scores' : 'Current player scores'}
        className="overflow-hidden rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-sm"
      >
        <div className="grid grid-cols-[minmax(5.5rem,1fr)_minmax(4rem,5.25rem)_minmax(4rem,5.25rem)] items-stretch border-b border-white/10">
          <div className="flex items-end px-3 py-3 text-xs text-muted">Score</div>
          {([0, 1] as const).map((player) => {
            const active = !game.over && game.toMove === player;
            const color = PLAYER_COLORS[player];
            return (
              <div
                key={player}
                className="min-w-0 border-l border-white/10 px-2 py-2.5 text-center transition-[background-color,box-shadow] duration-200"
                style={{
                  background: active ? color.soft : 'transparent',
                  boxShadow: active ? `inset 0 -2px 0 ${color.base}` : 'none',
                }}
              >
                <span
                  aria-hidden
                  className="mx-auto mb-1 block h-2.5 w-2.5 rounded-full"
                  style={{
                    background: `radial-gradient(circle at 35% 30%, ${color.bright}, ${color.base} 60%, ${color.deep})`,
                  }}
                />
                <h2
                  className="truncate text-xs font-medium"
                  style={{ color: color.bright }}
                  title={config.playerNames[player]}
                >
                  {config.playerNames[player]}
                </h2>
                <span className="font-display mt-0.5 block text-3xl leading-none text-ink tabular-nums">
                  {score.players[player].total}
                </span>
                <span className="sr-only">
                  {active ? ', currently to play' : ''}
                </span>
              </div>
            );
          })}
        </div>
        <div className="divide-y divide-white/[0.07]">
          {rows.map((row) => (
            <div
              key={row.label}
              className="grid min-h-8 grid-cols-[minmax(5.5rem,1fr)_minmax(4rem,5.25rem)_minmax(4rem,5.25rem)] items-center text-xs"
            >
              <span className="px-3 text-muted">{row.label}</span>
              {([0, 1] as const).map((player) => (
                <span
                  key={player}
                  className={`border-l border-white/[0.07] px-2 text-center tabular-nums ${
                    row.accents?.[player] ? 'text-gold' : 'text-ink'
                  }`}
                >
                  {row.values[player]}
                </span>
              ))}
            </div>
          ))}
        </div>
      </section>

      <p className="min-h-8 px-1 text-center text-xs leading-relaxed text-muted">
        {score.contestedPeries > 0 ? (
          <>
            <span className="text-gold">{score.contestedPeries}</span> per
            {score.contestedPeries === 1 ? 'i is' : 'ies are'} still contested · totals reach{' '}
            <span className="text-gold">{game.board.periCount + 1}</span> when decided
          </>
        ) : game.over ? (
          <>
            every peri is claimed — totals sum to{' '}
            <span className="text-gold">{game.board.periCount + 1}</span>
          </>
        ) : (
          <>the current projection assigns every peri — play can still change it</>
        )}
      </p>

      {!game.over && completionBounds && (
        <CompletionForecast game={game} bounds={completionBounds} />
      )}
    </div>
  );
}
