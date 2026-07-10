'use client';

import { useMemo, useState } from 'react';
import { Sparkles, Star, Users } from 'lucide-react';
import { getBoard, MAX_RINGS, MIN_RINGS } from '@/lib/star/board';
import { EMPTY } from '@/lib/star/scoring';
import type { GameConfig, Mode } from '@/lib/star/game';
import { useAppStore } from '@/lib/store';
import { StarBoard } from './StarBoard';
import { BOARD_PRESETS, PLAYER_COLORS } from './theme';

export function SetupScreen() {
  const startGame = useAppStore((s) => s.startGame);
  const lastConfig = useAppStore((s) => s.config);

  const [mode, setMode] = useState<Mode>(lastConfig.mode);
  const [rings, setRings] = useState(lastConfig.rings);
  const [pieRule, setPieRule] = useState(lastConfig.pieRule);
  const [names, setNames] = useState<[string, string]>([...lastConfig.playerNames]);

  const board = useMemo(() => getBoard(rings), [rings]);
  const emptyStones = useMemo(() => new Int8Array(board.n).fill(EMPTY), [board]);

  const start = () => {
    const config: GameConfig = {
      rings,
      mode,
      pieRule,
      playerNames: [names[0].trim() || 'Player 1', names[1].trim() || 'Player 2'],
    };
    startGame(config);
  };

  const preset = BOARD_PRESETS.find((p) => p.rings === rings);

  return (
    <main className="relative z-10 mx-auto flex w-full max-w-6xl flex-1 flex-col items-center px-6 py-10">
      <header className="fade-up text-center">
        <p className="mb-2 flex items-center justify-center gap-2 text-sm uppercase tracking-[0.35em] text-muted">
          <Sparkles className="h-4 w-4 text-gold" aria-hidden />
          a connection game by Ea Ea
        </p>
        <h1 className="font-display text-shimmer text-7xl font-semibold leading-none sm:text-8xl">
          ✳Star
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-balance text-sm leading-relaxed text-muted">
          Claim the edge, join your stars through the heavens, and let no light go to waste.
          Two players, one sky.
        </p>
      </header>

      <div className="mt-10 grid w-full gap-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,4fr)]">
        {/* Board preview */}
        <div
          className="pop-in relative mx-auto w-full max-w-lg self-center"
          style={{ animationDelay: '0.12s' }}
        >
          <StarBoard key={rings} board={board} stones={emptyStones} className="h-auto w-full" />
          <div className="pointer-events-none absolute inset-x-0 -bottom-1 text-center text-xs text-muted">
            {board.n} nodes · {board.periCount} peries · 5 quarks · match total{' '}
            <span className="text-gold">{board.periCount + 1}</span>
          </div>
        </div>

        {/* Controls */}
        <div className="fade-up flex flex-col gap-6" style={{ animationDelay: '0.2s' }}>
          {/* Mode */}
          <section>
            <h2 className="mb-2 text-xs font-medium uppercase tracking-[0.25em] text-muted">
              Variant
            </h2>
            <div className="grid grid-cols-2 gap-3">
              {(
                [
                  { id: 'classic', title: '*Star', sub: '1 stone per turn' },
                  { id: 'double', title: 'Double *Star', sub: '2 stones per turn · first turn 1' },
                ] as const
              ).map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setMode(m.id)}
                  aria-pressed={mode === m.id}
                  className={`rounded-2xl border px-4 py-4 text-left transition-all ${
                    mode === m.id
                      ? 'border-gold/70 bg-gold-faint shadow-[0_0_28px_rgba(232,196,139,0.15)]'
                      : 'border-white/10 bg-white/[0.03] hover:border-gold/35'
                  }`}
                >
                  <span className="font-display block text-xl text-ink">{m.title}</span>
                  <span className="mt-1 block text-xs text-muted">{m.sub}</span>
                </button>
              ))}
            </div>
          </section>

          {/* Board size */}
          <section>
            <h2 className="mb-2 text-xs font-medium uppercase tracking-[0.25em] text-muted">
              Board
            </h2>
            <div className="grid grid-cols-4 gap-2">
              {BOARD_PRESETS.map((p) => (
                <button
                  key={p.rings}
                  type="button"
                  onClick={() => setRings(p.rings)}
                  aria-pressed={rings === p.rings}
                  className={`rounded-xl border px-2 py-2.5 text-center transition-all ${
                    rings === p.rings
                      ? 'border-gold/70 bg-gold-faint'
                      : 'border-white/10 bg-white/[0.03] hover:border-gold/35'
                  }`}
                >
                  <span className="block text-sm text-ink">{p.label}</span>
                  <span className="block text-[11px] text-muted">{p.rings} rings</span>
                </button>
              ))}
            </div>
            <div className="mt-3 flex items-center gap-3 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3">
              <label htmlFor="rings" className="shrink-0 text-xs text-muted">
                Custom
              </label>
              <input
                id="rings"
                type="range"
                min={MIN_RINGS}
                max={MAX_RINGS}
                value={rings}
                onChange={(e) => setRings(Number(e.target.value))}
                className="w-full accent-[#e8c48b]"
              />
              <span className="w-20 shrink-0 text-right text-sm text-ink">
                {rings} rings{preset ? '' : ' ·'}
                {preset ? '' : <span className="text-muted"> {board.n}n</span>}
              </span>
            </div>
          </section>

          {/* Players */}
          <section>
            <h2 className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.25em] text-muted">
              <Users className="h-3.5 w-3.5" aria-hidden /> Players
            </h2>
            <div className="grid grid-cols-2 gap-3">
              {[0, 1].map((i) => (
                <div
                  key={i}
                  className="flex items-center gap-3 rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2.5"
                >
                  <span
                    aria-hidden
                    className="h-5 w-5 shrink-0 rounded-full shadow-inner"
                    style={{
                      background: `radial-gradient(circle at 35% 30%, ${PLAYER_COLORS[i].bright}, ${PLAYER_COLORS[i].base} 55%, ${PLAYER_COLORS[i].deep})`,
                    }}
                  />
                  <input
                    value={names[i]}
                    maxLength={18}
                    aria-label={`Player ${i + 1} name`}
                    onChange={(e) =>
                      setNames((prev) => {
                        const next: [string, string] = [...prev];
                        next[i] = e.target.value;
                        return next;
                      })
                    }
                    className="w-full bg-transparent text-sm text-ink outline-none placeholder:text-muted"
                    placeholder={`Player ${i + 1}`}
                  />
                  <span className="text-[10px] uppercase tracking-wider text-muted">
                    {i === 0 ? 'first' : 'second'}
                  </span>
                </div>
              ))}
            </div>
            <label className="mt-3 flex cursor-pointer items-center justify-between rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3">
              <span>
                <span className="block text-sm text-ink">Pie rule</span>
                <span className="block text-xs text-muted">
                  After the opening stone, {names[1].trim() || 'Player 2'} may steal it
                </span>
              </span>
              <input
                type="checkbox"
                checked={pieRule}
                onChange={(e) => setPieRule(e.target.checked)}
                className="h-4 w-4 accent-[#e8c48b]"
              />
            </label>
          </section>

          <button
            type="button"
            onClick={start}
            className="font-display group mt-1 flex items-center justify-center gap-2.5 rounded-2xl border border-gold/60 bg-gradient-to-b from-[#e8c48b] to-[#c99d5f] px-6 py-4 text-xl font-medium text-[#241703] shadow-[0_8px_40px_rgba(232,196,139,0.25)] transition-[transform,box-shadow] duration-200 hover:scale-[1.015] hover:shadow-[0_8px_54px_rgba(232,196,139,0.4)] active:scale-[0.985]"
          >
            <Star className="h-5 w-5 transition-transform group-hover:rotate-[72deg]" aria-hidden />
            Begin the game
          </button>
        </div>
      </div>
    </main>
  );
}
