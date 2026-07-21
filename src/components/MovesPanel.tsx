'use client';

import { useEffect, useRef } from 'react';
import {
  ChevronFirst,
  ChevronLast,
  ChevronLeft,
  ChevronRight,
  CornerUpLeft,
  Replace,
} from 'lucide-react';
import type { Timeline, TimelineTurn } from '@/lib/star/timeline';
import { PLAYER_COLORS } from './theme';

interface MovesPanelProps {
  timeline: Timeline;
  /** Total number of actions in the live log. */
  total: number;
  /** Number of actions applied to the position on display; `total` when live. */
  currentPly: number;
  playerNames: readonly [string, string];
  /** Whether "Play from here" may truncate the log (mirrors Undo gating). */
  canRewind: boolean;
  /** Seek the review position; `ply >= total` returns to live. */
  onSeek: (ply: number) => void;
  onRewind: () => void;
}

function TurnRow({
  turn,
  currentPly,
  playerNames,
  onSeek,
}: {
  turn: TimelineTurn;
  currentPly: number;
  playerNames: readonly [string, string];
  onSeek: (ply: number) => void;
}) {
  const color = PLAYER_COLORS[turn.player];
  return (
    <li
      className="grid grid-cols-[1.75rem_auto_minmax(0,1fr)] items-center gap-x-2 py-0.5"
      data-turn-row={turn.turnNumber}
    >
      <span className="text-right font-mono text-[0.7rem] tabular-nums text-muted">
        {turn.turnNumber + 1}
      </span>
      <span
        aria-hidden
        className="h-2.5 w-2.5 rounded-full"
        style={{
          background: `radial-gradient(circle at 35% 30%, ${color.bright}, ${color.base} 60%, ${color.deep})`,
        }}
        title={playerNames[turn.player]}
      />
      <span className="flex min-w-0 flex-wrap items-center gap-1">
        {turn.entries.map((entry) => {
          const ply = entry.index + 1;
          const isCurrent = ply === currentPly;
          const isFuture = entry.index >= currentPly;
          const isSwap = entry.action.type === 'swap';
          return (
            <button
              key={entry.index}
              type="button"
              data-move-chip={entry.index}
              aria-current={isCurrent ? 'step' : undefined}
              aria-label={
                isSwap
                  ? `Go to move ${ply}: ${playerNames[entry.player]} swapped sides`
                  : `Go to move ${ply}: ${playerNames[entry.player]} at ${entry.label}`
              }
              onClick={() => onSeek(ply)}
              className={`flex min-h-7 items-center gap-1 rounded-md border px-1.5 font-mono text-xs tabular-nums transition-colors ${
                isFuture && !isCurrent ? 'opacity-45' : ''
              }`}
              style={{
                borderColor: isCurrent ? color.base : 'rgba(255,255,255,0.1)',
                background: isCurrent ? color.soft : 'rgba(255,255,255,0.03)',
                color: isCurrent ? color.bright : 'var(--ink)',
              }}
            >
              {isSwap && <Replace className="h-3 w-3" aria-hidden />}
              {entry.label}
            </button>
          );
        })}
      </span>
    </li>
  );
}

export function MovesPanel({
  timeline,
  total,
  currentPly,
  playerNames,
  canRewind,
  onSeek,
  onRewind,
}: MovesPanelProps) {
  const listRef = useRef<HTMLOListElement>(null);
  const live = currentPly >= total;

  useEffect(() => {
    const list = listRef.current;
    if (!list) return;
    const active = list.querySelector<HTMLElement>(
      currentPly > 0 ? `[data-move-chip="${currentPly - 1}"]` : '[data-turn-row]',
    );
    if (!active) return;
    // Scroll only the list itself; scrollIntoView would also scroll ancestors,
    // yanking the whole page down to the panel on small screens.
    const listRect = list.getBoundingClientRect();
    const activeRect = active.getBoundingClientRect();
    if (activeRect.top < listRect.top) {
      list.scrollTop += activeRect.top - listRect.top;
    } else if (activeRect.bottom > listRect.bottom) {
      list.scrollTop += activeRect.bottom - listRect.bottom;
    }
  }, [currentPly, total]);

  return (
    <section
      aria-label="Move history"
      data-moves-panel
      className="rounded-2xl border border-white/10 bg-white/[0.025]"
    >
      <header className="flex items-center justify-between gap-2 px-3 pt-2.5">
        <h3 className="text-xs font-semibold uppercase tracking-[0.14em] text-muted">
          Moves
          <span className="ml-2 font-mono text-[0.7rem] normal-case tracking-normal text-muted/80">
            {total}
          </span>
        </h3>
        <div
          role="group"
          aria-label="Review navigation"
          className="flex items-center gap-0.5"
        >
          <button
            type="button"
            aria-label="Jump to the empty board"
            disabled={currentPly === 0 || total === 0}
            onClick={() => onSeek(0)}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 text-ink transition-colors enabled:hover:border-gold/50 disabled:opacity-30"
          >
            <ChevronFirst className="h-4 w-4" aria-hidden />
          </button>
          <button
            type="button"
            aria-label="Step one move back"
            disabled={currentPly === 0 || total === 0}
            onClick={() => onSeek(currentPly - 1)}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 text-ink transition-colors enabled:hover:border-gold/50 disabled:opacity-30"
          >
            <ChevronLeft className="h-4 w-4" aria-hidden />
          </button>
          <button
            type="button"
            aria-label="Step one move forward"
            disabled={live}
            onClick={() => onSeek(currentPly + 1)}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 text-ink transition-colors enabled:hover:border-gold/50 disabled:opacity-30"
          >
            <ChevronRight className="h-4 w-4" aria-hidden />
          </button>
          <button
            type="button"
            aria-label="Jump to the live position"
            disabled={live}
            onClick={() => onSeek(total)}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 text-ink transition-colors enabled:hover:border-gold/50 disabled:opacity-30"
          >
            <ChevronLast className="h-4 w-4" aria-hidden />
          </button>
        </div>
      </header>

      {timeline.turns.length === 0 ? (
        <p className="px-3 pb-3 pt-2 text-xs leading-relaxed text-muted">
          No moves yet — placements will appear here as the game unfolds.
        </p>
      ) : (
        <ol
          ref={listRef}
          className="thin-scroll mt-1.5 max-h-56 overflow-y-auto px-2 pb-2"
        >
          {timeline.turns.map((turn) => (
            <TurnRow
              key={turn.turnNumber}
              turn={turn}
              currentPly={currentPly}
              playerNames={playerNames}
              onSeek={onSeek}
            />
          ))}
        </ol>
      )}

      <footer className="flex min-h-9 items-center justify-between gap-2 border-t border-white/[0.07] px-3 py-1.5">
        <span className="text-[0.7rem] text-muted" data-moves-position>
          {live ? (
            <>
              <span
                aria-hidden
                className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full bg-gold align-middle"
              />
              Live position
            </>
          ) : currentPly === 0 ? (
            'Viewing the start'
          ) : (
            `Viewing move ${currentPly} of ${total}`
          )}
        </span>
        {!live && canRewind && (
          <button
            type="button"
            onClick={onRewind}
            className="flex min-h-8 items-center gap-1.5 rounded-lg border border-gold/50 px-2.5 py-1 text-xs text-gold-strong transition-colors hover:bg-gold/15"
          >
            <CornerUpLeft className="h-3.5 w-3.5" aria-hidden /> Play from here
          </button>
        )}
      </footer>
    </section>
  );
}
