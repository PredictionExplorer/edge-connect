import {
  CircleAlert,
  LoaderCircle,
  PauseCircle,
  Sparkles,
  Trophy,
} from 'lucide-react';
import type { Mode } from '@/lib/star/game';

export type GameStatusState = 'human' | 'thinking' | 'paused' | 'error' | 'over';

interface GameStatusProps {
  state: GameStatusState;
  playerName: string;
  controllerName: string;
  mode: Mode;
  movesLeft: number;
  color: {
    base: string;
    bright: string;
    soft: string;
  };
  className?: string;
}

export function GameStatus({
  state,
  playerName,
  controllerName,
  mode,
  movesLeft,
  color,
  className = '',
}: GameStatusProps) {
  const presentation =
    state === 'over'
      ? {
          icon: Trophy,
          title: 'The sky is settled',
          detail: 'Review the final position or start another game.',
        }
      : state === 'thinking'
        ? {
            icon: LoaderCircle,
            title: `${controllerName} is thinking…`,
            detail: `Choosing a move for ${playerName}.`,
          }
        : state === 'paused'
          ? {
              icon: PauseCircle,
              title: 'AI turn paused',
              detail: 'History navigation paused automatic play.',
            }
          : state === 'error'
            ? {
                icon: CircleAlert,
                title: 'AI needs attention',
                detail: `${playerName}’s turn is waiting for recovery.`,
              }
            : {
                icon: Sparkles,
                title: `${playerName} to play`,
                detail:
                  mode === 'double'
                    ? `${movesLeft} stone${movesLeft === 1 ? '' : 's'} left this turn`
                    : 'Place one stone on any open node.',
              };

  const Icon = presentation.icon;

  return (
    <section
      aria-busy={state === 'thinking'}
      aria-live="polite"
      data-game-status={state}
      className={`panel-surface flex h-[4.5rem] min-w-0 items-center gap-3 overflow-hidden rounded-2xl px-4 ${className}`}
      style={{
        borderColor: state === 'over' ? 'rgba(232,196,139,0.5)' : `${color.base}66`,
        background:
          state === 'over'
            ? 'linear-gradient(145deg, rgba(232,196,139,0.14), rgba(16,21,42,0.9))'
            : `linear-gradient(145deg, ${color.soft}, rgba(16,21,42,0.9))`,
      }}
    >
      <span
        aria-hidden
        className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border"
        style={{ borderColor: `${color.base}55`, color: color.bright }}
      >
        <Icon
          className={`h-5 w-5 ${state === 'thinking' ? 'motion-safe:animate-spin' : ''}`}
        />
      </span>
      <div className="min-w-0">
        <p className="truncate text-sm font-medium" style={{ color: color.bright }}>
          {presentation.title}
        </p>
        <p className="truncate text-xs text-muted">{presentation.detail}</p>
      </div>
    </section>
  );
}
