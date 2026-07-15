import {
  CircleAlert,
  Eye,
  LoaderCircle,
  PauseCircle,
  ShieldCheck,
  Sparkles,
  Trophy,
} from 'lucide-react';
import type { Mode } from '@/lib/star/game';

export type GameStatusState =
  | 'human'
  | 'thinking'
  | 'paused'
  | 'error'
  | 'clinch'
  | 'proof'
  | 'confirming'
  | 'over';

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
          detail: 'Review the result or start another game.',
        }
      : state === 'clinch'
        ? {
            icon: ShieldCheck,
            title: `${playerName} has clinched`,
            detail: 'Choose whether to end now or continue playing.',
          }
        : state === 'proof'
          ? {
              icon: Eye,
              title: 'Clinch proof on board',
              detail: `${playerName} still wins in this strongest-case scenario.`,
            }
          : state === 'confirming'
            ? {
                icon: PauseCircle,
                title: 'Play paused for confirmation',
                detail: 'Confirm the choice or return to the game.',
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
      className={`panel-surface flex min-h-[4.875rem] min-w-0 items-center gap-3 rounded-2xl px-4 py-3 ${className}`}
      style={{
        borderColor:
          state === 'over' || state === 'clinch' || state === 'proof'
            ? 'rgba(232,196,139,0.5)'
            : `${color.base}66`,
        background:
          state === 'over' || state === 'clinch' || state === 'proof'
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
        <p className="text-sm font-medium leading-tight" style={{ color: color.bright }}>
          {presentation.title}
        </p>
        <p className="mt-0.5 text-xs leading-tight text-muted">{presentation.detail}</p>
      </div>
    </section>
  );
}
