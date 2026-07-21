'use client';

import {
  memo,
  useCallback,
  useState,
  type RefObject,
} from 'react';
import { History } from 'lucide-react';
import type { StarBoardProps } from './StarBoard';
import { StarBoard } from './StarBoard';
import styles from './GameScreen.module.css';

type BoardStageProps = Pick<
  StarBoardProps,
  | 'board'
  | 'stones'
  | 'nodeOwner'
  | 'aliveStone'
  | 'provablyDeadStone'
  | 'syntheticStone'
  | 'showTerritory'
  | 'lastMove'
  | 'currentTurnMoves'
  | 'lastTurnMoves'
  | 'currentTurnCapacity'
  | 'toMove'
  | 'interactive'
  | 'playerNames'
  | 'onPlace'
> & {
  filledCount: number;
  proof?: {
    label: string;
    detail: string;
    description: string;
  } | null;
  /** Non-null while an earlier position is on the board. */
  review?: {
    ply: number;
    total: number;
    onExit: () => void;
  } | null;
  focusRef?: RefObject<HTMLElement | null>;
  className?: string;
};

export const BoardStage = memo(function BoardStage({
  board,
  stones,
  nodeOwner,
  aliveStone,
  provablyDeadStone,
  syntheticStone,
  showTerritory,
  lastMove,
  currentTurnMoves,
  lastTurnMoves,
  currentTurnCapacity,
  toMove,
  interactive,
  playerNames,
  onPlace,
  filledCount,
  proof,
  review,
  focusRef,
  className = '',
}: BoardStageProps) {
  const [hoverNode, setHoverNode] = useState(-1);
  const handleHover = useCallback((node: number) => setHoverNode(node), []);

  return (
    <section
      ref={focusRef}
      tabIndex={-1}
      aria-label={proof ? 'Clinch proof board' : 'Game board'}
      data-board-stage
      data-proof-active={proof ? 'true' : undefined}
      data-review-active={review ? 'true' : undefined}
      className={`${styles.boardStage} ${className}`}
    >
      <div className={styles.boardFrame}>
        <StarBoard
          board={board}
          stones={stones}
          nodeOwner={nodeOwner}
          aliveStone={aliveStone}
          provablyDeadStone={provablyDeadStone}
          syntheticStone={syntheticStone}
          proofDescription={proof?.description}
          showTerritory={showTerritory}
          lastMove={lastMove}
          currentTurnMoves={currentTurnMoves}
          lastTurnMoves={lastTurnMoves}
          currentTurnCapacity={currentTurnCapacity}
          toMove={toMove}
          interactive={interactive}
          playerNames={playerNames}
          onPlace={onPlace}
          onHover={handleHover}
          className="block h-full w-full"
        />
        {review && (
          <div
            data-review-banner
            className="pop-in absolute left-1/2 top-2 z-10 flex max-w-[calc(100%-1rem)] -translate-x-1/2 items-center gap-2.5 rounded-full border border-gold/45 bg-night-surface-strong/95 py-1.5 pl-3.5 pr-1.5 shadow-[0_10px_36px_rgba(0,0,0,0.5)] backdrop-blur-md"
          >
            <History className="h-3.5 w-3.5 shrink-0 text-gold" aria-hidden />
            <span className="whitespace-nowrap text-xs text-ink">
              {review.ply === 0 ? (
                'Start'
              ) : (
                <>
                  Move{' '}
                  <span className="font-mono tabular-nums text-gold-strong">
                    {review.ply}
                  </span>
                </>
              )}{' '}
              of{' '}
              <span className="font-mono tabular-nums">{review.total}</span>
            </span>
            <button
              type="button"
              onClick={review.onExit}
              className="min-h-8 whitespace-nowrap rounded-full border border-gold/60 bg-gold-faint px-3 py-1 text-xs font-medium text-gold-strong transition-colors hover:bg-gold/25"
            >
              Back to live
            </button>
          </div>
        )}
        <div
          aria-live="polite"
          className={`pointer-events-none absolute bottom-2 left-2 max-w-[calc(100%-1rem)] rounded-lg border px-2.5 py-1 font-mono text-xs backdrop-blur-sm ${
            proof
              ? 'border-gold/35 bg-night-surface-strong/90 text-ink shadow-lg'
              : 'truncate border-white/10 bg-black/45 text-muted'
          }`}
        >
          {proof ? (
            <>
              <span className="font-sans text-[0.65rem] font-semibold uppercase tracking-[0.13em] text-gold">
                {proof.label}
              </span>
              <span className="ml-2 text-muted">{proof.detail}</span>
            </>
          ) : hoverNode >= 0 ? (
            <>
              node <span className="text-gold">{board.labels[hoverNode]}</span>
              {board.isQuark[hoverNode]
                ? ' · quark'
                : board.isPeri[hoverNode]
                  ? ' · peri'
                  : ''}
            </>
          ) : (
            `${filledCount} / ${board.n} nodes filled`
          )}
        </div>
      </div>
    </section>
  );
});
