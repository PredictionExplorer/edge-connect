'use client';

import {
  memo,
  useCallback,
  useState,
  type RefObject,
} from 'react';
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
  toMove,
  interactive,
  playerNames,
  onPlace,
  filledCount,
  proof,
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
          toMove={toMove}
          interactive={interactive}
          playerNames={playerNames}
          onPlace={onPlace}
          onHover={handleHover}
          className="block h-full w-full"
        />
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
