'use client';

import { memo, useCallback, useState } from 'react';
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
  | 'showTerritory'
  | 'lastMove'
  | 'currentTurnMoves'
  | 'toMove'
  | 'interactive'
  | 'playerNames'
  | 'onPlace'
> & {
  filledCount: number;
  className?: string;
};

export const BoardStage = memo(function BoardStage({
  board,
  stones,
  nodeOwner,
  aliveStone,
  provablyDeadStone,
  showTerritory,
  lastMove,
  currentTurnMoves,
  toMove,
  interactive,
  playerNames,
  onPlace,
  filledCount,
  className = '',
}: BoardStageProps) {
  const [hoverNode, setHoverNode] = useState(-1);
  const handleHover = useCallback((node: number) => setHoverNode(node), []);

  return (
    <section
      aria-label="Game board"
      data-board-stage
      className={`${styles.boardStage} ${className}`}
    >
      <div className={styles.boardFrame}>
        <StarBoard
          board={board}
          stones={stones}
          nodeOwner={nodeOwner}
          aliveStone={aliveStone}
          provablyDeadStone={provablyDeadStone}
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
          className="pointer-events-none absolute bottom-2 left-2 max-w-[calc(100%-1rem)] truncate rounded-lg border border-white/10 bg-black/45 px-2.5 py-1 font-mono text-xs text-muted backdrop-blur-sm"
        >
          {hoverNode >= 0 ? (
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
