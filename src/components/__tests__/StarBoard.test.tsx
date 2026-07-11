import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { getBoard } from '@/lib/star/board';
import { EMPTY } from '@/lib/star/scoring';
import { StarBoard } from '../StarBoard';

const board = getBoard(4);

function emptyBoard(): Int8Array {
  return new Int8Array(board.n).fill(EMPTY);
}

afterEach(cleanup);

describe('StarBoard', () => {
  it('uses one tab stop and spatial arrow-key navigation', async () => {
    const user = userEvent.setup();
    const onPlace = vi.fn();
    render(
      <StarBoard
        board={board}
        stones={emptyBoard()}
        interactive
        playerNames={['Ada', 'Grace']}
        onPlace={onPlace}
      />,
    );

    const nodes = screen.getAllByRole('button');
    const startingNode = screen.getByRole('button', {
      name: /node \*10, empty interior node; ada may place here/i,
    });

    expect(nodes).toHaveLength(board.n);
    expect(nodes.filter((node) => node.tabIndex === 0)).toEqual([startingNode]);

    startingNode.focus();
    await user.keyboard('{ArrowRight}');

    const nextNode = document.activeElement as HTMLElement;
    expect(nextNode).toBeInstanceOf(SVGElement);
    expect(nextNode).not.toBe(startingNode);
    expect(nextNode).toHaveAttribute('tabindex', '0');
    expect(startingNode).toHaveAttribute('tabindex', '-1');

    const nextNodeIndex = nodes.indexOf(nextNode);
    await user.keyboard('{Enter}');
    fireEvent.keyDown(nextNode, { key: ' ', code: 'Space' });
    expect(onPlace).toHaveBeenNthCalledWith(1, nextNodeIndex);
    expect(onPlace).toHaveBeenNthCalledWith(2, nextNodeIndex);

    await user.keyboard('{End}');
    expect(document.activeElement).toBe(nodes.at(-1));
    await user.keyboard('{Home}');
    expect(document.activeElement).toBe(nodes[0]);
  });

  it('preserves hover and mouse placement behavior', async () => {
    const user = userEvent.setup();
    const onHover = vi.fn();
    const onPlace = vi.fn();
    render(
      <StarBoard
        board={board}
        stones={emptyBoard()}
        interactive
        onHover={onHover}
        onPlace={onPlace}
      />,
    );

    const firstNode = screen.getByRole('button', { name: /node \*10, empty/i });
    fireEvent.mouseEnter(firstNode);
    expect(onHover).toHaveBeenLastCalledWith(0);

    await user.click(firstNode);
    expect(onPlace).toHaveBeenCalledOnce();
    expect(onPlace).toHaveBeenCalledWith(0);

    fireEvent.mouseLeave(screen.getByRole('group', { name: /\*star board with 4 rings/i }));
    expect(onHover).toHaveBeenLastCalledWith(-1);
  });

  it('announces occupied-node state and does not reactivate it', async () => {
    const user = userEvent.setup();
    const stones = emptyBoard();
    stones[0] = 0;
    const onPlace = vi.fn();
    render(
      <StarBoard
        board={board}
        stones={stones}
        interactive
        lastMove={0}
        playerNames={['Ada', 'Grace']}
        onPlace={onPlace}
      />,
    );

    const occupiedNode = screen.getByRole('button', {
      name: /node \*10, ada stone on interior node, last move/i,
    });
    expect(occupiedNode).toHaveAttribute('aria-disabled', 'true');

    occupiedNode.focus();
    await user.keyboard('{Enter}');
    await user.click(occupiedNode);
    expect(onPlace).not.toHaveBeenCalled();
  });

  it('has no detectable accessibility violations', async () => {
    const { container } = render(
      <StarBoard
        board={board}
        stones={emptyBoard()}
        interactive
        playerNames={['Ada', 'Grace']}
      />,
    );

    expect((await axe(container)).violations).toEqual([]);
  });
});
