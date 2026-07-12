import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { getBoard, parseLabel } from '@/lib/star/board';
import { scoreCompletionBounds } from '@/lib/star/completion-bounds';
import { EMPTY, scorePosition } from '@/lib/star/scoring';
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

    const svg = screen.getByRole('group', { name: /\*star board with 4 rings/i });
    vi.spyOn(svg, 'getBoundingClientRect').mockReturnValue({
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      right: 238,
      bottom: 238,
      width: 238,
      height: 238,
      toJSON: () => ({}),
    });
    fireEvent.pointerMove(svg, {
      clientX: board.xs[1] * 100 + 119,
      clientY: board.ys[1] * 100 + 119,
    });
    expect(onHover).toHaveBeenLastCalledWith(1);
    fireEvent.click(svg, {
      clientX: board.xs[1] * 100 + 119,
      clientY: board.ys[1] * 100 + 119,
    });
    expect(onPlace).toHaveBeenNthCalledWith(2, 1);

    fireEvent.mouseLeave(svg);
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

  it('draws same-color connections and highlights a whole group', () => {
    const stones = emptyBoard();
    const first = parseLabel(board, '*40');
    const second = parseLabel(board, '*41');
    stones[first] = 0;
    stones[second] = 0;
    const score = scorePosition(board, stones);
    const { container } = render(
      <StarBoard
        board={board}
        stones={stones}
        aliveStone={score.aliveStone}
        interactive
        playerNames={['Ada', 'Grace']}
      />,
    );

    const groupPath = container.querySelector(
      'path[data-connection-layer="group"][data-player="0"]',
    );
    const starPath = container.querySelector(
      'path[data-connection-layer="star"][data-player="0"]',
    );
    const groupPathData = groupPath?.getAttribute('d');
    expect(groupPathData).toContain('M');
    expect(starPath?.getAttribute('d')).toContain('M');

    fireEvent.mouseEnter(
      screen.getByRole('button', { name: /node \*40, ada stone/i }),
    );
    expect(
      container
        .querySelector('path[data-connection-layer="highlight"]')
        ?.getAttribute('d'),
    ).toBe(groupPathData);
    expect(container.querySelectorAll('[data-group-highlight]')).toHaveLength(2);
  });

  it('does not cross out rescuable stones in projected opponent territory', () => {
    const stones = emptyBoard();
    const amber = [parseLabel(board, 'S10'), parseLabel(board, 'R10')];
    for (const node of amber) stones[node] = 0;
    stones[parseLabel(board, '*40')] = 1;
    stones[parseLabel(board, '*41')] = 1;
    const score = scorePosition(board, stones);
    const bounds = scoreCompletionBounds(board, stones);
    const { container, rerender } = render(
      <StarBoard
        board={board}
        stones={stones}
        nodeOwner={score.nodeOwner}
        aliveStone={score.aliveStone}
        provablyDeadStone={bounds.provablyDeadStone}
        interactive
        playerNames={['Ada', 'Grace']}
      />,
    );

    for (const node of amber) {
      expect(score.nodeOwner[node]).toBe(1);
      expect(bounds.provablyDeadStone[node]).toBe(0);
      expect(
        container.querySelector(`[data-stone-node="${node}"]`),
      ).toHaveAttribute('opacity', '1');
    }
    expect(
      container.querySelectorAll('[data-provably-dead-stone]'),
    ).toHaveLength(0);

    rerender(
      <StarBoard
        board={board}
        stones={stones}
        nodeOwner={score.nodeOwner}
        aliveStone={score.aliveStone}
        provablyDeadStone={bounds.provablyDeadStone}
        showTerritory
        interactive
        playerNames={['Ada', 'Grace']}
      />,
    );
    for (const node of amber) {
      expect(
        container.querySelector(`[data-stone-node="${node}"]`),
      ).toHaveAttribute('opacity', '0.35');
    }
    expect(
      container.querySelectorAll('[data-provably-dead-stone]'),
    ).toHaveLength(0);
  });

  it('marks a provably dead stone without influence enabled', () => {
    const stones = emptyBoard();
    for (const label of ['*43', 'T42', 'T43']) {
      stones[parseLabel(board, label)] = 0;
    }
    for (const label of ['*42', '*32', 'S30', 'S40']) {
      stones[parseLabel(board, label)] = 1;
    }
    const score = scorePosition(board, stones);
    const bounds = scoreCompletionBounds(board, stones);
    const captured = parseLabel(board, '*43');
    const { container } = render(
      <StarBoard
        board={board}
        stones={stones}
        nodeOwner={score.nodeOwner}
        aliveStone={score.aliveStone}
        provablyDeadStone={bounds.provablyDeadStone}
        interactive
        playerNames={['Ada', 'Grace']}
      />,
    );

    expect(score.nodeOwner[captured]).toBe(1);
    expect(bounds.provablyDeadStone[captured]).toBe(1);
    expect(
      container.querySelector(`[data-provably-dead-stone="${captured}"]`),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', {
        name: /node \*43, ada stone on peri, provably dead; cannot form a living star in any completion/i,
      }),
    ).toBeInTheDocument();
  });

  it('crosses every stone in a supplied provably dead group', () => {
    const stones = emptyBoard();
    const first = parseLabel(board, 'S20');
    const second = parseLabel(board, 'S30');
    stones[first] = 0;
    stones[second] = 0;
    const dead = new Uint8Array(board.n);
    dead[first] = 1;
    dead[second] = 1;
    const { container } = render(
      <StarBoard
        board={board}
        stones={stones}
        provablyDeadStone={dead}
        interactive
        playerNames={['Ada', 'Grace']}
      />,
    );

    expect(
      container.querySelectorAll('[data-provably-dead-stone]'),
    ).toHaveLength(2);
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
