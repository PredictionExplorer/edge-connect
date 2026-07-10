import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { describe, expect, it, vi } from 'vitest';
import { initialState } from '@/lib/star/game';
import type { ScoreResult } from '@/lib/star/scoring';
import { GameOverOverlay } from '../GameOverOverlay';
import { RulesDialog } from '../RulesDialog';
import { Starfield } from '../Starfield';

function score(leader: -1 | 0 | 1): ScoreResult {
  const tied = leader !== -1;
  return {
    players: [
      {
        peries: 7,
        quarks: tied ? 3 : 2,
        stars: 1,
        quarkPeri: tied ? 1 : 0,
        award: 2,
        total: 10,
      },
      {
        peries: 8,
        quarks: 2,
        stars: 2,
        quarkPeri: 0,
        award: -2,
        total: tied ? 10 : 8,
      },
    ],
    nodeOwner: new Int8Array(30).fill(-1),
    aliveStone: new Uint8Array(30),
    contestedPeries: 15,
    leader,
  };
}

describe('RulesDialog', () => {
  it('focuses close, ignores content clicks, and supports backdrop/Escape dismissal', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { container, rerender } = render(
      <RulesDialog open={false} onClose={onClose} />,
    );
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();

    rerender(<RulesDialog open onClose={onClose} />);
    const close = screen.getByRole('button', { name: 'Close rules' });
    expect(close).toHaveFocus();
    await user.click(screen.getByRole('heading', { name: 'How to play' }));
    expect(onClose).not.toHaveBeenCalled();
    await user.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalledOnce();
    await user.click(screen.getByRole('dialog'));
    expect(onClose).toHaveBeenCalledTimes(2);
    expect((await axe(container)).violations).toEqual([]);
  });
});

describe('GameOverOverlay', () => {
  const game = initialState({
    rings: 3,
    mode: 'double',
    pieRule: false,
    playerNames: ['Aurora', 'Vega'],
  });

  it('reports quark tie-breaks and dispatches every next action', async () => {
    const user = userEvent.setup();
    const onReview = vi.fn();
    const onRematch = vi.fn();
    const onSetup = vi.fn();
    const { container } = render(
      <GameOverOverlay
        open
        game={game}
        score={score(0)}
        onReview={onReview}
        onRematch={onRematch}
        onSetup={onSetup}
      />,
    );
    expect(
      screen.getByRole('heading', { name: /Aurora wins/ }),
    ).toBeInTheDocument();
    expect(screen.getByText(/decided on quarks, 3 to 2/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Review board' }));
    await user.click(screen.getByRole('button', { name: 'Rematch' }));
    await user.click(screen.getByRole('button', { name: 'New setup' }));
    expect(onReview).toHaveBeenCalledOnce();
    expect(onRematch).toHaveBeenCalledOnce();
    expect(onSetup).toHaveBeenCalledOnce();
    expect((await axe(container)).violations).toEqual([]);
  });

  it('renders tie and closed states without inventing a winner', () => {
    const callbacks = {
      onReview: vi.fn(),
      onRematch: vi.fn(),
      onSetup: vi.fn(),
    };
    const { rerender } = render(
      <GameOverOverlay open game={game} score={score(-1)} {...callbacks} />,
    );
    expect(screen.getByText('A perfect tie')).toBeInTheDocument();
    rerender(
      <GameOverOverlay open={false} game={game} score={score(-1)} {...callbacks} />,
    );
    expect(
      screen.queryByRole('dialog', { name: 'Game over' }),
    ).not.toBeInTheDocument();
  });
});

describe('Starfield', () => {
  it('is deterministic, decorative, and hidden from accessibility APIs', () => {
    const { container, rerender } = render(<Starfield />);
    const first = Array.from(container.querySelectorAll('circle'), (node) =>
      node.getAttribute('cx'),
    );
    expect(first).toHaveLength(140);
    expect(container.querySelector('svg')).toHaveAttribute('aria-hidden', 'true');
    rerender(<Starfield />);
    expect(
      Array.from(container.querySelectorAll('circle'), (node) =>
        node.getAttribute('cx'),
      ),
    ).toEqual(first);
  });
});

