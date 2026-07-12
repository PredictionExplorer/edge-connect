import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { describe, expect, it, vi } from 'vitest';
import { initialState } from '@/lib/star/game';
import type { ScoreResult } from '@/lib/star/scoring';
import { GameOverOverlay } from '../GameOverOverlay';
import { RulesDialog } from '../RulesDialog';
import { Starfield } from '../Starfield';

function score(): ScoreResult {
  return {
    players: [
      {
        peries: 7,
        quarks: 3,
        stars: 1,
        quarkPeri: 1,
        award: 2,
        total: 11,
      },
      {
        peries: 8,
        quarks: 2,
        stars: 2,
        quarkPeri: 0,
        award: -2,
        total: 8,
      },
    ],
    nodeOwner: new Int8Array(50).fill(-1),
    aliveStone: new Uint8Array(50),
    contestedPeries: 15,
    leader: 0,
  };
}

describe('RulesDialog', () => {
  it('focuses close, ignores content clicks, and supports backdrop/Escape dismissal', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const { container, rerender } = render(
      <>
        <button type="button">Open rules</button>
        <RulesDialog open={false} onClose={onClose} />
      </>,
    );
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();

    const opener = screen.getByRole('button', { name: 'Open rules' });
    opener.focus();
    rerender(
      <>
        <button type="button">Open rules</button>
        <RulesDialog open onClose={onClose} />
      </>,
    );
    const close = screen.getByRole('button', { name: 'Close rules' });
    expect(close).toHaveFocus();
    expect(document.body.style.overflow).toBe('hidden');
    await user.click(screen.getByRole('heading', { name: 'How to play' }));
    expect(onClose).not.toHaveBeenCalled();
    await user.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalledOnce();
    await user.click(screen.getByRole('dialog'));
    expect(onClose).toHaveBeenCalledTimes(2);
    expect((await axe(container)).violations).toEqual([]);

    rerender(
      <>
        <button type="button">Open rules</button>
        <RulesDialog open={false} onClose={onClose} />
      </>,
    );
    expect(document.body.style.overflow).toBe('');
    expect(screen.getByRole('button', { name: 'Open rules' })).toHaveFocus();
  });
});

describe('GameOverOverlay', () => {
  const game = initialState({
    rings: 4,
    mode: 'double',
    pieRule: false,
    playerNames: ['Aurora', 'Vega'],
  });

  it('reports a binary winner and dispatches every next action', async () => {
    const user = userEvent.setup();
    const onReview = vi.fn();
    const onRematch = vi.fn();
    const onSetup = vi.fn();
    const { container } = render(
      <GameOverOverlay
        open
        game={game}
        score={score()}
        winner={0}
        onReview={onReview}
        onRematch={onRematch}
        onSetup={onSetup}
      />,
    );
    expect(
      screen.getByRole('heading', { name: /Aurora wins/ }),
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Rematch' })).toHaveFocus();
    await user.click(screen.getByRole('button', { name: 'Review board' }));
    await user.click(screen.getByRole('button', { name: 'Rematch' }));
    await user.click(screen.getByRole('button', { name: 'New setup' }));
    expect(onReview).toHaveBeenCalledOnce();
    expect(onRematch).toHaveBeenCalledOnce();
    expect(onSetup).toHaveBeenCalledOnce();
    expect((await axe(container)).violations).toEqual([]);
  });

  it('renders the supplied winner and supports a closed state', () => {
    const callbacks = {
      onReview: vi.fn(),
      onRematch: vi.fn(),
      onSetup: vi.fn(),
    };
    const { rerender } = render(
      <GameOverOverlay
        open
        game={game}
        score={score()}
        winner={1}
        {...callbacks}
      />,
    );
    expect(screen.getByRole('heading', { name: /Vega wins/ })).toBeInTheDocument();
    rerender(
      <GameOverOverlay
        open={false}
        game={game}
        score={score()}
        winner={1}
        {...callbacks}
      />,
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

