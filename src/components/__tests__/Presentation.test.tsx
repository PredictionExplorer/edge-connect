import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { describe, expect, it, vi } from 'vitest';
import { initialState } from '@/lib/star/game';
import type { ScoreResult } from '@/lib/star/scoring';
import {
  ClinchDialog,
  EndGameConfirmDialog,
  ResignDialog,
} from '../EndGameDialogs';
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
        result={{ reason: 'full-board', winner: 0, score: score() }}
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
        result={{ reason: 'full-board', winner: 1, score: score() }}
        {...callbacks}
      />,
    );
    expect(screen.getByRole('heading', { name: /Vega wins/ })).toBeInTheDocument();
    rerender(
      <GameOverOverlay
        open={false}
        game={game}
        result={{ reason: 'full-board', winner: 1, score: score() }}
        {...callbacks}
      />,
    );
    expect(
      screen.queryByRole('dialog', { name: 'Game over' }),
    ).not.toBeInTheDocument();
  });

  it('distinguishes a clinched result from a real final score', async () => {
    const user = userEvent.setup();
    const onReview = vi.fn();
    const { container } = render(
      <GameOverOverlay
        open
        game={game}
        result={{
          reason: 'clinch',
          winner: 1,
          loser: 0,
          emptyNodes: 7,
        }}
        onReview={onReview}
        onRematch={vi.fn()}
        onSetup={vi.fn()}
      />,
    );

    expect(screen.getByRole('heading', { name: 'Vega wins' })).toBeInTheDocument();
    expect(screen.getByText(/no final score was recorded/i)).toBeInTheDocument();
    expect(screen.queryByText('11')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Review proof' }));
    expect(onReview).toHaveBeenCalledOnce();
    expect((await axe(container)).violations).toEqual([]);
  });
});

describe('end-game decision dialogs', () => {
  it('keeps the clinch decision non-destructive by default', async () => {
    const user = userEvent.setup();
    const onContinue = vi.fn();
    const onProof = vi.fn();
    const onEnd = vi.fn();
    const { container } = render(
      <ClinchDialog
        open
        winner={1}
        winnerName="Grace"
        loserName="Ada"
        emptyNodes={3}
        proofScores={[10, 11]}
        onContinue={onContinue}
        onProof={onProof}
        onEnd={onEnd}
      />,
    );

    const dialog = screen.getByRole('dialog', {
      name: 'Grace cannot be caught',
    });
    expect(dialog).toHaveAccessibleDescription(
      /even if every remaining open node became ada/i,
    );
    expect(
      within(dialog).getByRole('button', { name: 'Continue playing' }),
    ).toHaveFocus();
    await user.click(
      within(dialog).getByRole('button', { name: 'Show proof board' }),
    );
    await user.click(
      within(dialog).getByRole('button', { name: 'End game now' }),
    );
    await user.keyboard('{Escape}');
    expect(onProof).toHaveBeenCalledOnce();
    expect(onEnd).toHaveBeenCalledOnce();
    expect(onContinue).toHaveBeenCalledOnce();
    expect((await axe(container)).violations).toEqual([]);
  });

  it('names safe and destructive confirmation actions', () => {
    const { rerender } = render(
      <EndGameConfirmDialog
        open
        winnerName="Grace"
        emptyNodes={4}
        onCancel={vi.fn()}
        onConfirm={vi.fn()}
      />,
    );
    expect(
      screen.getByRole('dialog', { name: 'End this clinched game?' }),
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Keep playing' })).toHaveFocus();

    rerender(
      <ResignDialog
        open
        loserName="Ada"
        winnerName="Grace"
        onCancel={vi.fn()}
        onConfirm={vi.fn()}
      />,
    );
    expect(screen.getByRole('dialog', { name: 'Resign Ada?' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Resign Ada' })).toBeInTheDocument();
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

