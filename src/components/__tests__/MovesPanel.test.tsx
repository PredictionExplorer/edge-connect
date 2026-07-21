import { cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { GameAction, GameConfig } from '@/lib/star/game';
import { buildTimeline } from '@/lib/star/timeline';
import { MovesPanel } from '../MovesPanel';

const double: GameConfig = {
  rings: 4,
  mode: 'double',
  pieRule: false,
  playerNames: ['Ada', 'Grace'],
};

function places(...nodes: number[]): GameAction[] {
  return nodes.map((node) => ({ type: 'place', node }));
}

const names = ['Ada', 'Grace'] as const;

afterEach(cleanup);

describe('MovesPanel', () => {
  it('groups placements by turn and seeks to a clicked move', async () => {
    const user = userEvent.setup();
    const onSeek = vi.fn();
    const log = places(0, 1, 2, 3);
    render(
      <MovesPanel
        timeline={buildTimeline(double, log)}
        total={log.length}
        currentPly={log.length}
        playerNames={names}
        canRewind
        onSeek={onSeek}
        onRewind={vi.fn()}
      />,
    );

    const panel = screen.getByRole('region', { name: 'Move history' });
    expect(panel.querySelectorAll('[data-turn-row]')).toHaveLength(3);
    expect(within(panel).getByText('Live position')).toBeInTheDocument();
    expect(
      within(panel).queryByRole('button', { name: 'Play from here' }),
    ).not.toBeInTheDocument();

    await user.click(
      within(panel).getByRole('button', { name: 'Go to move 2: Grace at S10' }),
    );
    expect(onSeek).toHaveBeenCalledWith(2);
  });

  it('marks the reviewed move, steps with the transport, and offers rewind', async () => {
    const user = userEvent.setup();
    const onSeek = vi.fn();
    const onRewind = vi.fn();
    const log = places(0, 1, 2, 3);
    const { container } = render(
      <MovesPanel
        timeline={buildTimeline(double, log)}
        total={log.length}
        currentPly={2}
        playerNames={names}
        canRewind
        onSeek={onSeek}
        onRewind={onRewind}
      />,
    );

    const current = screen.getByRole('button', {
      name: 'Go to move 2: Grace at S10',
    });
    expect(current).toHaveAttribute('aria-current', 'step');
    expect(screen.getByText('Viewing move 2 of 4')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Step one move back' }));
    expect(onSeek).toHaveBeenLastCalledWith(1);
    await user.click(
      screen.getByRole('button', { name: 'Step one move forward' }),
    );
    expect(onSeek).toHaveBeenLastCalledWith(3);
    await user.click(
      screen.getByRole('button', { name: 'Jump to the empty board' }),
    );
    expect(onSeek).toHaveBeenLastCalledWith(0);
    await user.click(
      screen.getByRole('button', { name: 'Jump to the live position' }),
    );
    expect(onSeek).toHaveBeenLastCalledWith(4);

    await user.click(screen.getByRole('button', { name: 'Play from here' }));
    expect(onRewind).toHaveBeenCalledOnce();
    expect((await axe(container)).violations).toEqual([]);
  });

  it('hides rewind when branching is blocked and renders the swap row', () => {
    const config: GameConfig = { ...double, pieRule: true };
    const log: GameAction[] = [{ type: 'place', node: 7 }, { type: 'swap' }];
    render(
      <MovesPanel
        timeline={buildTimeline(config, log)}
        total={log.length}
        currentPly={1}
        playerNames={names}
        canRewind={false}
        onSeek={vi.fn()}
        onRewind={vi.fn()}
      />,
    );

    expect(
      screen.getByRole('button', { name: 'Go to move 2: Grace swapped sides' }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: 'Play from here' }),
    ).not.toBeInTheDocument();
  });

  it('disables the transport on an empty log', () => {
    render(
      <MovesPanel
        timeline={buildTimeline(double, [])}
        total={0}
        currentPly={0}
        playerNames={names}
        canRewind
        onSeek={vi.fn()}
        onRewind={vi.fn()}
      />,
    );

    expect(screen.getByText(/no moves yet/i)).toBeInTheDocument();
    for (const name of [
      'Jump to the empty board',
      'Step one move back',
      'Step one move forward',
      'Jump to the live position',
    ]) {
      expect(screen.getByRole('button', { name })).toBeDisabled();
    }
  });
});
