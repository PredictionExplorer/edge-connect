import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const state = vi.hoisted(() => ({ mounted: true, phase: 'setup' as 'setup' | 'playing' }));

vi.mock('@/components/GameScreen', () => ({
  GameScreen: () => <div>game-screen</div>,
}));
vi.mock('@/components/SetupScreen', () => ({
  SetupScreen: () => <div>setup-screen</div>,
}));
vi.mock('@/components/Starfield', () => ({
  Starfield: () => <div>starfield</div>,
}));
vi.mock('@/lib/store', () => ({
  useMounted: () => state.mounted,
  useAppStore: (selector: (value: typeof state) => unknown) => selector(state),
}));

import Home from './page';

describe('Home phase routing', () => {
  beforeEach(() => {
    state.mounted = true;
    state.phase = 'setup';
  });

  it('always renders decoration and gates setup until hydration', () => {
    state.mounted = false;
    const { container } = render(<Home />);
    expect(screen.getByText('starfield')).toBeInTheDocument();
    expect(screen.queryByText('setup-screen')).not.toBeInTheDocument();
    expect(container.querySelector('.app-skeleton')).toBeInTheDocument();
  });

  it('selects setup and game screens from the persisted phase', () => {
    const { rerender } = render(<Home />);
    expect(screen.getByText('setup-screen')).toBeInTheDocument();
    state.phase = 'playing';
    rerender(<Home />);
    expect(screen.getByText('game-screen')).toBeInTheDocument();
  });

  it('resets document scroll whenever the persisted phase changes', () => {
    const scrollTo = vi.spyOn(window, 'scrollTo');
    const { rerender } = render(<Home />);
    scrollTo.mockClear();

    state.phase = 'playing';
    rerender(<Home />);

    expect(scrollTo).toHaveBeenCalledWith({
      top: 0,
      left: 0,
      behavior: 'auto',
    });
    scrollTo.mockRestore();
  });
});

