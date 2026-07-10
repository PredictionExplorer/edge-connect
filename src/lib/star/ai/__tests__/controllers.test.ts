import { describe, expect, it } from 'vitest';
import {
  normalizeControllers,
  supportsAiControllers,
  type PlayerControllers,
} from '../controllers';
import type { GameConfig } from '../../game';

const double: GameConfig = {
  rings: 6,
  mode: 'double',
  pieRule: false,
  playerNames: ['A', 'B'],
};

describe('AI controller validation', () => {
  it('keeps independent valid controllers only for no-pie Double Star', () => {
    const controllers: PlayerControllers = ['server', 'local'];
    expect(supportsAiControllers(double)).toBe(true);
    expect(normalizeControllers(double, controllers)).toEqual(controllers);
  });

  it('forces persisted AI controllers back to human for unsupported variants', () => {
    expect(normalizeControllers({ ...double, mode: 'classic' }, ['server', 'local'])).toEqual([
      'human',
      'human',
    ]);
    expect(normalizeControllers({ ...double, pieRule: true }, ['server', 'local'])).toEqual([
      'human',
      'human',
    ]);
  });

  it('sanitizes malformed persisted controller tuples per player', () => {
    expect(normalizeControllers(double, ['server', 'remote'])).toEqual(['server', 'human']);
    expect(normalizeControllers(double, ['local'])).toEqual(['human', 'human']);
    expect(normalizeControllers(double, null)).toEqual(['human', 'human']);
  });
});
