import { describe, expect, it } from 'vitest';
import type { GameAction, GameConfig } from '../../game';
import {
  STAR_FEATURE_SCHEMA_HASH,
  acceptAiResponse,
  buildAiRequest,
  makeAiResponse,
  parseAiResponse,
} from '../protocol';

const config: GameConfig = {
  rings: 3,
  mode: 'double',
  pieRule: false,
  playerNames: ['A', 'B'],
};

describe('atomic AI protocol', () => {
  it('builds the exact opening semantic request', () => {
    const request = buildAiRequest(config, [], 'opening');
    expect(request.rulesHash).toBe('fnv1a64:cdb34fb02be82843');
    expect(request.featureSchemaHash).toBe('59a7da1c00bac4d2');
    expect(STAR_FEATURE_SCHEMA_HASH).toBe('59a7da1c00bac4d2');
    expect(request.state).toMatchObject({
      rings: 3,
      toMove: 0,
      movesLeft: 1,
      opening: true,
      passStreak: 0,
      terminal: false,
    });
    expect(request.state.stones).toHaveLength(30);
    expect(request.legalActions).toEqual([...Array.from({ length: 30 }, (_, i) => i), -1]);
    expect(request.actionLog).toEqual([]);
    expect(request.stateHash).toMatch(/^zobrist64:[0-9a-f]{16}$/);
  });

  it('represents the second stone as a new atomic decision for the same player', () => {
    const log: GameAction[] = [
      { type: 'place', node: 0 },
      { type: 'place', node: 1 },
    ];
    const request = buildAiRequest(config, log, 'second-stone');
    expect(request.actionLog).toEqual([0, 1]);
    expect(request.state.toMove).toBe(1);
    expect(request.state.movesLeft).toBe(1);
    expect(request.state.opening).toBe(false);
    expect(request.legalActions).not.toContain(0);
    expect(request.legalActions).not.toContain(1);
  });

  it('hashes semantically equivalent within-turn placement orders identically', () => {
    const ab = buildAiRequest(
      config,
      [
        { type: 'place', node: 0 },
        { type: 'place', node: 1 },
        { type: 'place', node: 2 },
      ],
      'ab',
    );
    const ba = buildAiRequest(
      config,
      [
        { type: 'place', node: 0 },
        { type: 'place', node: 2 },
        { type: 'place', node: 1 },
      ],
      'ba',
    );
    expect(ab.state).toEqual(ba.state);
    expect(ab.stateHash).toBe(ba.stateHash);
  });

  it('rejects stale and illegal responses at the final mutation gate', () => {
    const request = buildAiRequest(config, [], 'gate');
    const valid = makeAiResponse(request, { type: 'place', node: 0 });
    expect(
      acceptAiResponse(request, valid, config, [{ type: 'place', node: 2 }]),
    ).toMatchObject({ ok: false, code: 'stale' });

    const illegal = makeAiResponse(request, { type: 'place', node: 30 });
    expect(acceptAiResponse(request, illegal, config, [])).toMatchObject({
      ok: false,
      code: 'illegal',
    });
  });

  it('accepts exactly one atomic action and rejects turn-shaped payloads', () => {
    const request = buildAiRequest(config, [], 'atomic');
    expect(parseAiResponse(request, makeAiResponse(request, { type: 'pass' })).action).toEqual({
      type: 'pass',
    });
    expect(() =>
      parseAiResponse(request, {
        ...makeAiResponse(request, { type: 'pass' }),
        action: [{ type: 'place', node: 0 }, { type: 'place', node: 1 }],
      }),
    ).toThrow(/one atomic action/i);
  });

  it('refuses AI requests for classic and pie-rule configurations', () => {
    expect(() => buildAiRequest({ ...config, mode: 'classic' }, [], 'classic')).toThrow(
      /require Double/i,
    );
    expect(() => buildAiRequest({ ...config, pieRule: true }, [], 'pie')).toThrow(
      /pie rule disabled/i,
    );
  });
});
