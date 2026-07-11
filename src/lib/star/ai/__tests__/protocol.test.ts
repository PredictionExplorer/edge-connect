import { describe, expect, it, vi } from 'vitest';
import { fnv1a64 } from '../../rules';
import { initialState, replay, type GameAction, type GameConfig } from '../../game';
import { StarAiError, type StarAiErrorCode } from '../errors';
import {
  STAR_AI_PROTOCOL_SCHEMA_ID,
  STAR_AI_PROTOCOL_VERSION,
  STAR_FEATURE_CONTRACT,
  STAR_FEATURE_SCHEMA_HASH,
  STAR_FEATURE_SCHEMA_VERSION,
  acceptAiResponse,
  actionToCode,
  buildAiRequest,
  codeToAction,
  legalActionCodes,
  makeAiResponse,
  newAiRequestId,
  parseAiResponse,
  semanticStateFromGame,
  semanticStateHash,
} from '../protocol';

const config: GameConfig = {
  rings: 4,
  mode: 'double',
  pieRule: false,
  playerNames: ['A', 'B'],
};

function expectAiError(
  operation: () => unknown,
  code: StarAiErrorCode,
  message: string,
) {
  try {
    operation();
    throw new Error('expected StarAiError');
  } catch (error) {
    expect(error).toBeInstanceOf(StarAiError);
    expect(error).toMatchObject({ code, message });
  }
}

describe('placement-only AI protocol v2', () => {
  it('builds the exact opening semantic request', () => {
    const request = buildAiRequest(config, [], 'opening');
    expect(request).toMatchObject({
      schema: STAR_AI_PROTOCOL_SCHEMA_ID,
      version: STAR_AI_PROTOCOL_VERSION,
      rulesHash: 'fnv1a64:2da3783519381453',
      featureSchemaVersion: 3,
      state: {
        rings: 4,
        toMove: 0,
        movesLeft: 1,
        opening: true,
        terminal: false,
      },
    });
    expect(STAR_AI_PROTOCOL_VERSION).toBe(2);
    expect(STAR_FEATURE_SCHEMA_VERSION).toBe(3);
    expect(STAR_FEATURE_SCHEMA_HASH).toBe(fnv1a64(STAR_FEATURE_CONTRACT));
    expect(STAR_FEATURE_SCHEMA_HASH).toBe('6b5b00f638e9c16b');
    expect('passStreak' in request.state).toBe(false);
    expect(request.state.stones).toHaveLength(50);
    expect(request.legalActions).toEqual(
      Array.from({ length: 50 }, (_, node) => node),
    );
    expect(request.actionLog).toEqual([]);
    expect(request.stateHash).toMatch(/^zobrist64:[0-9a-f]{16}$/);
  });

  it('represents each placement as a separate atomic decision', () => {
    const log: GameAction[] = [
      { type: 'place', node: 0 },
      { type: 'place', node: 1 },
    ];
    const request = buildAiRequest(config, log, 'second-stone');
    expect(request.actionLog).toEqual([0, 1]);
    expect(request.state.toMove).toBe(1);
    expect(request.state.movesLeft).toBe(1);
    expect(request.legalActions).not.toContain(0);
    expect(request.legalActions).not.toContain(1);
    expect(request.legalActions.every((action) => action >= 0)).toBe(true);
  });

  it('hashes semantically equivalent within-turn orders identically', () => {
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

  it('round-trips only nonnegative placement codes', () => {
    expect(actionToCode({ type: 'place', node: 12 })).toBe(12);
    expect(codeToAction(12)).toEqual({ type: 'place', node: 12 });
    for (const code of [-1, -0.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expectAiError(
        () => codeToAction(code),
        'protocol',
        `Invalid atomic action code: ${String(code)}.`,
      );
    }
  });

  it('accepts one placement and rejects legacy or turn-shaped payloads', () => {
    const request = buildAiRequest(config, [], 'atomic');
    expect(
      parseAiResponse(
        request,
        makeAiResponse(request, { type: 'place', node: 0 }),
      ).action,
    ).toEqual({ type: 'place', node: 0 });
    expect(() =>
      parseAiResponse(request, {
        ...makeAiResponse(request, { type: 'place', node: 0 }),
        action: [{ type: 'place', node: 0 }, { type: 'place', node: 1 }],
      }),
    ).toThrow(/one atomic action/i);
    expect(() =>
      parseAiResponse(request, {
        ...makeAiResponse(request, { type: 'place', node: 0 }),
        action: { type: 'pass' },
      }),
    ).toThrow(/one atomic action/i);
  });

  it('rejects unsupported variants, rings, and web-only swaps', () => {
    expect(() =>
      buildAiRequest({ ...config, mode: 'classic' }, [], 'classic'),
    ).toThrow(/require Double/i);
    expect(() => buildAiRequest({ ...config, pieRule: true }, [], 'pie')).toThrow(
      /pie rule disabled/i,
    );
    expect(() => buildAiRequest({ ...config, rings: 5 }, [], 'rings')).toThrow(
      /one of 4, 6, 8, 10/,
    );
    expect(() =>
      buildAiRequest(config, [{ type: 'swap' }], 'swap'),
    ).toThrow(/outside the AI protocol/);
  });

  it('emits no legal actions and refuses requests after a full board', () => {
    const terminalLog: GameAction[] = Array.from(
      { length: 50 },
      (_, node) => ({ type: 'place', node }) as const,
    );
    const terminal = replay(config, terminalLog);
    expect(terminal.over).toBe(true);
    expect(legalActionCodes(terminal)).toEqual([]);
    expectAiError(
      () => buildAiRequest(config, terminalLog, 'terminal'),
      'protocol',
      'Cannot request an action for a terminal position.',
    );
  });

  it('rejects stale and illegal responses at the final mutation gate', () => {
    const request = buildAiRequest(config, [], 'gate');
    const valid = makeAiResponse(request, { type: 'place', node: 0 });
    expect(
      acceptAiResponse(request, valid, config, [{ type: 'place', node: 2 }]),
    ).toMatchObject({ ok: false, code: 'stale' });

    const illegal = makeAiResponse(request, { type: 'place', node: 50 });
    expect(acceptAiResponse(request, illegal, config, [])).toMatchObject({
      ok: false,
      code: 'illegal',
    });
  });

  it('strictly rejects malformed response shapes and identities', () => {
    const request = buildAiRequest(config, [], 'strict-response');
    const valid = makeAiResponse(request, { type: 'place', node: 0 });
    const cases: Array<[unknown, StarAiErrorCode, string]> = [
      [null, 'protocol', 'AI response must be an object.'],
      [{ ...valid, schema: 'wrong' }, 'protocol', 'AI response schema or rules hash is incompatible.'],
      [{ ...valid, version: 1 }, 'protocol', 'AI response schema or rules hash is incompatible.'],
      [{ ...valid, rulesHash: 'wrong' }, 'protocol', 'AI response schema or rules hash is incompatible.'],
      [{ ...valid, requestId: 'old' }, 'stale', 'AI response belongs to an obsolete position.'],
      [{ ...valid, stateHash: 'old' }, 'stale', 'AI response belongs to an obsolete position.'],
      [{ ...valid, actions: [] }, 'protocol', 'AI response must not contain a multi-action turn.'],
      [{ ...valid, action: { type: 'swap' } }, 'protocol', 'AI response must contain one atomic action.'],
      [{ ...valid, action: { type: 'place' } }, 'protocol', 'A placement must contain a non-negative node id.'],
      [{ ...valid, action: { type: 'place', node: -1 } }, 'protocol', 'A placement must contain a non-negative node id.'],
      [{ ...valid, action: { type: 'place', node: 0.5 } }, 'protocol', 'A placement must contain a non-negative node id.'],
      [
        { ...valid, action: { type: 'place', node: 0, extra: true } },
        'protocol',
        'A placement action contains unknown fields.',
      ],
      [
        { ...valid, action: { type: 'place', node: 50 } },
        'illegal',
        'AI returned an illegal atomic action.',
      ],
    ];
    for (const [payload, code, message] of cases) {
      expectAiError(() => parseAiResponse(request, payload), code, message);
    }
  });

  it('normalizes final-gate failures and accepts a current legal response', () => {
    const request = buildAiRequest(config, [], 'acceptance');
    const valid = makeAiResponse(request, { type: 'place', node: 0 });
    expect(acceptAiResponse(request, valid, config, [])).toMatchObject({
      ok: true,
      action: { type: 'place', node: 0 },
    });
    expect(
      acceptAiResponse(request, valid, { ...config, mode: 'classic' }, []),
    ).toMatchObject({ ok: false, code: 'stale' });
    expect(acceptAiResponse(request, null, config, [])).toMatchObject({
      ok: false,
      code: 'protocol',
    });
  });

  it('creates unique fallback identities and guards missing BigInt', () => {
    vi.stubGlobal('crypto', undefined);
    vi.spyOn(Date, 'now').mockReturnValue(1234);
    expect(newAiRequestId()).not.toBe(newAiRequestId());

    const semantic = semanticStateFromGame(initialState(config));
    vi.stubGlobal('BigInt', undefined);
    expectAiError(
      () => semanticStateHash(semantic),
      'unavailable',
      'AI controllers require BigInt browser support.',
    );
  });
});
