import { describe, expect, it, vi } from 'vitest';
import { initialState, replay, type GameAction, type GameConfig } from '../../game';
import { StarAiError, type StarAiErrorCode } from '../errors';
import {
  STAR_FEATURE_SCHEMA_HASH,
  actionToCode,
  acceptAiResponse,
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
  rings: 3,
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

  it('creates unique fallback identities and round-trips atomic action codes', () => {
    vi.stubGlobal('crypto', undefined);
    vi.spyOn(Date, 'now').mockReturnValue(1234);
    const first = newAiRequestId();
    const second = newAiRequestId();
    expect(first).toMatch(/^star-ai-ya-/);
    expect(second).not.toBe(first);
    expect(actionToCode({ type: 'pass' })).toBe(-1);
    expect(actionToCode({ type: 'place', node: 12 })).toBe(12);
    expect(codeToAction(-1)).toEqual({ type: 'pass' });
    expect(codeToAction(12)).toEqual({ type: 'place', node: 12 });
    for (const code of [-2, -0.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expectAiError(
        () => codeToAction(code),
        'protocol',
        `Invalid atomic action code: ${String(code)}.`,
      );
    }
  });

  it('rejects hashing without BigInt and emits no legal actions after termination', () => {
    const initial = initialState(config);
    const semantic = semanticStateFromGame(initial);
    vi.stubGlobal('BigInt', undefined);
    expectAiError(
      () => semanticStateHash(semantic),
      'unavailable',
      'AI controllers require BigInt browser support.',
    );
    vi.unstubAllGlobals();

    const terminalLog: GameAction[] = [{ type: 'pass' }, { type: 'pass' }];
    const terminal = replay(config, terminalLog);
    expect(legalActionCodes(terminal)).toEqual([]);
    expectAiError(
      () => buildAiRequest(config, terminalLog, 'terminal'),
      'protocol',
      'Cannot request an action for a terminal position.',
    );
    expectAiError(
      () => buildAiRequest(config, [{ type: 'swap' }], 'swap'),
      'protocol',
      'Pie-rule swaps are outside the AI protocol.',
    );
  });

  it('strictly rejects every malformed response shape and identity field', () => {
    const request = buildAiRequest(config, [], 'strict-response');
    const valid = makeAiResponse(request, { type: 'pass' });
    const cases: Array<[unknown, StarAiErrorCode, string]> = [
      [null, 'protocol', 'AI response must be an object.'],
      [[], 'protocol', 'AI response must be an object.'],
      [{ ...valid, schema: 'wrong' }, 'protocol', 'AI response schema or rules hash is incompatible.'],
      [{ ...valid, version: 2 }, 'protocol', 'AI response schema or rules hash is incompatible.'],
      [{ ...valid, rulesHash: 'wrong' }, 'protocol', 'AI response schema or rules hash is incompatible.'],
      [{ ...valid, requestId: 'old' }, 'stale', 'AI response belongs to an obsolete position.'],
      [{ ...valid, stateHash: 'old' }, 'stale', 'AI response belongs to an obsolete position.'],
      [{ ...valid, actions: [] }, 'protocol', 'AI response must not contain a multi-action turn.'],
      [{ ...valid, action: null }, 'protocol', 'AI response must contain one atomic action.'],
      [{ ...valid, action: [] }, 'protocol', 'AI response must contain one atomic action.'],
      [{ ...valid, action: { type: 'swap' } }, 'protocol', 'AI response must contain one atomic action.'],
      [{ ...valid, action: { type: 'pass', node: null } }, 'protocol', 'A pass action cannot include a node.'],
      [{ ...valid, action: { type: 'pass', extra: true } }, 'protocol', 'A pass action cannot include a node.'],
      [{ ...valid, action: { type: 'place' } }, 'protocol', 'A placement must contain a non-negative node id.'],
      [{ ...valid, action: { type: 'place', node: '0' } }, 'protocol', 'A placement must contain a non-negative node id.'],
      [{ ...valid, action: { type: 'place', node: -1 } }, 'protocol', 'A placement must contain a non-negative node id.'],
      [{ ...valid, action: { type: 'place', node: 0.5 } }, 'protocol', 'A placement must contain a non-negative node id.'],
      [
        { ...valid, action: { type: 'place', node: 0, extra: true } },
        'protocol',
        'A placement action contains unknown fields.',
      ],
      [
        { ...valid, action: { type: 'place', node: 30 } },
        'illegal',
        'AI returned an illegal atomic action.',
      ],
    ];
    for (const [payload, code, message] of cases) {
      expectAiError(() => parseAiResponse(request, payload), code, message);
    }
  });

  it('normalizes every final-gate failure and accepts a current legal response', () => {
    const request = buildAiRequest(config, [], 'acceptance');
    const valid = makeAiResponse(request, { type: 'place', node: 0 });
    expect(acceptAiResponse(request, valid, config, [])).toMatchObject({
      ok: true,
      action: { type: 'place', node: 0 },
    });
    expect(
      acceptAiResponse(request, valid, { ...config, mode: 'classic' }, []),
    ).toEqual({
      ok: false,
      code: 'stale',
      message: 'The game changed before AI replied.',
    });
    expect(
      acceptAiResponse(
        request,
        valid,
        config,
        [
          { type: 'place', node: 0 },
          { type: 'place', node: 0 },
        ],
      ),
    ).toEqual({
      ok: false,
      code: 'stale',
      message: 'The game changed before AI replied.',
    });
    expect(acceptAiResponse(request, null, config, [])).toEqual({
      ok: false,
      code: 'protocol',
      message: 'AI response must be an object.',
    });
    expect(
      acceptAiResponse(
        request,
        new Proxy(valid, {
          get() {
            throw new TypeError('hostile getter');
          },
        }),
        config,
        [],
      ),
    ).toEqual({
      ok: false,
      code: 'protocol',
      message: 'AI response is invalid.',
    });

    const permissive = { ...request, legalActions: [...request.legalActions, 30] };
    const outsideBoard = makeAiResponse(permissive, { type: 'place', node: 30 });
    expect(acceptAiResponse(permissive, outsideBoard, config, [])).toEqual({
      ok: false,
      code: 'illegal',
      message: 'AI returned an illegal atomic action.',
    });
  });
});
