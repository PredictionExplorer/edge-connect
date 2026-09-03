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

describe('atomic AI protocol v3', () => {
  it('builds the exact opening semantic request', () => {
    const request = buildAiRequest(config, [], 'opening');
    expect(request).toMatchObject({
      schema: STAR_AI_PROTOCOL_SCHEMA_ID,
      version: STAR_AI_PROTOCOL_VERSION,
      rulesHash: 'fnv1a64:a5d932b0ef8354e8',
      featureSchemaVersion: 4,
      state: {
        rings: 4,
        toMove: 0,
        movesLeft: 1,
        opening: true,
        terminal: false,
        mode: 'double',
        handicap: 1,
        pie: false,
        swapAvailable: false,
        swapped: false,
        history: {
          currentTurn: [],
          previousTurn: [],
          ownPreviousTurn: [],
          handicapStones: [],
        },
      },
    });
    expect(STAR_AI_PROTOCOL_VERSION).toBe(3);
    expect(STAR_FEATURE_SCHEMA_VERSION).toBe(4);
    expect(STAR_FEATURE_SCHEMA_HASH).toBe(fnv1a64(STAR_FEATURE_CONTRACT));
    expect(STAR_FEATURE_SCHEMA_HASH).toBe('cb0e1e89a6ce3540');
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
    expect(request.state.history).toEqual({
      currentTurn: [1],
      previousTurn: [0],
      ownPreviousTurn: [],
      handicapStones: [0],
    });
    expect(request.legalActions).not.toContain(0);
    expect(request.legalActions).not.toContain(1);
    expect(request.legalActions.every((action) => action >= 0)).toBe(true);
  });

  it('carries every rule variant and codes the swap as the node count', () => {
    const classic = buildAiRequest({ ...config, mode: 'classic' }, [], 'classic');
    expect(classic.state.mode).toBe('classic');
    const handicap = buildAiRequest(
      { ...config, handicap: 3 },
      [{ type: 'place', node: 4 }],
      'handicap',
    );
    expect(handicap.state).toMatchObject({
      handicap: 3,
      opening: true,
      movesLeft: 2,
      toMove: 0,
      history: {
        currentTurn: [4],
        previousTurn: [],
        ownPreviousTurn: [],
        handicapStones: [4],
      },
    });
    const pie = buildAiRequest(
      { ...config, pieRule: true },
      [{ type: 'place', node: 7 }],
      'pie-responder',
    );
    expect(pie.state).toMatchObject({ pie: true, swapAvailable: true, toMove: 1 });
    const swapped = buildAiRequest(
      { ...config, pieRule: true },
      [{ type: 'place', node: 7 }, { type: 'swap' }],
      'pie-swapped',
    );
    expect(swapped.actionLog).toEqual([7, 50]);
    expect(swapped.state).toMatchObject({ swapAvailable: false, swapped: true, toMove: 0 });
    expect(swapped.state.stones[7]).toBe(1);
    // The opener's pie decision and the responder's swap both enter the hash.
    const openerPending = buildAiRequest({ ...config, pieRule: true }, [], 'pie-opener');
    expect(openerPending.stateHash).not.toBe(
      buildAiRequest(config, [], 'plain-opener').stateHash,
    );
    expect(pie.stateHash).not.toBe(
      buildAiRequest(config, [{ type: 'place', node: 7 }], 'plain-responder').stateHash,
    );
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

  it('round-trips placement codes and the swap code', () => {
    expect(actionToCode({ type: 'place', node: 12 }, 50)).toBe(12);
    expect(actionToCode({ type: 'swap' }, 50)).toBe(50);
    expect(codeToAction(12, 50)).toEqual({ type: 'place', node: 12 });
    expect(codeToAction(50, 50)).toEqual({ type: 'swap' });
    for (const code of [-1, -0.5, 51, Number.NaN, Number.POSITIVE_INFINITY]) {
      expectAiError(
        () => codeToAction(code, 50),
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

  it('rejects configurations outside the rules family and illegal swaps', () => {
    expect(() =>
      buildAiRequest({ ...config, handicap: 10 }, [], 'handicap'),
    ).toThrow(/invalid rule variant/i);
    expect(() =>
      buildAiRequest({ ...config, pieRule: true, handicap: 2 }, [], 'pie-handicap'),
    ).toThrow(/invalid rule variant/i);
    expect(() => buildAiRequest({ ...config, rings: 5 }, [], 'rings')).toThrow(
      /one of 4, 6, 8, 10/,
    );
    expect(() => buildAiRequest(config, [{ type: 'swap' }], 'swap')).toThrow(
      /illegal action/,
    );
  });

  it('accepts a swap only while the responder may take it', () => {
    const pie = buildAiRequest(
      { ...config, pieRule: true },
      [{ type: 'place', node: 7 }],
      'pie-swap',
    );
    const swap = makeAiResponse(pie, { type: 'swap' });
    expect(parseAiResponse(pie, swap).action).toEqual({ type: 'swap' });
    expect(
      acceptAiResponse(pie, swap, { ...config, pieRule: true }, [{ type: 'place', node: 7 }]),
    ).toMatchObject({ ok: true, action: { type: 'swap' } });
    const plain = buildAiRequest(config, [{ type: 'place', node: 7 }], 'plain');
    expectAiError(
      () => parseAiResponse(plain, makeAiResponse(plain, { type: 'swap' })),
      'illegal',
      'AI returned an illegal atomic action.',
    );
    expectAiError(
      () => parseAiResponse(pie, { ...swap, action: { type: 'swap', node: 3 } }),
      'protocol',
      'A swap action contains unknown fields.',
    );
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
      [{ ...valid, action: { type: 'swap' } }, 'illegal', 'AI returned an illegal atomic action.'],
      [{ ...valid, action: { type: 'pass' } }, 'protocol', 'AI response must contain one atomic action.'],
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
    expect(
      acceptAiResponse(request, valid, { ...config, handicap: 12 }, []),
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
