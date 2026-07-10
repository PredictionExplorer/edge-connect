import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  createStarConformance,
  serializeStarConformance,
} from '../conformance';
import {
  fnv1a64,
  STAR_ACTION_LAYOUT_SCHEMA_ID,
  STAR_CONFORMANCE_SCHEMA_ID,
  STAR_FEATURE_SCHEMA_ID,
  STAR_PASS_ACTION_CODE,
  STAR_RULES_CANONICAL,
  STAR_RULES_HASH,
  STAR_RULES_HASH_ALGORITHM,
  STAR_RULES_SCHEMA_ID,
  STAR_RULES_VERSION,
} from '../rules';

const EXPECTED_RULES_HASH = 'fnv1a64:cdb34fb02be82843';
const EXPECTED_FIXTURE_SHA256 =
  'sha256:57e533c5c247b65f043546dd1d8abf43174c9257155cacdb1dcba98140292ea4';

function sha256(value: string): string {
  return `sha256:${createHash('sha256').update(value, 'utf8').digest('hex')}`;
}

describe('versioned rules contract', () => {
  it('pins the complete cross-language gameplay contract', () => {
    expect(STAR_RULES_VERSION).toBe(1);
    expect(STAR_RULES_SCHEMA_ID).toBe('edgeconnect.star.rules.v1');
    expect(STAR_RULES_HASH_ALGORITHM).toBe('fnv1a64');
    expect(`${STAR_RULES_HASH_ALGORITHM}:${fnv1a64(STAR_RULES_CANONICAL)}`).toBe(
      STAR_RULES_HASH,
    );
    expect(STAR_RULES_HASH).toBe(EXPECTED_RULES_HASH);

    for (const requiredClause of [
      'node-order=x:1..r,s:0..4,y:0..x-1',
      'node-id=5*x*(x-1)/2+s*x+y',
      'csr-neighbor-order=edge-insertion-order',
      'action-wire=place(node)->node,pass->-1',
      'legal-order=empty-node-id-ascending,pass-last',
      'full-terminal=decrement-movesLeft',
      'passStreak=2,movesLeft=2',
      'terminal-value=toMove-perspective',
      'd5-rk=t+k*x(mod5*x)',
      'd5-fk=k*x-t(mod5*x)',
    ]) {
      expect(STAR_RULES_CANONICAL).toContain(requiredClause);
    }
  });

  it('keeps model feature and action-layout schemas separate', () => {
    expect(STAR_FEATURE_SCHEMA_ID).toBe(
      'edgeconnect.star.model-features.external.v1',
    );
    expect(STAR_ACTION_LAYOUT_SCHEMA_ID).toBe(
      'edgeconnect.star.action-layout.nodes-then-pass.v1',
    );
    expect(STAR_RULES_CANONICAL).not.toContain(STAR_FEATURE_SCHEMA_ID);
    expect(STAR_RULES_CANONICAL).not.toContain(STAR_ACTION_LAYOUT_SCHEMA_ID);
  });
});

describe('Rust and Python conformance export', () => {
  it('is deterministic with a pinned fixture digest', () => {
    const first = serializeStarConformance(createStarConformance());
    const second = serializeStarConformance(createStarConformance());
    expect(first).toBe(second);
    expect(sha256(first)).toBe(EXPECTED_FIXTURE_SHA256);
  });

  it('exports exact topology and D5 maps for rings 3..12', () => {
    const conformance = createStarConformance();
    expect(conformance.schema).toBe(STAR_CONFORMANCE_SCHEMA_ID);
    expect(conformance.schemas).toEqual({
      rules: STAR_RULES_SCHEMA_ID,
      conformance: STAR_CONFORMANCE_SCHEMA_ID,
      modelFeatures: STAR_FEATURE_SCHEMA_ID,
      actionLayout: STAR_ACTION_LAYOUT_SCHEMA_ID,
    });
    expect(conformance.boards.map((board) => board.rings)).toEqual([
      3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    ]);

    for (const board of conformance.boards) {
      expect(board.sectorOf).toHaveLength(board.nodeCount);
      expect(board.ringOf).toHaveLength(board.nodeCount);
      expect(board.positionOf).toHaveLength(board.nodeCount);
      expect(board.perimeterMask).toHaveLength(board.nodeCount);
      expect(board.quarkMask).toHaveLength(board.nodeCount);
      expect(board.labels).toHaveLength(board.nodeCount);
      expect(board.adjacencyOffsets).toHaveLength(board.nodeCount + 1);
      expect(board.adjacencyOffsets[board.nodeCount]).toBe(
        board.adjacency.length,
      );
      expect(board.adjacency.length).toBe(2 * board.edgeCount);
      expect(board.symmetries.map((symmetry) => symmetry.id)).toEqual([
        'r0',
        'r1',
        'r2',
        'r3',
        'r4',
        'f0',
        'f1',
        'f2',
        'f3',
        'f4',
      ]);
      for (const symmetry of board.symmetries) {
        expect(symmetry.map).toHaveLength(board.nodeCount);
        expect(symmetry.inverseMap).toHaveLength(board.nodeCount);
        for (let node = 0; node < board.nodeCount; node++) {
          expect(symmetry.inverseMap[symmetry.map[node]]).toBe(node);
        }
      }
    }
  });

  it('contains sparse static-aliveness scoring vectors', () => {
    const conformance = createStarConformance();
    const singlePerimeter = conformance.scores.filter((score) =>
      score.id.endsWith('-single-perimeter'),
    );
    const twoPerimeter = conformance.scores.filter((score) =>
      score.id.endsWith('-two-perimeter-star'),
    );
    expect(singlePerimeter).toHaveLength(10);
    expect(twoPerimeter).toHaveLength(10);
    for (const score of singlePerimeter) {
      expect(score.expected.players[0].stars).toBe(0);
      expect(score.expected.aliveStone.every((alive) => alive === 0)).toBe(true);
      expect(score.expected.contestedPeries).toBe(5 * score.rings);
    }
    for (const score of twoPerimeter) {
      expect(score.expected.players[0].stars).toBe(1);
      expect(score.expected.aliveStone.filter(Boolean)).toHaveLength(2);
    }
  });

  it('exports both board-full residual parities and double-pass metadata', () => {
    const conformance = createStarConformance();
    const byId = (id: string) => {
      const game = conformance.games.find((candidate) => candidate.id === id);
      if (!game) throw new Error(`missing game vector ${id}`);
      return game;
    };
    const residualOne = byId('rings-3-board-full-residual-1');
    const residualZero = byId('rings-5-board-full-residual-0');
    const doublePass = byId('double-midturn-two-pass-terminal');

    const one = residualOne.states[residualOne.states.length - 1];
    const beforeOne = residualOne.states[residualOne.states.length - 2];
    expect(one).toMatchObject({
      over: true,
      movesLeft: 1,
      midTurn: true,
      passStreak: 0,
    });
    expect(one.toMove).toBe(beforeOne.toMove);
    expect(one.turnCount).toBe(beforeOne.turnCount);
    expect(one.currentTurnMoves).toHaveLength(1);
    expect(one.lastMove).toBe(one.stones.length - 1);
    expect(residualOne.terminal.reason).toBe('board-full');
    expect(residualOne.actionCodes).not.toContain(STAR_PASS_ACTION_CODE);

    const zero = residualZero.states[residualZero.states.length - 1];
    const beforeZero = residualZero.states[residualZero.states.length - 2];
    expect(zero).toMatchObject({
      over: true,
      movesLeft: 0,
      midTurn: false,
      passStreak: 0,
    });
    expect(zero.toMove).toBe(beforeZero.toMove);
    expect(zero.turnCount).toBe(beforeZero.turnCount);
    expect(zero.currentTurnMoves).toHaveLength(2);
    expect(zero.lastMove).toBe(zero.stones.length - 1);
    expect(residualZero.terminal.reason).toBe('board-full');

    const passed = doublePass.states[doublePass.states.length - 1];
    const beforeSecondPass =
      doublePass.states[doublePass.states.length - 2];
    expect(passed).toMatchObject({
      over: true,
      movesLeft: 2,
      passStreak: 2,
    });
    expect(passed.toMove).toBe(beforeSecondPass.toMove);
    expect(passed.turnCount).toBe(beforeSecondPass.turnCount);
    expect(passed.midTurn).toBe(false);
    expect(passed.currentTurnMoves).toEqual([]);
    expect(passed.lastMove).toBe(beforeSecondPass.lastMove);
    expect(doublePass.actionCodes.slice(-2)).toEqual([
      STAR_PASS_ACTION_CODE,
      STAR_PASS_ACTION_CODE,
    ]);
    expect(doublePass.terminal.reason).toBe('double-pass');
    expect(doublePass.terminal.winner).toBe(0);
    expect(doublePass.terminal.valuePerspective).toEqual({
      kind: 'toMove',
      player: 1,
      value: -1,
      wdlClass: 0,
      scoreMargin: doublePass.terminal.scoreMarginsByPlayer[1],
    });

    for (const game of conformance.games) {
      const final = game.states[game.states.length - 1];
      expect(game.terminal.winner).toBe(game.terminal.score.leader);
      expect(game.terminal.valuePerspective.player).toBe(final.toMove);
      expect(game.terminal.valuePerspective.value).toBe(
        game.terminal.valuesByPlayer[final.toMove],
      );
      expect(game.terminal.valuePerspective.wdlClass).toBe(
        game.terminal.valuePerspective.value + 1,
      );
    }
  });

  it('exports AB/BA semantic equivalence and mixed action layouts', () => {
    const conformance = createStarConformance();
    const pair = conformance.pairEquivalences[0];
    expect(pair.ab.semanticState).toEqual(pair.ba.semanticState);
    expect(pair.ab.lastMove).toBe(pair.pair.b);
    expect(pair.ba.lastMove).toBe(pair.pair.a);
    expect(pair.ab.lastMove).not.toBe(pair.ba.lastMove);
    expect(pair.excludedPresentationFields).toEqual(['lastMove']);

    expect(conformance.actionEncoding.passCode).toBe(STAR_PASS_ACTION_CODE);
    for (const batch of conformance.actionLayouts.mixedBatches) {
      for (const row of batch.rows) {
        expect(row.native.passSlot).toBe(row.nodeCount);
        expect(row.padded.passSlot).toBe(batch.maximumNodes);
        expect(row.examples[2]).toMatchObject({
          action: { type: 'pass' },
          wireCode: STAR_PASS_ACTION_CODE,
          nativeIndex: row.nodeCount,
          paddedIndex: batch.maximumNodes,
        });
      }
    }
  });

  it('matches the checked-in deterministic JSON fixture', () => {
    const fixture = readFileSync(
      new URL('../../../../testdata/star/conformance-v1.json', import.meta.url),
      'utf8',
    );
    const generated = serializeStarConformance(createStarConformance());
    expect(sha256(fixture)).toBe(EXPECTED_FIXTURE_SHA256);
    expect(fixture).toBe(generated);
  });
});
