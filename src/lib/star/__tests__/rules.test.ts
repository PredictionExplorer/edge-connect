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
  STAR_RULES_CANONICAL,
  STAR_RULES_HASH,
  STAR_RULES_HASH_ALGORITHM,
  STAR_RULES_SCHEMA_ID,
  STAR_RULES_VERSION,
} from '../rules';

const EXPECTED_RULES_HASH = 'fnv1a64:2da3783519381453';

function sha256(value: string): string {
  return `sha256:${createHash('sha256').update(value, 'utf8').digest('hex')}`;
}

describe('versioned rules contract', () => {
  it('pins the complete v2 gameplay contract', () => {
    expect(STAR_RULES_VERSION).toBe(2);
    expect(STAR_RULES_SCHEMA_ID).toBe('edgeconnect.star.rules.v2');
    expect(STAR_RULES_HASH_ALGORITHM).toBe('fnv1a64');
    expect(`${STAR_RULES_HASH_ALGORITHM}:${fnv1a64(STAR_RULES_CANONICAL)}`).toBe(
      STAR_RULES_HASH,
    );
    expect(STAR_RULES_HASH).toBe(EXPECTED_RULES_HASH);

    for (const requiredClause of [
      'rings=even:{4,6,8,10}',
      'node-order=x:1..r,s:0..4,y:0..x-1',
      'node-id=5*x*(x-1)/2+s*x+y',
      'csr-neighbor-order=edge-insertion-order',
      'action-wire=place(node)->node',
      'legal-order=empty-node-id-ascending',
      'native-action-layout=node-u-at-u',
      'terminal=full',
      'terminal-value=toMove-perspective:win=1,loss=-1,tie=invalid',
      'outcome-class=loss:0,win:1',
      'd5-rk=t+k*x(mod5*x)',
      'd5-fk=k*x-t(mod5*x)',
    ]) {
      expect(STAR_RULES_CANONICAL).toContain(requiredClause);
    }
  });

  it('keeps model features and node-only layout separately versioned', () => {
    expect(STAR_FEATURE_SCHEMA_ID).toBe(
      'edgeconnect.star.model-features.external.v2',
    );
    expect(STAR_ACTION_LAYOUT_SCHEMA_ID).toBe(
      'edgeconnect.star.action-layout.nodes-only.v1',
    );
    expect(STAR_RULES_CANONICAL).not.toContain(STAR_FEATURE_SCHEMA_ID);
    expect(STAR_RULES_CANONICAL).not.toContain(STAR_ACTION_LAYOUT_SCHEMA_ID);
  });
});

describe('v2 conformance export', () => {
  it('is deterministic and contains no legacy action or outcome fields', () => {
    const first = serializeStarConformance(createStarConformance());
    const second = serializeStarConformance(createStarConformance());
    expect(first).toBe(second);
    expect(sha256(first)).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(first).not.toContain('"pass"');
    expect(first).not.toContain('"draw"');
  });

  it('exports exact topology and D5 maps only for supported rings', () => {
    const conformance = createStarConformance();
    expect(conformance.schema).toBe(STAR_CONFORMANCE_SCHEMA_ID);
    expect(conformance.schemas).toEqual({
      rules: STAR_RULES_SCHEMA_ID,
      conformance: STAR_CONFORMANCE_SCHEMA_ID,
      modelFeatures: STAR_FEATURE_SCHEMA_ID,
      actionLayout: STAR_ACTION_LAYOUT_SCHEMA_ID,
    });
    expect(conformance.boards.map((board) => board.rings)).toEqual([
      4, 6, 8, 10,
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
        for (let node = 0; node < board.nodeCount; node++) {
          expect(symmetry.inverseMap[symmetry.map[node]]).toBe(node);
        }
      }
    }
  });

  it('contains full-board traces with binary terminal outcomes', () => {
    const conformance = createStarConformance();
    expect(conformance.games.map((game) => game.config.rings)).toEqual([
      4, 6, 8, 10,
    ]);
    for (const game of conformance.games) {
      const final = game.states.at(-1)!;
      expect(final.over).toBe(true);
      expect(final.stonesPlaced).toBe(final.stones.length);
      expect(game.actions).toHaveLength(final.stones.length);
      expect(game.actionCodes).toEqual(
        Array.from({ length: final.stones.length }, (_, node) => node),
      );
      expect(game.terminal.reason).toBe('board-full');
      expect([0, 1]).toContain(game.terminal.winner);
      expect(game.terminal.score.contestedPeries).toBe(0);
      expect(
        game.terminal.score.players[0].total +
          game.terminal.score.players[1].total,
      ).toBe(5 * game.config.rings + 1);
      expect([...game.terminal.valuesByPlayer].sort()).toEqual([-1, 1]);
      expect([...game.terminal.outcomeClassesByPlayer].sort()).toEqual([0, 1]);
      expect(game.terminal.valuePerspective.outcomeClass).toBe(
        game.terminal.valuePerspective.value === 1 ? 1 : 0,
      );
    }

    const residualOne = conformance.games.find(
      (game) => game.config.rings === 4,
    )!.states.at(-1)!;
    const residualZero = conformance.games.find(
      (game) => game.config.rings === 6,
    )!.states.at(-1)!;
    expect(residualOne).toMatchObject({ movesLeft: 1, midTurn: true });
    expect(residualZero).toMatchObject({ movesLeft: 0, midTurn: false });
  });

  it('exports AB/BA equivalence and nodes-only mixed layouts', () => {
    const conformance = createStarConformance();
    const pair = conformance.pairEquivalences[0];
    expect(pair.ab.semanticState).toEqual(pair.ba.semanticState);
    expect(pair.ab.lastMove).toBe(pair.pair.b);
    expect(pair.ba.lastMove).toBe(pair.pair.a);
    expect(pair.excludedPresentationFields).toEqual(['lastMove']);

    const batch = conformance.actionLayouts.mixedBatches[0];
    expect(batch.maximumNodes).toBe(275);
    expect(batch.batchActionCount).toBe(275);
    for (const row of batch.rows) {
      expect(row.native.actionCount).toBe(row.nodeCount);
      expect(row.padded.actionCount).toBe(batch.maximumNodes);
      expect(row.examples).toHaveLength(2);
      expect(row.examples.every((example) => example.wireCode >= 0)).toBe(
        true,
      );
    }
  });

  it('matches the checked-in deterministic v2 fixture', () => {
    const fixture = readFileSync(
      new URL('../../../../testdata/star/conformance-v2.json', import.meta.url),
      'utf8',
    );
    expect(fixture).toBe(serializeStarConformance(createStarConformance()));
  });
});
