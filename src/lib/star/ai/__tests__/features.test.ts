import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { getBoard } from '../../board';
import type { GameConfig } from '../../game';
import { STAR_RULES_HASH } from '../../rules';
import {
  STAR_GLOBAL_FEATURE_DIM,
  STAR_GLOBAL_FEATURE_NAMES,
  STAR_NODE_FEATURE_DIM,
  STAR_NODE_FEATURE_NAMES,
  actionCodeToModelIndex,
  encodeStarFeatures,
  float16BitsToNumber,
  float16ToFloat32Array,
  float32ToFloat16Array,
  modelIndexToActionCode,
  numberToFloat16Bits,
} from '../features';
import {
  STAR_FEATURE_SCHEMA_HASH,
  STAR_FEATURE_SCHEMA_VERSION,
  buildAiRequest,
} from '../protocol';

const config: GameConfig = {
  rings: 4,
  mode: 'double',
  pieRule: false,
  playerNames: ['A', 'B'],
};

describe('schema-v3 browser features', () => {
  it('uses exact feature dimensions and a nodes-only action layout', () => {
    const state = buildAiRequest(config, [], 'features').state;
    const encoded = encodeStarFeatures(state);
    const board = getBoard(4);

    expect(STAR_NODE_FEATURE_DIM).toBe(19);
    expect(STAR_GLOBAL_FEATURE_DIM).toBe(25);
    expect(encoded.nodeFeatures).toHaveLength(board.n * 19);
    expect(encoded.globalFeatures).toHaveLength(25);
    expect(Array.from(encoded.rings, Number)).toEqual([4]);
    expect(encoded.neighborIndex).toHaveLength(board.n * encoded.maxDegree);
    expect(encoded.legalActionMask).toHaveLength(board.n);
    expect(Array.from(encoded.legalActionMask)).toEqual(new Array(board.n).fill(1));
    expect(actionCodeToModelIndex(7, board.n)).toBe(7);
    expect(modelIndexToActionCode(7, board.n)).toBe(7);
    expect(() => actionCodeToModelIndex(-1, board.n)).toThrow(/nodes-only/);
    expect(() => modelIndexToActionCode(board.n, board.n)).toThrow(
      /outside the action layout/,
    );
  });

  it('matches Python opening features and topology edge classes', () => {
    const state = buildAiRequest(config, [], 'opening-features').state;
    const encoded = encodeStarFeatures(state);
    const board = getBoard(4);
    const degree = board.adjOff[1] - board.adjOff[0];

    expect(Array.from(encoded.nodeFeatures.slice(0, 19))).toEqual([
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      Math.fround(1 / 4),
      0,
      Math.fround(degree / encoded.maxDegree),
      1,
      1,
      0,
      0,
      0,
      0,
    ]);
    expect(Array.from(encoded.globalFeatures)).toEqual([
      Math.fround(4 / 10),
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      Math.fround(1 / 9),
      0,
      Math.fround(1 / 9),
      0,
      0,
      1,
      0,
    ]);

    const neighborOne = Array.from(
      encoded.neighborIndex.slice(0, encoded.maxDegree),
      Number,
    ).indexOf(1);
    expect(neighborOne).toBeGreaterThanOrEqual(0);
    expect(encoded.neighborEdgeType[neighborOne]).toBe(BigInt(2));
  });

  it('canonicalizes stones to the current-player perspective', () => {
    const state = buildAiRequest(
      config,
      [{ type: 'place', node: 0 }],
      'perspective',
    ).state;
    const encoded = encodeStarFeatures(state);
    const nodeZero = Array.from(encoded.nodeFeatures.slice(0, 15));
    expect(state.toMove).toBe(1);
    expect(nodeZero[1]).toBe(0);
    expect(nodeZero[2]).toBe(1);
    expect(encoded.legalActionMask[0]).toBe(0);
  });

  it('normalizes terminal score support by 151 and masks every action', () => {
    const board = getBoard(4);
    const encoded = encodeStarFeatures({
      rings: 4,
      stones: new Array(board.n).fill(0),
      toMove: 0,
      movesLeft: 1,
      opening: false,
      terminal: true,
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
    });
    expect(Array.from(encoded.legalActionMask)).toEqual(
      new Array(board.n).fill(0),
    );
    expect(encoded.globalFeatures[7]).toBeCloseTo(19 / 151);
    expect(encoded.globalFeatures[8]).toBeCloseTo(2 / 151);
    expect(encoded.globalFeatures[9]).toBeCloseTo(17 / 151);
  });

  it('converts schema-v3 floating features to browser FP16 tensors', () => {
    expect(numberToFloat16Bits(0)).toBe(0x0000);
    expect(numberToFloat16Bits(1)).toBe(0x3c00);
    expect(numberToFloat16Bits(-2)).toBe(0xc000);
    expect(numberToFloat16Bits(65_504)).toBe(0x7bff);
    expect(float16BitsToNumber(0x3800)).toBe(0.5);

    const encoded = float32ToFloat16Array(new Float32Array([0, 1, -2, 0.5]));
    expect(Array.from(encoded)).toEqual([0x0000, 0x3c00, 0xc000, 0x3800]);
    expect(Array.from(float16ToFloat32Array(encoded))).toEqual([0, 1, -2, 0.5]);
  });
});

describe('cross-language feature parity', () => {
  it('reproduces every Python schema-v4 vector of the checked-in fixture', () => {
    const fixture = JSON.parse(
      readFileSync(
        new URL('../../../../../testdata/star/features-v4.json', import.meta.url),
        'utf8',
      ),
    ) as {
      featureSchemaVersion: number;
      featureSchemaHash: string;
      rulesHash: string;
      nodeFeatureNames: string[];
      globalFeatureNames: string[];
      positions: Array<{
        game: string;
        stateIndex: number;
        rings: number;
        nodeFeatures: number[];
        globalFeatures: number[];
        legalActionMask: number[];
      }>;
    };
    const conformance = JSON.parse(
      readFileSync(
        new URL('../../../../../testdata/star/conformance-v3.json', import.meta.url),
        'utf8',
      ),
    ) as {
      games: Array<{
        id: string;
        config: { rings: number; mode: 'classic' | 'double'; pieRule: boolean; handicap: number };
        states: Array<{
          stones: number[];
          toMove: 0 | 1;
          movesLeft: number;
          opening: boolean;
          over: boolean;
          canSwap: boolean;
          swapped: boolean;
          currentTurnMoves: number[];
          previousTurnMoves: number[];
          ownPreviousTurnMoves: number[];
          handicapStones: number[];
        }>;
      }>;
    };
    expect(fixture.featureSchemaVersion).toBe(STAR_FEATURE_SCHEMA_VERSION);
    expect(fixture.featureSchemaHash).toBe(STAR_FEATURE_SCHEMA_HASH);
    expect(fixture.rulesHash).toBe(STAR_RULES_HASH);
    expect(fixture.nodeFeatureNames).toEqual([...STAR_NODE_FEATURE_NAMES]);
    expect(fixture.globalFeatureNames).toEqual([...STAR_GLOBAL_FEATURE_NAMES]);
    expect(fixture.positions.length).toBeGreaterThanOrEqual(30);
    const sorted = (nodes: number[]) => [...nodes].sort((a, b) => a - b);
    for (const position of fixture.positions) {
      const game = conformance.games.find((candidate) => candidate.id === position.game);
      expect(game).toBeDefined();
      const state = game!.states[position.stateIndex];
      const encoded = encodeStarFeatures({
        rings: game!.config.rings,
        stones: state.stones,
        toMove: state.toMove,
        movesLeft: state.movesLeft,
        opening: state.opening,
        terminal: state.over,
        mode: game!.config.mode,
        handicap: game!.config.handicap,
        pie: game!.config.pieRule,
        swapAvailable: state.canSwap,
        swapped: state.swapped,
        history: {
          currentTurn: sorted(state.currentTurnMoves),
          previousTurn: sorted(state.previousTurnMoves),
          ownPreviousTurn: sorted(state.ownPreviousTurnMoves),
          handicapStones: sorted(state.handicapStones),
        },
      });
      const label = `${position.game}#${position.stateIndex}`;
      expect(encoded.nodeFeatures.length, label).toBe(position.nodeFeatures.length);
      for (let index = 0; index < position.nodeFeatures.length; index++) {
        expect(
          Math.abs(encoded.nodeFeatures[index] - position.nodeFeatures[index]),
          `${label} node feature ${index}`,
        ).toBeLessThan(1e-6);
      }
      for (let index = 0; index < position.globalFeatures.length; index++) {
        expect(
          Math.abs(encoded.globalFeatures[index] - position.globalFeatures[index]),
          `${label} global feature ${index}`,
        ).toBeLessThan(1e-6);
      }
      expect(Array.from(encoded.legalActionMask), label).toEqual(position.legalActionMask);
      expect(Array.from(encoded.rings, Number)).toEqual([position.rings]);
    }
  });
});
