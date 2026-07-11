import { describe, expect, it } from 'vitest';
import { getBoard } from '../../board';
import type { GameConfig } from '../../game';
import {
  STAR_GLOBAL_FEATURE_DIM,
  STAR_NODE_FEATURE_DIM,
  actionCodeToModelIndex,
  encodeStarFeatures,
  float16BitsToNumber,
  float16ToFloat32Array,
  float32ToFloat16Array,
  modelIndexToActionCode,
  numberToFloat16Bits,
} from '../features';
import { buildAiRequest } from '../protocol';

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

    expect(STAR_NODE_FEATURE_DIM).toBe(15);
    expect(STAR_GLOBAL_FEATURE_DIM).toBe(17);
    expect(encoded.nodeFeatures).toHaveLength(board.n * 15);
    expect(encoded.globalFeatures).toHaveLength(17);
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

    expect(Array.from(encoded.nodeFeatures.slice(0, 15))).toEqual([
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
    ]);
    expect(Array.from(encoded.globalFeatures)).toEqual([
      Math.fround(4 / 10),
      0,
      0,
      0,
      1 / 2,
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
