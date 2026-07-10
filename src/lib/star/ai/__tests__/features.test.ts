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
  rings: 3,
  mode: 'double',
  pieRule: false,
  playerNames: ['A', 'B'],
};

describe('schema-v2 browser features', () => {
  it('uses the exact node/global dimensions and nodes-then-pass action layout', () => {
    const state = buildAiRequest(config, [], 'features').state;
    const encoded = encodeStarFeatures(state);
    const board = getBoard(3);

    expect(STAR_NODE_FEATURE_DIM).toBe(15);
    expect(STAR_GLOBAL_FEATURE_DIM).toBe(18);
    expect(encoded.nodeFeatures).toHaveLength(board.n * 15);
    expect(encoded.globalFeatures).toHaveLength(18);
    expect(encoded.neighborIndex).toHaveLength(board.n * encoded.maxDegree);
    expect(encoded.legalActionMask).toHaveLength(board.n + 1);
    expect(Array.from(encoded.legalActionMask)).toEqual(new Array(board.n + 1).fill(1));
    expect(actionCodeToModelIndex(-1, board.n)).toBe(board.n);
    expect(modelIndexToActionCode(board.n, board.n)).toBe(-1);
    expect(actionCodeToModelIndex(7, board.n)).toBe(7);
  });

  it('matches Python opening features and topology edge classes', () => {
    const state = buildAiRequest(config, [], 'opening-features').state;
    const encoded = encodeStarFeatures(state);
    const board = getBoard(3);
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
      Math.fround(1 / 3),
      0,
      Math.fround(degree / encoded.maxDegree),
      1,
      1,
    ]);
    expect(Array.from(encoded.globalFeatures)).toEqual([
      3 / 12,
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

  it('converts schema-v2 floating features to browser FP16 tensors', () => {
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
