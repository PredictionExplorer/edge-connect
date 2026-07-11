import { beforeAll, describe, expect, it, vi } from 'vitest';
import type * as Ort from 'onnxruntime-web';
import type { WasmState } from '@/workers/star-ai.worker';
import {
  STAR_GLOBAL_FEATURE_DIM,
  STAR_MODEL_INPUT_NAMES,
  STAR_MODEL_OUTPUT_NAMES,
  STAR_NODE_FEATURE_DIM,
  float32ToFloat16Array,
} from '../features';
import { buildAiRequest } from '../protocol';

let registeredHandler: ((event: MessageEvent<unknown>) => void) | undefined;
let readyEvent: unknown;
const postMessage = vi.fn((message: unknown) => {
  readyEvent = message;
});
const addEventListener = vi.fn(
  (_type: string, listener: (event: MessageEvent<unknown>) => void) => {
    registeredHandler = listener;
  },
);
let runtime: typeof import('@/workers/star-ai.worker');

beforeAll(async () => {
  vi.stubGlobal('postMessage', postMessage);
  vi.stubGlobal('addEventListener', addEventListener);
  runtime = await import('@/workers/star-ai.worker');
});

describe('local worker runtime contract', () => {
  it('registers exactly one message handler and announces readiness', () => {
    expect(registeredHandler).toEqual(expect.any(Function));
    expect(readyEvent).toEqual({ type: 'ready', protocolVersion: 2 });
  });

  it('decodes bitboards while rejecting overlap and off-board bits', () => {
    const state = {
      zero_bits: () => new BigUint64Array([BigInt(1), ...Array(6).fill(BigInt(0))]),
      one_bits: () => new BigUint64Array([BigInt(2), ...Array(6).fill(BigInt(0))]),
    } as WasmState;
    expect(runtime.stonesFromWasm(state, 50).slice(0, 3)).toEqual([0, 1, -1]);

    const overlap = {
      ...state,
      one_bits: () => new BigUint64Array([BigInt(1), ...Array(6).fill(BigInt(0))]),
    } as WasmState;
    expect(() => runtime.stonesFromWasm(overlap, 50)).toThrow(/overlapping stones/i);

    const offBoard = {
      ...state,
      zero_bits: () => new BigUint64Array([BigInt(1) << BigInt(50), ...Array(6).fill(BigInt(0))]),
      one_bits: () => new BigUint64Array(7),
    } as WasmState;
    expect(() => runtime.stonesFromWasm(offBoard, 50)).toThrow(/off-board stones/i);
  });

  it('validates ONNX names, tensor types, ranks, and fixed head dimensions', () => {
    const metadata = (
      type: Ort.Tensor.Type,
      shape: readonly number[],
    ): Ort.InferenceSession.ValueMetadata =>
      ({ isTensor: true, type, shape }) as Ort.InferenceSession.ValueMetadata;
    const session = {
      inputNames: [...STAR_MODEL_INPUT_NAMES],
      outputNames: [...STAR_MODEL_OUTPUT_NAMES],
      inputMetadata: [
        metadata('float16', [1, 50, STAR_NODE_FEATURE_DIM]),
        metadata('float16', [1, STAR_GLOBAL_FEATURE_DIM]),
        metadata('int64', [1, 50, 6]),
        metadata('bool', [1, 50, 6]),
        metadata('int64', [1, 50, 6]),
        metadata('bool', [1, 50]),
        metadata('bool', [1, 50]),
      ],
      outputMetadata: [
        metadata('float16', [1, 50]),
        metadata('float16', [1, 2]),
        metadata('float16', [1, 303]),
        metadata('float16', [1, 50, 3]),
        metadata('float16', [1, 50]),
        metadata('float16', [1, 50]),
      ],
    } as unknown as Ort.InferenceSession;
    expect(runtime.hasExpectedOnnxSchema(session)).toBe(true);
    const invalid = {
      ...session,
      outputNames: ['wrong', ...session.outputNames.slice(1)],
    } as unknown as Ort.InferenceSession;
    expect(runtime.hasExpectedOnnxSchema(invalid)).toBe(false);
  });

  it('decodes finite FP16 outputs and normalizes binary outcome logits', () => {
    const encoded = float32ToFloat16Array(new Float32Array([-1, 0, 2]));
    const decoded = runtime.finiteFloatData(
      { data: encoded } as unknown as Ort.OnnxValue,
      'policy',
    );
    expect(Array.from(decoded)).toEqual([-1, 0, 2]);
    expect(runtime.outcomeValue(new Float32Array([0, 0]))).toBe(0);
    expect(runtime.outcomeValue(new Float32Array([-10, 10]))).toBeGreaterThan(0.99);
    expect(() => runtime.outcomeValue(new Float32Array([0, 0, 0]))).toThrow(
      /two logits/i,
    );

    const nonFinite = float32ToFloat16Array(new Float32Array([Number.NaN]));
    expect(() =>
      runtime.finiteFloatData(
        { data: nonFinite } as unknown as Ort.OnnxValue,
        'policy',
      ),
    ).toThrow(/non-finite/i);
  });

  it('replays and verifies semantic identity before local search', () => {
    const request = buildAiRequest(
      {
        rings: 4,
        mode: 'double',
        pieRule: false,
        playerNames: ['A', 'B'],
      },
      [],
    );
    const hash = BigInt(`0x${request.stateHash.slice('zobrist64:'.length)}`);
    class FakeState {
      readonly to_move = request.state.toMove;
      readonly moves_left = request.state.movesLeft;
      readonly terminal = request.state.terminal;
      apply = vi.fn();
      zero_bits = () => new BigUint64Array(7);
      one_bits = () => new BigUint64Array(7);
      legal_actions = () => Int32Array.from(request.legalActions);
      hash64 = () => hash;
    }
    const wasm = { WasmState: FakeState };
    expect(runtime.replayAndVerify(request, wasm as never)).toBeInstanceOf(FakeState);

    class BadState extends FakeState {
      legal_actions = () => Int32Array.from([-1]);
    }
    expect(() =>
      runtime.replayAndVerify(request, { WasmState: BadState } as never),
    ).toThrow(/disagrees with the AI request/i);
  });
});

