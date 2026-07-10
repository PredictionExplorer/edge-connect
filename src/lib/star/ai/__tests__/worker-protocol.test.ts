import { describe, expect, it } from 'vitest';
import {
  STAR_MODEL_INPUT_NAMES,
  STAR_MODEL_OUTPUT_NAMES,
} from '../features';
import {
  STAR_BROWSER_MODEL_MANIFEST_SCHEMA_ID,
  STAR_WASM_BINARY_PATH,
  STAR_WASM_MODULE_PATH,
  parseStarBrowserModelManifest,
} from '../manifest';
import { STAR_FEATURE_SCHEMA_HASH, buildAiRequest } from '../protocol';
import { parseWorkerCommand, parseWorkerEvent } from '../worker-protocol';

const request = buildAiRequest(
  {
    rings: 3,
    mode: 'double',
    pieRule: false,
    playerNames: ['A', 'B'],
  },
  [],
  'worker-task',
);

const manifest = {
  format: STAR_BROWSER_MODEL_MANIFEST_SCHEMA_ID,
  schema_version: 1,
  model_version: 'browser-smoke-v1',
  created_ns: 1_700_000_000_000_000_000,
  weights: 'ema',
  rules: {
    schema_id: 'edgeconnect.star.rules.v1',
    hash: 'fnv1a64:cdb34fb02be82843',
    mode: 'double',
    pie_rule: false,
    rings: { minimum: 3, maximum: 12 },
  },
  features: {
    schema_id: 'edgeconnect.star.model-features.external.v1',
    version: 2,
    hash: STAR_FEATURE_SCHEMA_HASH,
    node_feature_count: 15,
    global_feature_count: 18,
  },
  actions: { schema_id: 'edgeconnect.star.action-layout.nodes-then-pass.v1' },
  architecture: {
    name: 'GraphResTNet',
    all_size: true,
    parameter_count: 12_345,
    config: {
      node_feature_dim: 15,
      global_feature_dim: 18,
      width: 64,
      rrt_groups: 5,
      attention_heads: 8,
      kv_heads: 2,
      bottleneck_ratio: 0.5,
      ff_multiplier: 2,
      dropout: 0,
      rms_norm_eps: 0.000001,
      score_margin_min: -181,
      score_margin_max: 181,
      soft_policy_temperature: 4,
    },
  },
  precision: 'float16',
  artifacts: {
    onnx: {
      file: 'browser-smoke-v1.fp16.onnx',
      sha256: 'a'.repeat(64),
      bytes: 123_456,
      opset: 18,
    },
    checkpoint: {
      file: 'browser-smoke-v1.pt',
      sha256: 'b'.repeat(64),
      bytes: 234_567,
    },
  },
  tensors: {
    inputs: {
      node_features: { dtype: 'float16', shape: ['batch', 'nodes', 15] },
      global_features: { dtype: 'float16', shape: ['batch', 18] },
      neighbor_index: { dtype: 'int64', shape: ['batch', 'nodes', 'degree'] },
      neighbor_mask: { dtype: 'bool', shape: ['batch', 'nodes', 'degree'] },
      neighbor_edge_type: { dtype: 'int64', shape: ['batch', 'nodes', 'degree'] },
      node_mask: { dtype: 'bool', shape: ['batch', 'nodes'] },
      legal_action_mask: { dtype: 'bool', shape: ['batch', 'nodes + 1'] },
    },
    outputs: {
      policy_logits: { dtype: 'float16', shape: ['batch', 'nodes + 1'] },
      wdl_logits: { dtype: 'float16', shape: ['batch', 3] },
      score_margin_logits: { dtype: 'float16', shape: ['batch', 363] },
      ownership_logits: { dtype: 'float16', shape: ['batch', 'nodes', 3] },
      alive_logits: { dtype: 'float16', shape: ['batch', 'nodes'] },
      soft_policy_logits: { dtype: 'float16', shape: ['batch', 'nodes + 1'] },
    },
  },
  recommended_local_search: {
    simulations: 64,
    max_considered: 16,
    c_visit: 50,
    c_scale: 1,
  },
  training: {
    steps: 10_000,
    replay_samples: 1_000_000,
    teacher_model_version: 'teacher-v1',
    teacher_logit_kl: true,
  },
};

describe('local worker protocol', () => {
  it('round-trips a typed choose command and cancellation', () => {
    expect(
      parseWorkerCommand({ type: 'choose', taskId: request.requestId, request }),
    ).toMatchObject({ type: 'choose', taskId: 'worker-task' });
    expect(parseWorkerCommand({ type: 'cancel', taskId: 'worker-task' })).toEqual({
      type: 'cancel',
      taskId: 'worker-task',
    });
  });

  it('rejects a semantic payload whose state hash was altered', () => {
    expect(() =>
      parseWorkerCommand({
        type: 'choose',
        taskId: 'worker-task',
        request: { ...request, stateHash: 'zobrist64:0000000000000000' },
      }),
    ).toThrow(/state hash/i);
  });

  it('parses structured worker errors without trusting arbitrary codes', () => {
    expect(parseWorkerEvent({ type: 'ready', protocolVersion: 1 })).toEqual({
      type: 'ready',
      protocolVersion: 1,
    });
    expect(
      parseWorkerEvent({
        type: 'error',
        taskId: 'worker-task',
        error: { code: 'unavailable', message: 'missing', retryable: false },
      }),
    ).toMatchObject({ type: 'error', error: { code: 'unavailable' } });
    expect(() =>
      parseWorkerEvent({
        type: 'error',
        taskId: 'worker-task',
        error: { code: 'anything', message: 'bad', retryable: false },
      }),
    ).toThrow(/invalid event/i);
  });

  it('accepts only fully pinned browser model manifests', () => {
    expect(parseStarBrowserModelManifest(manifest)).toMatchObject({
      rulesHash: 'fnv1a64:cdb34fb02be82843',
      featureSchemaHash: '59a7da1c00bac4d2',
      weights: 'ema',
      wasm: {
        moduleUrl: STAR_WASM_MODULE_PATH,
        binaryUrl: STAR_WASM_BINARY_PATH,
      },
      model: {
        precision: 'float16',
        url: '/models/star/browser-smoke-v1.fp16.onnx',
        sha256: `sha256:${'a'.repeat(64)}`,
        inputs: STAR_MODEL_INPUT_NAMES,
        outputs: STAR_MODEL_OUTPUT_NAMES,
      },
    });
    expect(() =>
      parseStarBrowserModelManifest({
        ...manifest,
        rules: { ...manifest.rules, hash: 'fnv1a64:0000000000000000' },
      }),
    ).toThrow(/do not match/i);
    expect(() =>
      parseStarBrowserModelManifest({
        ...manifest,
        artifacts: {
          ...manifest.artifacts,
          onnx: { ...manifest.artifacts.onnx, sha256: 'unverified' },
        },
      }),
    ).toThrow(/fields are invalid/i);
  });
});
