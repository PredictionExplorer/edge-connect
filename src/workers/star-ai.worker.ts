import type * as Ort from 'onnxruntime-web';
import { getBoard } from '@/lib/star/board';
import { STAR_RULES_HASH, STAR_RULES_SCHEMA_ID } from '@/lib/star/rules';
import { StarAiError, asStarAiError } from '@/lib/star/ai/errors';
import {
  STAR_GLOBAL_FEATURE_DIM,
  STAR_MODEL_INPUT_NAMES,
  STAR_MODEL_OUTPUT_NAMES,
  STAR_NODE_FEATURE_DIM,
  actionCodeToModelIndex,
  encodeStarFeatures,
  float16ToFloat32Array,
  float32ToFloat16Array,
} from '@/lib/star/ai/features';
import {
  STAR_BROWSER_MODEL_MANIFEST_PATH,
  parseStarBrowserModelManifest,
  type StarBrowserModelManifest,
} from '@/lib/star/ai/manifest';
import {
  codeToAction,
  makeAiResponse,
  type StarAiRequest,
  type StarAiSemanticState,
} from '@/lib/star/ai/protocol';
import {
  parseWorkerCommand,
  workerErrorEvent,
  type StarAiWorkerCommand,
  type StarAiWorkerEvent,
} from '@/lib/star/ai/worker-protocol';

interface WorkerScope {
  addEventListener(type: 'message', listener: (event: MessageEvent<unknown>) => void): void;
  postMessage(message: StarAiWorkerEvent): void;
}

export interface WasmState {
  readonly to_move: number;
  readonly moves_left: number;
  readonly terminal: boolean;
  apply(action: number): void;
  zero_bits(): BigUint64Array;
  one_bits(): BigUint64Array;
  legal_actions(): Int32Array;
  hash64(): bigint;
  free?(): void;
}

interface WasmStateConstructor {
  new (rings: number): WasmState;
  rules_hash_tag(): string;
  rules_schema(): string;
}

interface WasmSearchTree {
  root_actions(): Int32Array;
  root_token(): bigint;
  initialize_root(token: bigint, value: number, policyLogits: Float32Array): void;
  start(rootAction: number): boolean;
  pending_state(): WasmState;
  pending_actions(): Int32Array;
  pending_token(): bigint;
  finish(token: bigint, value: number, policyLogits: Float32Array): void;
  actions(): Int32Array;
  visits(): Uint32Array;
  completed_q(): Float32Array;
  free?(): void;
}

interface WasmSearchTreeConstructor {
  new (state: WasmState, cVisit: number, cScale: number): WasmSearchTree;
}

interface WasmGumbel {
  next(completedQ: Float32Array, visits: Uint32Array): number;
  record(candidate: number): void;
  done(): boolean;
  selected(completedQ: Float32Array, visits: Uint32Array): number;
  free?(): void;
}

interface WasmGumbelConstructor {
  new (
    logits: Float32Array,
    simulations: number,
    maxConsidered: number,
    cVisit: number,
    cScale: number,
    seed: bigint,
  ): WasmGumbel;
}

interface StarWasmModule {
  default(input?: string | URL | BufferSource): Promise<unknown>;
  WasmState: WasmStateConstructor;
  WasmSearchTree: WasmSearchTreeConstructor;
  WasmGumbel: WasmGumbelConstructor;
}

interface LocalRuntime {
  manifest: StarBrowserModelManifest;
  ort: typeof Ort;
  session: Ort.InferenceSession;
  wasm: StarWasmModule;
}

interface Evaluation {
  value: number;
  logits: Float32Array;
}

const scope = globalThis as unknown as WorkerScope;
const cancelled = new Set<string>();
const knownTasks = new Set<string>();
const taskControllers = new Map<string, AbortController>();
let runtimePromise: Promise<LocalRuntime> | null = null;
let queue = Promise.resolve();

export function arraysEqual(left: ArrayLike<number>, right: ArrayLike<number>): boolean {
  if (left.length !== right.length) return false;
  for (let index = 0; index < left.length; index++) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}

function ensureNotCancelled(taskId: string): void {
  if (cancelled.has(taskId)) {
    throw new StarAiError('cancelled', 'Local AI request cancelled.');
  }
}

async function fetchJson(url: string, signal: AbortSignal): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(url, { cache: 'no-cache', signal });
  } catch (error) {
    throw new StarAiError('unavailable', 'Local AI assets could not be loaded.', true, error);
  }
  if (!response.ok) {
    throw new StarAiError(
      'unavailable',
      response.status === 404
        ? 'Local AI model is not installed.'
        : `Local AI manifest returned HTTP ${response.status}.`,
      response.status >= 500,
    );
  }
  try {
    return await response.json();
  } catch (error) {
    throw new StarAiError('unavailable', 'Local AI model manifest is invalid.', false, error);
  }
}

async function fetchBytes(
  url: string,
  label: string,
  signal: AbortSignal,
): Promise<ArrayBuffer> {
  let response: Response;
  try {
    response = await fetch(url, { cache: 'force-cache', signal });
  } catch (error) {
    throw new StarAiError('unavailable', `${label} could not be loaded.`, true, error);
  }
  if (!response.ok) {
    throw new StarAiError(
      'unavailable',
      `${label} is unavailable (HTTP ${response.status}).`,
      response.status >= 500,
    );
  }
  return response.arrayBuffer();
}

async function sha256(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  const bytes = new Uint8Array(digest);
  return `sha256:${Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('')}`;
}

async function importWasm(
  manifest: StarBrowserModelManifest,
  signal: AbortSignal,
): Promise<StarWasmModule> {
  let wasmModule: StarWasmModule;
  try {
    wasmModule = (await import(
      /* webpackIgnore: true */
      /* turbopackIgnore: true */
      manifest.wasm.moduleUrl
    )) as unknown as StarWasmModule;
    const binary = await fetchBytes(
      manifest.wasm.binaryUrl,
      'Local AI WASM binary',
      signal,
    );
    await wasmModule.default(binary);
  } catch (error) {
    throw new StarAiError(
      'unavailable',
      'Local AI WASM package is not installed. Run npm run build:star-wasm.',
      false,
      error,
    );
  }
  if (
    typeof wasmModule.WasmState !== 'function' ||
    typeof wasmModule.WasmSearchTree !== 'function' ||
    typeof wasmModule.WasmGumbel !== 'function' ||
    wasmModule.WasmState.rules_hash_tag() !== STAR_RULES_HASH ||
    wasmModule.WasmState.rules_schema() !== STAR_RULES_SCHEMA_ID
  ) {
    throw new StarAiError('unavailable', 'Local AI WASM rules are incompatible.');
  }
  return wasmModule;
}

function sameNames(actual: readonly string[], expected: readonly string[]): boolean {
  return (
    actual.length === expected.length &&
    actual.every((name, index) => name === expected[index])
  );
}

export function tensorMetadataMatches(
  metadata: Ort.InferenceSession.ValueMetadata,
  type: Ort.Tensor.Type,
  rank: number,
  lastDimension?: number,
): boolean {
  return (
    metadata.isTensor &&
    metadata.type === type &&
    metadata.shape.length === rank &&
    (lastDimension === undefined || metadata.shape[rank - 1] === lastDimension)
  );
}

export function hasExpectedOnnxSchema(session: Ort.InferenceSession): boolean {
  const inputSchema = [
    ['float16', 3, STAR_NODE_FEATURE_DIM],
    ['float16', 2, STAR_GLOBAL_FEATURE_DIM],
    ['int64', 3],
    ['bool', 3],
    ['int64', 3],
    ['bool', 2],
    ['bool', 2],
  ] as const;
  const outputSchema = [
    ['float16', 2],
    ['float16', 2, 2],
    ['float16', 2, 303],
    ['float16', 3, 3],
    ['float16', 2],
    ['float16', 2],
  ] as const;
  return (
    sameNames(session.inputNames, STAR_MODEL_INPUT_NAMES) &&
    sameNames(session.outputNames, STAR_MODEL_OUTPUT_NAMES) &&
    session.inputMetadata.length === inputSchema.length &&
    session.outputMetadata.length === outputSchema.length &&
    inputSchema.every(([type, rank, last], index) =>
      tensorMetadataMatches(session.inputMetadata[index], type, rank, last),
    ) &&
    outputSchema.every(([type, rank, last], index) =>
      tensorMetadataMatches(session.outputMetadata[index], type, rank, last),
    )
  );
}

async function createSession(
  ort: typeof Ort,
  model: ArrayBuffer,
): Promise<Ort.InferenceSession> {
  ort.env.wasm.numThreads = 1;
  let webGpuFailure: unknown;
  if ('gpu' in navigator) {
    try {
      return await ort.InferenceSession.create(model, {
        executionProviders: ['webgpu'],
        executionMode: 'sequential',
        graphOptimizationLevel: 'all',
      });
    } catch (error) {
      webGpuFailure = error;
    }
  }
  try {
    return await ort.InferenceSession.create(model, {
      executionProviders: ['wasm'],
      executionMode: 'sequential',
      graphOptimizationLevel: 'all',
    });
  } catch (error) {
    throw new StarAiError(
      'unavailable',
      'Local AI model is unsupported by WebGPU and WASM.',
      false,
      webGpuFailure ?? error,
    );
  }
}

async function loadRuntime(signal: AbortSignal): Promise<LocalRuntime> {
  const manifest = parseStarBrowserModelManifest(
    await fetchJson(STAR_BROWSER_MODEL_MANIFEST_PATH, signal),
  );
  const [wasm, model] = await Promise.all([
    importWasm(manifest, signal),
    fetchBytes(manifest.model.url, 'Local AI ONNX model', signal),
  ]);
  if (model.byteLength !== manifest.model.bytes) {
    throw new StarAiError('unavailable', 'Local AI model size does not match its manifest.');
  }
  if ((await sha256(model)) !== manifest.model.sha256) {
    throw new StarAiError('unavailable', 'Local AI model checksum does not match its manifest.');
  }

  const ort = await import('onnxruntime-web/webgpu');
  const session = await createSession(ort, model);
  if (!hasExpectedOnnxSchema(session)) {
    await session.release();
    throw new StarAiError('unavailable', 'Local AI ONNX schema is incompatible.');
  }
  return { manifest, ort, session, wasm };
}

function getRuntime(signal: AbortSignal): Promise<LocalRuntime> {
  if (!runtimePromise) {
    runtimePromise = loadRuntime(signal).catch((error) => {
      runtimePromise = null;
      throw error;
    });
  }
  return runtimePromise;
}

export function stonesFromWasm(state: WasmState, nodeCount: number): number[] {
  const stones = new Array<number>(nodeCount).fill(-1);
  const players = [state.zero_bits(), state.one_bits()];
  for (let player = 0; player < 2; player++) {
    const words = players[player];
    for (let wordIndex = 0; wordIndex < words.length; wordIndex++) {
      const word = words[wordIndex];
      const first = wordIndex * 64;
      const valid = Math.min(64, Math.max(0, nodeCount - first));
      for (let bit = 0; bit < valid; bit++) {
        if ((word & (BigInt(1) << BigInt(bit))) !== BigInt(0)) {
          const node = first + bit;
          if (stones[node] !== -1) {
            throw new StarAiError('protocol', 'WASM state contains overlapping stones.');
          }
          stones[node] = player;
        }
      }
      if (valid < 64 && (word >> BigInt(valid)) !== BigInt(0)) {
        throw new StarAiError('protocol', 'WASM state contains off-board stones.');
      }
    }
  }
  return stones;
}

export function semanticFromWasm(rings: number, state: WasmState): StarAiSemanticState {
  const nodeCount = getBoard(rings).n;
  const stones = stonesFromWasm(state, nodeCount);
  const occupied = stones.reduce((count, stone) => count + Number(stone !== -1), 0);
  return {
    rings,
    stones,
    toMove: state.to_move as 0 | 1,
    movesLeft: state.moves_left,
    opening:
      occupied === 0 &&
      state.to_move === 0 &&
      state.moves_left === 1 &&
      !state.terminal,
    terminal: state.terminal,
  };
}

export function replayAndVerify(request: StarAiRequest, wasm: StarWasmModule): WasmState {
  const state = new wasm.WasmState(request.state.rings);
  try {
    for (const action of request.actionLog) state.apply(action);
    const semantic = semanticFromWasm(request.state.rings, state);
    if (
      semantic.toMove !== request.state.toMove ||
      semantic.movesLeft !== request.state.movesLeft ||
      semantic.opening !== request.state.opening ||
      semantic.terminal !== request.state.terminal ||
      !arraysEqual(semantic.stones, request.state.stones) ||
      !arraysEqual(state.legal_actions(), request.legalActions) ||
      `zobrist64:${state.hash64().toString(16).padStart(16, '0')}` !== request.stateHash
    ) {
      throw new StarAiError('protocol', 'WASM replay disagrees with the AI request.');
    }
    return state;
  } catch (error) {
    state.free?.();
    throw error;
  }
}

function tensorFeeds(runtime: LocalRuntime, semantic: StarAiSemanticState) {
  const encoded = encodeStarFeatures(semantic);
  const { Tensor } = runtime.ort;
  return {
    node_features: new Tensor(
      'float16',
      float32ToFloat16Array(encoded.nodeFeatures),
      [1, encoded.nodeCount, STAR_NODE_FEATURE_DIM],
    ),
    global_features: new Tensor(
      'float16',
      float32ToFloat16Array(encoded.globalFeatures),
      [1, STAR_GLOBAL_FEATURE_DIM],
    ),
    neighbor_index: new Tensor(
      'int64',
      encoded.neighborIndex,
      [1, encoded.nodeCount, encoded.maxDegree],
    ),
    neighbor_mask: new Tensor(
      'bool',
      encoded.neighborMask,
      [1, encoded.nodeCount, encoded.maxDegree],
    ),
    neighbor_edge_type: new Tensor(
      'int64',
      encoded.neighborEdgeType,
      [1, encoded.nodeCount, encoded.maxDegree],
    ),
    node_mask: new Tensor('bool', encoded.nodeMask, [1, encoded.nodeCount]),
    legal_action_mask: new Tensor(
      'bool',
      encoded.legalActionMask,
      [1, encoded.nodeCount],
    ),
  };
}

export function finiteFloatData(
  value: Ort.OnnxValue | undefined,
  name: string,
): Float32Array {
  if (!value || !('data' in value)) {
    throw new StarAiError('protocol', `ONNX output ${name} is invalid.`);
  }
  const data = value.data as unknown;
  let decoded: Float32Array;
  if (data instanceof Uint16Array) {
    decoded = float16ToFloat32Array(data);
  } else {
    const Float16ArrayConstructor = (
      globalThis as typeof globalThis & {
        Float16Array?: new (
          buffer: ArrayBufferLike,
          byteOffset?: number,
          length?: number,
        ) => ArrayLike<number>;
      }
    ).Float16Array;
    if (!Float16ArrayConstructor || !(data instanceof Float16ArrayConstructor)) {
      throw new StarAiError('protocol', `ONNX output ${name} is not FP16.`);
    }
    decoded = Float32Array.from(data as ArrayLike<number>);
  }
  if (Array.from(decoded).some((item) => !Number.isFinite(item))) {
    throw new StarAiError('protocol', `ONNX output ${name} contains non-finite values.`);
  }
  return decoded;
}

export function outcomeValue(logits: Float32Array): number {
  if (logits.length !== 2) {
    throw new StarAiError('protocol', 'ONNX outcome output must contain two logits.');
  }
  const maximum = Math.max(...logits);
  const probabilities = Array.from(logits, (logit) => Math.exp(logit - maximum));
  const total = probabilities[0] + probabilities[1];
  if (!Number.isFinite(total) || total <= 0) {
    throw new StarAiError('protocol', 'ONNX outcome output cannot be normalized.');
  }
  return (probabilities[1] - probabilities[0]) / total;
}

async function evaluate(
  runtime: LocalRuntime,
  semantic: StarAiSemanticState,
  legalActions: Int32Array,
): Promise<Evaluation> {
  const outputs = await runtime.session.run(tensorFeeds(runtime, semantic));
  const densePolicy = finiteFloatData(outputs.policy_logits, 'policy_logits');
  const outcome = finiteFloatData(outputs.outcome_logits, 'outcome_logits');
  const nodeCount = semantic.stones.length;
  if (densePolicy.length !== nodeCount) {
    throw new StarAiError('protocol', 'ONNX policy output has the wrong action layout.');
  }
  const logits = new Float32Array(legalActions.length);
  for (let index = 0; index < legalActions.length; index++) {
    const action = legalActions[index];
    const modelIndex = actionCodeToModelIndex(action, nodeCount);
    const logit = densePolicy[modelIndex];
    if (!Number.isFinite(logit)) {
      throw new StarAiError('protocol', 'ONNX policy contains a non-finite legal logit.');
    }
    logits[index] = logit;
  }
  return { value: outcomeValue(outcome), logits };
}

async function yieldToCancellation(taskId: string): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
  ensureNotCancelled(taskId);
}

async function chooseAction(
  taskId: string,
  request: StarAiRequest,
  signal: AbortSignal,
): Promise<number> {
  ensureNotCancelled(taskId);
  const runtime = await getRuntime(signal);
  ensureNotCancelled(taskId);
  const root = replayAndVerify(request, runtime.wasm);
  let tree: WasmSearchTree | null = null;
  let scheduler: WasmGumbel | null = null;
  try {
    tree = new runtime.wasm.WasmSearchTree(
      root,
      runtime.manifest.search.cVisit,
      runtime.manifest.search.cScale,
    );
    const rootActions = tree.root_actions();
    if (!arraysEqual(rootActions, request.legalActions)) {
      throw new StarAiError('protocol', 'WASM root action layout is incompatible.');
    }
    const rootToken = tree.root_token();
    const rootEvaluation = await evaluate(runtime, request.state, rootActions);
    ensureNotCancelled(taskId);
    if (tree.root_token() !== rootToken) {
      throw new StarAiError('stale', 'WASM root evaluation token changed.');
    }
    tree.initialize_root(rootToken, rootEvaluation.value, rootEvaluation.logits);

    scheduler = new runtime.wasm.WasmGumbel(
      rootEvaluation.logits,
      runtime.manifest.search.simulations,
      runtime.manifest.search.maxConsidered,
      runtime.manifest.search.cVisit,
      runtime.manifest.search.cScale,
      root.hash64(),
    );
    let simulations = 0;
    while (!scheduler.done()) {
      ensureNotCancelled(taskId);
      const candidate = scheduler.next(tree.completed_q(), tree.visits());
      const actions = tree.actions();
      if (candidate < 0 || candidate >= actions.length) {
        throw new StarAiError('protocol', 'WASM Gumbel scheduler returned an invalid edge.');
      }
      const needsEvaluation = tree.start(actions[candidate]);
      if (needsEvaluation) {
        const token = tree.pending_token();
        const leaf = tree.pending_state();
        try {
          const leafActions = tree.pending_actions();
          const leafSemantic = semanticFromWasm(request.state.rings, leaf);
          const leafEvaluation = await evaluate(runtime, leafSemantic, leafActions);
          ensureNotCancelled(taskId);
          if (tree.pending_token() !== token) {
            throw new StarAiError('stale', 'WASM leaf evaluation token changed.');
          }
          tree.finish(token, leafEvaluation.value, leafEvaluation.logits);
        } finally {
          leaf.free?.();
        }
      }
      scheduler.record(candidate);
      simulations += 1;
      if (simulations % 8 === 0) await yieldToCancellation(taskId);
    }
    const selected = scheduler.selected(tree.completed_q(), tree.visits());
    const actions = tree.actions();
    if (selected < 0 || selected >= actions.length) {
      throw new StarAiError('protocol', 'WASM search selected an invalid edge.');
    }
    return actions[selected];
  } finally {
    scheduler?.free?.();
    tree?.free?.();
    root.free?.();
  }
}

async function runChoose(command: Extract<StarAiWorkerCommand, { type: 'choose' }>) {
  const taskController = taskControllers.get(command.taskId);
  try {
    if (!taskController) throw new StarAiError('cancelled', 'Local AI request cancelled.');
    const actionCode = await chooseAction(
      command.taskId,
      command.request,
      taskController.signal,
    );
    ensureNotCancelled(command.taskId);
    scope.postMessage({
      type: 'result',
      taskId: command.taskId,
      response: makeAiResponse(command.request, codeToAction(actionCode)),
    });
  } catch (error) {
    const aiError = asStarAiError(error);
    if (aiError.code !== 'cancelled') {
      scope.postMessage(workerErrorEvent(command.taskId, aiError));
    }
  } finally {
    knownTasks.delete(command.taskId);
    cancelled.delete(command.taskId);
    taskControllers.delete(command.taskId);
  }
}

scope.addEventListener('message', (event) => {
  let command: StarAiWorkerCommand;
  try {
    command = parseWorkerCommand(event.data);
  } catch (error) {
    const taskId =
      typeof event.data === 'object' &&
      event.data !== null &&
      'taskId' in event.data &&
      typeof event.data.taskId === 'string'
        ? event.data.taskId
        : 'invalid';
    scope.postMessage(workerErrorEvent(taskId, error));
    return;
  }

  if (command.type === 'cancel') {
    if (knownTasks.has(command.taskId)) {
      cancelled.add(command.taskId);
      taskControllers.get(command.taskId)?.abort();
    }
    return;
  }
  if (knownTasks.has(command.taskId)) {
    scope.postMessage(
      workerErrorEvent(
        command.taskId,
        new StarAiError('protocol', 'Duplicate local AI worker task id.'),
      ),
    );
    return;
  }

  knownTasks.add(command.taskId);
  taskControllers.set(command.taskId, new AbortController());
  queue = queue.then(() => runChoose(command)).catch(() => {
    // runChoose contains its own typed error boundary; keep the queue usable.
  });
});

scope.postMessage({ type: 'ready', protocolVersion: 2 });
